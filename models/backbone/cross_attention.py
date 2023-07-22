import torch
from torch import nn
from timm.models.layers import trunc_normal_


class FD_Block(nn.Module):
    def __init__(self, extractor_id, extractor_po, seq_len, hid_dim, fdca_num=0, use_bias=False, mode='', **kwargs):
        super(FD_Block, self).__init__()
        self.extractor_id = extractor_id
        self.extractor_po = extractor_po
        self.fdca_num = 0 if mode == '' else len(mode)
        print('FD_Block: fdca_num=', self.fdca_num, ' mode=', mode)
        for i in range(self.fdca_num):
            setattr(self, 'fdca' + str(i), FDCA(seq_len, hid_dim, use_bias=use_bias, mode=mode[i]))

    def forward(self, x_id, x_po):
        x_id, x_po = self.extractor_id(x_id), self.extractor_po(x_po)
        for i in range(self.fdca_num):
            x_id, x_po = getattr(self, 'fdca'+str(i))(x_id, x_po)
        return x_id, x_po


class FDCA(nn.Module):
    def __init__(self, seq_len, hid_dim, n_heads=2, dropout=0.1, use_bias=False, mode='p', **kwargs):
        super().__init__()
        self.mode = mode
        if self.mode in ['p', 'position']:
            self.seq_len = seq_len
            self.hid_dim = hid_dim
        elif self.mode in ['c', 'channel']:
            self.seq_len = hid_dim
            self.hid_dim = seq_len
        else:
            raise Exception('Invalid ca mode!')

        assert self.hid_dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = self.hid_dim // n_heads
        # self.pos_embed = nn.Parameter(torch.ones(1, seq_len, hid_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.hid_dim))
        trunc_normal_(self.pos_embed, std=.02)

        self.fc_q_id = nn.Linear(self.hid_dim, self.hid_dim, bias=use_bias)
        self.fc_k_id = nn.Linear(self.hid_dim, self.hid_dim, bias=use_bias)
        self.fc_v_id = nn.Linear(self.hid_dim, self.hid_dim, bias=use_bias)

        self.fc_q_cm = nn.Linear(self.hid_dim, self.hid_dim, bias=use_bias)
        self.fc_k_cm = nn.Linear(self.hid_dim, self.hid_dim, bias=use_bias)
        self.fc_v_cm = nn.Linear(self.hid_dim, self.hid_dim, bias=use_bias)

        self.dropout_id = nn.Dropout(dropout)
        self.dropout_cm = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_id, x_cm):
        batch_size, C, H, W = x_id.shape

        if self.mode in ['p', 'position']:
            # feature: C_dim (position attention)
            x_id0 = x_id.permute(0, 2, 3, 1).view(batch_size, H*W, C)
            x_cm0 = x_cm.permute(0, 2, 3, 1).view(batch_size, H*W, C)
        else:
            # feature: H*W_dim (channel attention)
            x_id0 = x_id.view(batch_size, C, H*W)
            x_cm0 = x_id.view(batch_size, C, H*W)

        x_id1 = x_id0 + self.pos_embed
        x_cm1 = x_cm0 + self.pos_embed

        # x_id: [batch size, seq len, hid dim]
        query_id, key_id, value_id = self.fc_q_id(x_id1), self.fc_k_id(x_id1), self.fc_v_id(x_id1)
        query_cm, key_cm, value_cm = self.fc_q_cm(x_cm1), self.fc_k_cm(x_cm1), self.fc_v_cm(x_cm1)

        o_id = x_id0 - self.farward_qkv(query_id, key_cm, value_id, attn_type='id')
        o_cm = x_cm0 - self.farward_qkv(query_cm, key_id, value_cm, attn_type='cm')

        if self.mode in ['p', 'position']:
            o_id = o_id.view(batch_size, H, W, C).permute(0, 3, 1, 2)
            o_cm = o_cm.view(batch_size, H, W, C).permute(0, 3, 1, 2)
        else:
            o_id = o_id.view(batch_size, C, H, W)
            o_cm = o_cm.view(batch_size, C, H, W)

        return o_id, o_cm

    def farward_qkv(self, query, key, value, attn_type):
        batch_size = query.shape[0]

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))
        attn1 = r_q1 @ r_k1.permute(0, 1, 3, 2)
        if self.scale.device != attn1.device:
            self.scale = self.scale.to(attn1.device)
        attn = attn1 / self.scale

        # attn = [batch size, n heads, query len, key len]
        attn = torch.softmax(attn, dim=-1)
        attn = getattr(self, 'dropout_' + attn_type)(attn)
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # weight1 = torch.matmul(attn, r_v1)
        if r_v1.device != attn.device:
            r_v1 = r_v1.to(attn.device)
        weight1 = attn @ r_v1
        x = weight1
        x = x.permute(0, 2, 1, 3).contiguous()  # x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)  # x = [batch size, query len, hid dim]
        return x


if __name__ == '__main__':
    batch_size = 64
    device = 'cuda'

    a = 32
    seq_len = 2*a*a
    hid_dim = 256

    # hid_dim = 2 * a * a
    # seq_len = 256

    x1 = torch.Tensor(batch_size, hid_dim, 2*a, a).to(device)
    x2 = torch.Tensor(batch_size, hid_dim, 2*a, a).to(device)
    model = FDCA(seq_len, hid_dim)
    print("{} model size: {:.5f}M".format('', sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("fc_q_id size: {:.5f}M".format(sum(p.numel() for p in model.fc_q_id.parameters()) / 1000000.0))
    # if torch.cuda.device_count() > 1:
    #     print("The model will be loaded onto multiple GPUs")
    #     model = nn.DataParallel(model)
    model.to(device)

    out1, out2 = model(x1, x2)
    print(out1.shape, out2.shape)
    pass
