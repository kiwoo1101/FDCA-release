from torch import nn
import copy

try:
    from cross_attention import FD_Block
    from resnet_nl import Bottleneck, model_urls, ResNetNL
    from resnet import ResNet
except:
    from .cross_attention import FD_Block
    from .resnet_nl import Bottleneck, model_urls, ResNetNL
    from .resnet import ResNet


__all__ = ['resnet50_nl_ca', 'resnet50_nl_resnet50']


class ResNetNL_ResNet(nn.Module):

    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3], non_layers=[0, 2, 3, 0], img_size=(256, 128),
                 fdca_mode='p_p_p_p_p', fdca_num=(0, 1, 1, 1, 1), url=None, root_path='/home/wuqi/', use_bias=False,
                 **kwargs):
        super(ResNetNL_ResNet, self).__init__()
        resnet_nl = ResNetNL(last_stride, block, layers, non_layers, url, root_path)
        resnet = ResNet(last_stride, block, layers, url, root_path)

        self.layer0 = nn.Sequential(resnet_nl.conv1, resnet_nl.bn1, resnet_nl.maxpool)
        self.layer1 = resnet_nl.block0
        self.layer2 = resnet_nl.block1
        self.layer3 = resnet_nl.block2
        self.layer4 = resnet_nl.block3

        # self.layer_po0 = copy.deepcopy(self.layer0)
        # self.layer_po1 = copy.deepcopy(self.layer1)
        # self.layer_po2 = copy.deepcopy(self.layer2)
        # self.layer_po3 = copy.deepcopy(self.layer3)
        # self.layer_po4 = copy.deepcopy(self.layer4)

        self.layer_po0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)
        self.layer_po1 = resnet.layer1
        self.layer_po2 = resnet.layer2
        self.layer_po3 = resnet.layer3
        self.layer_po4 = resnet.layer4

        h, w = img_size[0] // 4, img_size[1] // 4
        h_d = 64
        ex = block.expansion

        seq_len = [h*w, h*w, h*w//4, h*w//16, h*w//(16*last_stride**2)]
        hid_dim = [h_d, h_d*ex, h_d*ex*2, h_d*ex*4, h_d*ex*8]
        mode = fdca_mode.split('_')
        assert len(mode) >= 5

        for i in range(5):
            setattr(self, 'block'+str(i),
                    FD_Block(getattr(self, 'layer'+str(i)), getattr(self, 'layer_po'+str(i)), seq_len[i], hid_dim[i],
                             fdca_num=fdca_num[i], use_bias=use_bias, mode=mode[i])
                    )

        self.out_features = h_d * ex * 8
        assert self.out_features == resnet_nl.out_features

    def forward(self, x):
        x_id = x
        x_po = x.contiguous()

        for i in range(5):
            x_id, x_po = getattr(self, 'block' + str(i))(x_id, x_po)

        return x_id, x_po


class ResNetNL_ca(nn.Module):

    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3], non_layers=[0, 2, 3, 0], img_size=(256, 128),
                 fdca_mode='p_p_p_p_p', fdca_num=(0, 1, 1, 1, 1), url=None, root_path='/home/wuqi/', use_bias=False,
                 **kwargs):
        super(ResNetNL_ca, self).__init__()
        resnet_nl = ResNetNL(last_stride, block, layers, non_layers, url, root_path)

        self.layer0 = nn.Sequential(resnet_nl.conv1, resnet_nl.bn1, resnet_nl.maxpool)
        self.layer1 = resnet_nl.block0
        self.layer2 = resnet_nl.block1
        self.layer3 = resnet_nl.block2
        self.layer4 = resnet_nl.block3

        self.layer_po0 = copy.deepcopy(self.layer0)
        self.layer_po1 = copy.deepcopy(self.layer1)
        self.layer_po2 = copy.deepcopy(self.layer2)
        self.layer_po3 = copy.deepcopy(self.layer3)
        self.layer_po4 = copy.deepcopy(self.layer4)

        h, w = img_size[0] // 4, img_size[1] // 4
        h_d = 64
        ex = block.expansion

        seq_len = [h*w, h*w, h*w//4, h*w//16, h*w//(16*last_stride**2)]
        hid_dim = [h_d, h_d*ex, h_d*ex*2, h_d*ex*4, h_d*ex*8]
        mode = fdca_mode.split('_')
        assert len(mode) >= 5

        for i in range(5):
            setattr(self, 'block'+str(i),
                    FD_Block(getattr(self, 'layer'+str(i)), getattr(self, 'layer_po'+str(i)), seq_len[i], hid_dim[i],
                             fdca_num=fdca_num[i], use_bias=use_bias, mode=mode[i])
                    )

        self.out_features = h_d * ex * 8
        assert self.out_features == resnet_nl.out_features

    def forward(self, x):
        x_id = x
        x_po = x.contiguous()

        for i in range(5):
            x_id, x_po = getattr(self, 'block' + str(i))(x_id, x_po)

        return x_id, x_po


def resnet50_nl_ca(cfg):
    return ResNetNL_ca(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 4, 6, 3], non_layers=[0, 2, 3, 0],
                       img_size=cfg.INPUT.SIZE_TRAIN, fdca_mode=cfg.MODEL.FDCA_MODE, fdca_num=cfg.MODEL.FDCA_NUM,
                       url=model_urls['resnet50'], root_path=cfg.SYS.ROOT_PATH, use_bias=cfg.MODEL.CA_BIAS)


def resnet50_nl_resnet50(cfg):
    return ResNetNL_ResNet(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 4, 6, 3], non_layers=[0, 2, 3, 0],
                       img_size=cfg.INPUT.SIZE_TRAIN, fdca_mode=cfg.MODEL.FDCA_MODE, fdca_num=cfg.MODEL.FDCA_NUM,
                       url=model_urls['resnet50'], root_path=cfg.SYS.ROOT_PATH, use_bias=cfg.MODEL.CA_BIAS)


if __name__ == '__main__':
    import torch
    from yacs.config import CfgNode as CN
    cfg = CN()
    img_size = (256, 128)  #
    cfg.SYS = CN()
    cfg.SYS.ROOT_PATH = '/home/wuqi/'
    cfg.MODEL = CN()
    cfg.MODEL.LAST_STRIDE = 1
    cfg.MODEL.FDCA_NUM = (0, 1, 1, 1, 1)
    cfg.MODEL.RESCA = False
    cfg.MODEL.FDCA_MODE = '_c_c_p_p'
    cfg.MODEL.CA_BIAS = False
    cfg.INPUT = CN()
    cfg.INPUT.SIZE_TRAIN = img_size

    model = resnet50_nl_ca(cfg)
    print("{} model size: {:.5f}M".format('', sum(p.numel() for p in model.parameters()) / 1000000.0))
    img = torch.Tensor(64, 3, img_size[0], img_size[1])
    x_id, x_po = model(img)
    print(x_id.shape, x_po.shape)
    pass
