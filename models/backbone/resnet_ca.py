from torch import nn
import copy

try:
    from cross_attention import FDCA
    from basemodel import Basemodel
    from resnet import BasicBlock, Bottleneck, model_urls, ResNet
except:
    from .cross_attention import FDCA
    from .basemodel import Basemodel
    from .resnet import BasicBlock, Bottleneck, model_urls, ResNet


__all__ = ['resnet18_ca', 'resnet34_ca', 'resnet50_ca', 'resnet101_ca', 'resnet152_ca']


class FD_Block(nn.Module):
    def __init__(self, extractor_id, extractor_po, fdca, fdca_num, resca=None):
        super(FD_Block, self).__init__()
        self.extractor_id = extractor_id
        self.extractor_po = extractor_po
        self.resca = resca
        self.fdca_num = fdca_num
        for i in range(fdca_num):
            setattr(self, 'fdca' + str(i), copy.deepcopy(fdca))
        if self.fdca_num>0 and self.resca:
            self.relu = nn.ReLU(inplace=True)
            self.resca_id = copy.deepcopy(fdca)
            self.resca_po = copy.deepcopy(fdca)

    def forward(self, x_id, x_po):
        x_id, x_po = self.extractor_id(x_id), self.extractor_po(x_po)
        if self.fdca_num>0 and self.resca:
            res_id, res_po = x_id.contiguous(), x_po.contiguous()
        for i in range(self.fdca_num):
            x_id, x_po = getattr(self, 'fdca'+str(i))(x_id, x_po)
        if self.fdca_num>0 and self.resca:
            res_id, _ = self.resca_id(res_id, x_po)
            x_id += res_id
            x_id = self.relu(x_id)
            _, res_po = self.resca_po(x_id, res_po)
            x_po += res_po
            x_po = self.relu(x_po)
        return x_id, x_po


class ResNet_ca(Basemodel):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3], img_size=(256, 128),
                 fdca_num=(2, 2, 2, 2, 2), url=None, resca=False, root_path=''):
        super(ResNet_ca, self).__init__()
        resnet = ResNet(last_stride, block, layers, url, root_path)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.layer_po0 = copy.deepcopy(self.layer0)
        self.layer_po1 = copy.deepcopy(self.layer1)
        self.layer_po2 = copy.deepcopy(self.layer2)
        self.layer_po3 = copy.deepcopy(self.layer3)
        self.layer_po4 = copy.deepcopy(self.layer4)

        h, w = img_size[0]//4, img_size[1]//4
        hid_dim = 64
        ca0 = FDCA(seq_len=h*w, hid_dim=hid_dim); hid_dim *= block.expansion
        ca1 = FDCA(seq_len=h*w, hid_dim=hid_dim)
        ca2 = FDCA(seq_len=(h//2)*(w//2), hid_dim=hid_dim*2)
        ca3 = FDCA(seq_len=(h//4)*(w//4), hid_dim=hid_dim*4)
        ca4 = FDCA(seq_len=(h//(4*last_stride))*(w//(4*last_stride)), hid_dim=hid_dim*8)

        for i in range(5):
            setattr(self, 'block'+str(i),
                    FD_Block(getattr(self, 'layer'+str(i)), getattr(self, 'layer_po'+str(i)),
                             eval('ca'+str(i)), fdca_num=fdca_num[i], resca=resca)
                    )

        self.out_features = hid_dim*8
        assert self.out_features == resnet.out_features

    def forward(self, x):
        # x = self.layer0(x)
        x_id = x
        x_po = x.contiguous()

        for i in range(5):
            x_id, x_po = getattr(self, 'block'+str(i))(x_id, x_po)

        return x_id, x_po


def resnet18_ca(cfg):  # 24.73312M
    backbone = ResNet_ca(last_stride=cfg.MODEL.LAST_STRIDE, block=BasicBlock, layers=[2, 2, 2, 2], img_size=cfg.INPUT.SIZE_TRAIN,
                         fdca_num=cfg.MODEL.FDCA_NUM, url=model_urls['resnet18'], resca=cfg.MODEL.RESCA, root_path=cfg.SYS.ROOT_PATH)
    return backbone


def resnet34_ca(cfg):  # 44.94944M
    backbone = ResNet_ca(last_stride=cfg.MODEL.LAST_STRIDE, block=BasicBlock, layers=[3, 4, 6, 3], img_size=cfg.INPUT.SIZE_TRAIN,
                         fdca_num=cfg.MODEL.FDCA_NUM, url=model_urls['resnet34'], resca=cfg.MODEL.RESCA, root_path=cfg.SYS.ROOT_PATH)
    return backbone


def resnet50_ca(cfg):  # 81.63258M   92.8M
    backbone = ResNet_ca(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 4, 6, 3], img_size=cfg.INPUT.SIZE_TRAIN,
                         fdca_num=cfg.MODEL.FDCA_NUM, url=model_urls['resnet50'], resca=cfg.MODEL.RESCA, root_path=cfg.SYS.ROOT_PATH)
    return backbone


def resnet101_ca(cfg):  # 119.61683M
    backbone = ResNet_ca(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 4, 23, 3], img_size=cfg.INPUT.SIZE_TRAIN,
                         fdca_num=cfg.MODEL.FDCA_NUM, url=model_urls['resnet101'], resca=cfg.MODEL.RESCA, root_path=cfg.SYS.ROOT_PATH)
    return backbone


def resnet152_ca(cfg):  # 150.90413M
    backbone = ResNet_ca(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 8, 36, 3], img_size=cfg.INPUT.SIZE_TRAIN,
                         fdca_num=cfg.MODEL.FDCA_NUM, url=model_urls['resnet152'], resca=cfg.MODEL.RESCA, root_path=cfg.SYS.ROOT_PATH)
    return backbone


if __name__=='__main__':
    import torch
    from yacs.config import CfgNode as CN
    cfg = CN()
    img_size = (192, 96)  #

    cfg.MODEL = CN()
    cfg.MODEL.LAST_STRIDE = 1
    cfg.MODEL.FDCA_NUM = (0,1,1,1,1)
    cfg.MODEL.RESCA = True
    cfg.INPUT = CN()
    cfg.INPUT.SIZE_TRAIN = img_size

    device = 'cpu'
    model = resnet50_ca(cfg)
    print("{} model size: {:.5f}M".format('', sum(p.numel() for p in model.parameters()) / 1000000.0))
    if device=='cuda':
        if torch.cuda.device_count() > 1:
            print("The model will be loaded onto multiple GPUs")
            model = nn.DataParallel(model)
    model.to(device)
    img = torch.Tensor(64, 3, img_size[0], img_size[1]).to(device)
    # img = torch.Tensor(32,3,256,128)
    x_id, x_cm = model(img)
    print(x_id.shape, x_cm.shape)
    print(model.out_features)
    pass
