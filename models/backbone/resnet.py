import torch
from torch import nn
try:
    from basemodel import Basemodel
except:
    from .basemodel import Basemodel


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Basemodel):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3], url=None, root_path=''):
        super(ResNet, self).__init__()
        # self.out_features = 2048
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        if url is not None:
            self.load_param(url, root_path)

        self.out_features = self.inplanes

        # self.features = nn.Sequential(self.conv1, self.bn1, self.maxpool)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        return x


def resnet18(cfg):
    return ResNet(last_stride=cfg.MODEL.LAST_STRIDE, block=BasicBlock, layers=[2, 2, 2, 2], url=model_urls['resnet18'], root_path=cfg.SYS.ROOT_PATH)


def resnet34(cfg):
    return ResNet(last_stride=cfg.MODEL.LAST_STRIDE, block=BasicBlock, layers=[3, 4, 6, 3], url=model_urls['resnet34'], root_path=cfg.SYS.ROOT_PATH)


def resnet50(cfg):
    return ResNet(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 4, 6, 3], url=model_urls['resnet50'], root_path=cfg.SYS.ROOT_PATH)


def resnet101(cfg):
    return ResNet(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 4, 23, 3], url=model_urls['resnet101'], root_path=cfg.SYS.ROOT_PATH)


def resnet152(cfg):
    return ResNet(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 8, 36, 3], url=model_urls['resnet152'], root_path=cfg.SYS.ROOT_PATH)


if __name__ == '__main__':
    model = resnet50(1)
    print("{} model size: {:.5f}M".format('', sum(p.numel() for p in model.parameters()) / 1000000.0))
    img = torch.Tensor(32,3,256,128)
    out = model(img)
    print(out.shape)
    pass
