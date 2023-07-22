import torch
from torch import nn
try:
    from .basemodel import Basemodel
    from .resnet import BasicBlock, Bottleneck
    from .resnet import model_urls
except:
    from basemodel import Basemodel
    from resnet import BasicBlock, Bottleneck
    from resnet import model_urls


__all__ = ['resnet50_nl']


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class ResNetNL_Block(nn.Module):
    def __init__(self, layers, layers_num, nl_dim, nl_layer_num):
        super(ResNetNL_Block, self).__init__()
        self.layers = layers
        self.nl_layer_num = nl_layer_num

        self.NL_layer = nn.ModuleList(
            [Non_local(nl_dim) for _ in range(nl_layer_num)])
        self.NL_idx = sorted([layers_num - (i + 1) for i in range(nl_layer_num)])

    def forward(self, x):
        if self.nl_layer_num == 0:
            x = self.layers(x)
        else:
            NL_counter = 0
            if len(self.NL_idx) == 0: self.NL_idx = [-1]
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                if i == self.NL_idx[NL_counter]:
                    x = self.NL_layer[NL_counter](x)
                    NL_counter += 1
        return x


class ResNetNL(Basemodel):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3], non_layers=[0, 2, 3, 0], url=None, root_path=''):
        super(ResNetNL, self).__init__()
        self.inplanes = 64
        nl_dim = self.inplanes*block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        if url is not None:
            self.load_param(url, root_path)
        self.out_features = self.inplanes

        self.block_num = len(layers)



        for i in range(self.block_num):
            setattr(self, 'block'+str(i), ResNetNL_Block(getattr(self, 'layer'+str(i+1)),layers[i],nl_dim,non_layers[i]))
            nl_dim *= 2

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

        for i in range(self.block_num):
            x = getattr(self, 'block'+str(i))(x)

        return x


def resnet50_nl(cfg):
    return ResNetNL(last_stride=cfg.MODEL.LAST_STRIDE, block=Bottleneck, layers=[3, 4, 6, 3],
                        non_layers=[0, 2, 3, 0], url=model_urls['resnet50'], root_path=cfg.SYS.ROOT_PATH)


if __name__ == '__main__':
    pass
