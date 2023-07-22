from .resnet import *
from .resnet_ca import *
from .densenet import *
from .densenet_ca import *
from .resnet_nl import *
from .resnet_nl_ca import *
from .resnet_rga import *
from .resnet_rga_ca import *

__backbone_factory = {
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'resnet34_ca': resnet34_ca,
    'resnet50_ca': resnet50_ca,
    'densenet121_ca': densenet121_ca,
    'densenet169_ca': densenet169_ca,
    'densenet201_ca': densenet201_ca,
    'densenet161_ca': densenet161_ca,
    'resnet50_nl': resnet50_nl,
    'resnet50_nl_ca': resnet50_nl_ca,
    'resnet50_rga': resnet50_rga,
    'resnet50_rga_ca': resnet50_rga_ca,
    'resnet50_nl_resnet50': resnet50_nl_resnet50,
    'resnet50rga_resnet50nl': resnet50rga_resnet50nl,
    'resnet50rga_resnet50': resnet50rga_resnet50,
}


def backbone_names():
    return list(__backbone_factory.keys())


def init_backbone(name, **kwargs):
    if name not in __backbone_factory.keys():
        raise KeyError("Invalid backbone name, got '{}', but expected to be one of {}".
                       format(name, __backbone_factory.keys()))
    return __backbone_factory[name](**kwargs)
