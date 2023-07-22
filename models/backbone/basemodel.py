import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.utils import model_zoo


class Basemodel(nn.Module):

    # def __init__(self):
    #     super(Basemodel, self).__init__()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_param_0(self, model_url, root_path, model_dir='kr/.torch/'):
        print('Loading pretrained ImageNet model......from {}'.format(model_url))
        # param_dict = torch.load(model_path)
        param_dict = model_zoo.load_url(model_url, model_dir=root_path+model_dir)
        k = 0
        for i in param_dict:
            if 'fc' in i:
                continue
            k += 1
            self.state_dict()[i].copy_(param_dict[i])
        pass

    def load_param(self, model_url, root_path, model_dir='kr/.torch/'):
        pretrain_dict = model_zoo.load_url(model_url, model_dir=root_path+model_dir)
        model_dict = self.state_dict()
        pretrain_dict0 = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict0)
        self.load_state_dict(model_dict)
        print('Initialized model with pretrained weights from {}'.format(model_url))
        # denselayer4.conv.2.weight norm

    def load_param_rga(self, model_url, root_path, model_dir='kr/.torch/'):
        self.load_specific_param(self.conv1.state_dict(), 'conv1', model_url, root_path, model_dir=model_dir)
        self.load_specific_param(self.bn1.state_dict(), 'bn1', model_url, root_path, model_dir=model_dir)
        self.load_partial_param(self.layer1.state_dict(), 1, model_url, root_path, model_dir=model_dir)
        self.load_partial_param(self.layer2.state_dict(), 2, model_url, root_path, model_dir=model_dir)
        self.load_partial_param(self.layer3.state_dict(), 3, model_url, root_path, model_dir=model_dir)
        self.load_partial_param(self.layer4.state_dict(), 4, model_url, root_path, model_dir=model_dir)
        print('Initialized RGA model with pretrained weights from {}'.format(model_url))

    def load_param_dense(self, model_url, root_path, model_dir='kr/.torch/'):
        pretrain_dict = model_zoo.load_url(model_url, model_dir=root_path+model_dir)
        model_dict = self.state_dict()
        # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        pretrain_dict0 = {}
        for k, v in pretrain_dict.items():
            if k in model_dict:
                assert model_dict[k].size() == v.size()
                pretrain_dict0[k] = v
            elif 'denselayer' in k:
                if 'conv.' in k:
                    split_str = 'conv'
                elif 'norm.' in k:
                    split_str = 'norm'
                else:
                    raise Exception('Invalid.')
                k_spl = k.split(split_str+'.')
                k0 = k_spl[0] + split_str + k_spl[1]
                assert model_dict[k0].size() == v.size()
                pretrain_dict0[k0] = v
        model_dict.update(pretrain_dict0)
        self.load_state_dict(model_dict)
        print('Initialized densenet model with pretrained weights from {}'.format(model_url))

    def load_partial_param(self, state_dict, model_index, model_url, root_path, model_dir='kr/.torch/'):
        # param_dict = torch.load(model_path)
        param_dict = model_zoo.load_url(model_url, model_dir=root_path + model_dir)
        for i in state_dict:
            try:
                key = 'layer{}.'.format(model_index)+i
                state_dict[i].copy_(param_dict[key])
            except KeyError:
                pass
        del param_dict

    def load_specific_param(self, state_dict, param_name, model_url, root_path, model_dir='kr/.torch/'):
        # param_dict = torch.load(model_path)
        param_dict = model_zoo.load_url(model_url, model_dir=root_path + model_dir)
        for i in state_dict:
            try:
                key = param_name + '.' + i
                state_dict[i].copy_(param_dict[key])
            except KeyError:
                pass
        del param_dict

    def set_features(self, out_features):
        self.out_features = out_features

