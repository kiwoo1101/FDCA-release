from torch import nn


class feat_embed(nn.Module):
    def __init__(self, in_planes, out_features):
        super(feat_embed, self).__init__()
        self.linear = nn.Linear(in_planes, out_features)

    def forward(self, x):
        return self.linear(x)
