from torch import nn
from .backbone import init_backbone
from .layers import GeneralizedMeanPoolingP, weights_init_kaiming, weights_init_classifier


class baseline(nn.Module):
    # in_planes = 2048

    def __init__(self, cfg, num_classes, **kwargs):
        super(baseline, self).__init__()
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.MODEL.NECK_FEAT
        self.base = init_backbone(name=cfg.MODEL.BACKBONE_NAME, cfg=cfg)
        self.out_features = self.base.out_features
        if cfg.MODEL.GEM_POOL:
            print("Generalized Mean Pooling")
            self.global_pool = GeneralizedMeanPoolingP()
        else:
            print("Global Adaptive Pooling")
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        if self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.out_features)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.out_features, num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
        else:
            self.classifier = nn.Linear(self.out_features, num_classes)

    def forward(self, x):
        x = self.base(x)
        global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat_bn = self.bottleneck(global_feat) if self.neck == 'bnneck' else global_feat
        if self.training:
            cls_score = self.classifier(feat_bn)
            return cls_score, global_feat
        else:
            return feat_bn if self.neck_feat == 'after' else global_feat
