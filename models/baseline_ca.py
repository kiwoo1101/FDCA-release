from torch import nn
import copy
from .backbone import init_backbone
from .layers import GeneralizedMeanPoolingP, weights_init_kaiming, weights_init_classifier


class baseline_ca(nn.Module):
    # in_planes = 2048

    def __init__(self, cfg, num_classes, num_pose):
        super(baseline_ca, self).__init__()
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
        self.global_pool_po = copy.deepcopy(self.global_pool)
        if self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.out_features)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.out_features, num_classes, bias=False)
            self.classifier_po = nn.Linear(self.out_features, num_pose, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.bottleneck_po = copy.deepcopy(self.bottleneck)
        else:
            self.classifier = nn.Linear(self.out_features, num_classes, bias=False)
            self.classifier_po = nn.Linear(self.out_features, num_pose, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_po.apply(weights_init_classifier)

    def forward(self, x):
        x_id, x_po = self.base(x)
        global_feat_id = self.global_pool(x_id)  # (b, 2048, 1, 1)
        global_feat_id = global_feat_id.view(global_feat_id.shape[0], -1)  # flatten to (bs, 2048)
        feat_bn_id = self.bottleneck(global_feat_id) if self.neck == 'bnneck' else global_feat_id
        if self.training:
            global_feat_po = self.global_pool_po(x_po)  # (b, 2048, 1, 1)
            global_feat_po = global_feat_po.view(global_feat_po.shape[0], -1)  # flatten to (bs, 2048)
            feat_bn_po = self.bottleneck_po(global_feat_po) if self.neck == 'bnneck' else global_feat_po
            cls_score_id = self.classifier(feat_bn_id)
            cls_score_po = self.classifier_po(feat_bn_po)
            return cls_score_id, global_feat_id, cls_score_po, global_feat_po, x_id, x_po
        else:
            return feat_bn_id if self.neck_feat == 'after' else global_feat_id


