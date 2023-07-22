from .id_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss, TripletLoss_luo, WeightedRegularizedTriplet
from .center_loss import CenterLoss
from .consist_loss import Consist_Loss
import torch.nn.functional as F


# def make_losses(cfg, num_classes):
#     if 'resnet18' in cfg.MODEL.BACKBONE_NAME or 'resnet34' in cfg.MODEL.BACKBONE_NAME:
#         feat_dim = 512
#     else:
#         feat_dim = 2048
#     def getzero(a=0,b=0,c=0,**kwargs):
#         return 0
#     xent = getzero
#     htri = getzero
#     cent = getzero
#     if 'xent' in cfg.LOSS.LOSS_TYPE:
#         xent = CrossEntropyLabelSmooth(num_classes=num_classes) if cfg.LOSS.LABELSMOOTH == 'yes' else F.cross_entropy
#     if 'htri' in cfg.LOSS.LOSS_TYPE:
#         htri = WeightedRegularizedTriplet() if cfg.LOSS.WEIGHT_REGULARIZED_TRIPLET == 'yes' else \
#             TripletLoss_luo(cfg.LOSS.MARGIN)
#     if 'cent' in cfg.LOSS.LOSS_TYPE:
#         cent = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
#
#     def loss_func(score, feat, target):
#         return xent(score, target)+htri(feat, target)+\
#                cfg.LOSS.CENTER_LOSS_WEIGHT * cent(feat, target)
#
#     if 'cent' in cfg.LOSS.LOSS_TYPE:
#         return loss_func,cent
#     else:
#         return loss_func


def make_losses(cfg, num_classes, num_cam, feat_dim):
    criterion = {}
    device = cfg.SYS.DEVICE

    def getzero(a=0,b=0,c=0,**kwargs):
        return 0

    loss_list = ['xent', 'htri', 'cent', 'xent_po', 'cent_po', 'cons']
    for los in loss_list:
        criterion[los] = getzero

    if 'xent' in cfg.LOSS.LOSS_TYPE:
        criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes, device=device) if cfg.LOSS.LABELSMOOTH else F.cross_entropy
        if cfg.MODEL.NAME == 'CA':
            criterion['xent_po'] = CrossEntropyLabelSmooth(num_classes=num_cam, device=device) if cfg.LOSS.LABELSMOOTH else F.cross_entropy
    if 'htri' in cfg.LOSS.LOSS_TYPE:
        criterion['htri'] = WeightedRegularizedTriplet(dist_type=cfg.LOSS.DIST_TYPE) if cfg.LOSS.WEIGHT_REGULARIZED_TRIPLET else \
            TripletLoss_luo(margin=cfg.LOSS.MARGIN, dist_type=cfg.LOSS.DIST_TYPE)
    if 'cent' in cfg.LOSS.LOSS_TYPE:
        criterion['cent'] = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, device=device)  # center loss
        if cfg.MODEL.NAME == 'CA':
            criterion['cent_po'] = CenterLoss(num_classes=num_cam, feat_dim=feat_dim, device=device)
    if 'cons' in cfg.LOSS.LOSS_TYPE:
        criterion['cons'] = Consist_Loss()


    # from IPython import embed;embed()

    # def loss_total(score, feat, target):
    #     return cfg.LOSS.ID_LOSS_WEIGHT * criterion['xent'](score, target)+\
    #            cfg.LOSS.TRI_LOSS_WEIGHT * criterion['htri'](feat, target)+\
    #            cfg.LOSS.CENTER_LOSS_WEIGHT * criterion['cent'](feat, target)

    # for los in loss_list:
    #     if los not in cfg.LOSS.LOSS_TYPE:
    #         criterion.pop(los)

    return criterion

