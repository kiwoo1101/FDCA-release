import torch


def build_optimizer(cfg, model, criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.OPTIM.BASE_LR
        weight_decay = cfg.OPTIM.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.BIAS_LR_FACTOR
            weight_decay = cfg.OPTIM.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "initial_lr": lr, "weight_decay": weight_decay}]
    optimizer = {}
    if cfg.OPTIM.OPTIMIZER_NAME == 'SGD':#torch.optim.SGD
        optimizer['model'] = getattr(torch.optim, cfg.OPTIM.OPTIMIZER_NAME)(params, momentum=cfg.OPTIM.MOMENTUM)
    elif cfg.OPTIM.OPTIMIZER_NAME == 'AdamW':
        optimizer['model'] = torch.optim.AdamW(params, lr=cfg.OPTIM.BASE_LR, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    else:
        optimizer['model'] = getattr(torch.optim, cfg.OPTIM.OPTIMIZER_NAME)(params)
    if 'cent' in cfg.LOSS.LOSS_TYPE:
        optimizer['cent'] = torch.optim.SGD(criterion['cent'].parameters(), lr=cfg.OPTIM.CENTER_LR)
        if cfg.MODEL.NAME == 'CA':
            optimizer['cent_po'] = torch.optim.SGD(criterion['cent_po'].parameters(), lr=cfg.OPTIM.CENTER_LR)
    return optimizer


if __name__=='__main__':
    pass

