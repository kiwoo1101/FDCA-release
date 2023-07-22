import math
from bisect import bisect_right
from torch.optim import lr_scheduler


def build_scheduler(cfg, optimizer, start_epoch):
    scheduler = WarmupMultiStepLR(cfg, optimizer['model'], start_epoch - 1)
    return scheduler


class WarmupMultiStepLR(lr_scheduler._LRScheduler):

    def __init__(self, cfg, optimizer, last_epoch=-1):
        milestones = cfg.SCHEDULER.STEPS
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        self.milestones = milestones
        self.gamma = cfg.SCHEDULER.GAMMA
        self.warmup_factor = cfg.SCHEDULER.WARMUP_FACTOR
        self.warmup_iters = cfg.SCHEDULER.WARMUP_ITERS
        super().__init__(optimizer, last_epoch)
        if self.warmup_iters == 0:
            raise Exception("The warmup_iters cannot be set to {}".format(self.warmup_iters))

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                    for base_lr in self.base_lrs]


class WarmupCosineLR(lr_scheduler._LRScheduler):

    def __init__(self, cfg, optimizer, last_epoch=-1):
        self.warmup_factor = cfg.SCHEDULER.WARMUP_FACTOR
        self.warmup_iters = cfg.SCHEDULER.WARMUP_ITERS
        self.max_iter, self.eta_min = cfg.TRAIN.MAX_EPOCHS, cfg.SCHEDULER.ETA_MIN
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            # print ("after warmup")
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(
                        math.pi * (self.last_epoch - self.warmup_iters) / (self.max_iter - self.warmup_iters))) / 2
                    for base_lr in self.base_lrs]


class WarmupPolyLR(lr_scheduler._LRScheduler):
    def __init__(self, cfg, optimizer, last_epoch=-1):
        self.warmup_factor = cfg.SCHEDULER.WARMUP_FACTOR
        self.warmup_iters = cfg.SCHEDULER.WARMUP_ITERS
        self.power = cfg.SCHEDULER.POWER # power越接近1，下降曲线越直； power越接近0，下降曲线越弯曲
        self.T_max, self.eta_min = cfg.TRAIN.MAX_EPOCHS, cfg.SCHEDULER.ETA_MIN
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    math.pow(1 - (self.last_epoch - self.warmup_iters) / (self.T_max - self.warmup_iters),
                             self.power) for base_lr in self.base_lrs]


if __name__=='__main__':
    import sys, path
    import matplotlib.pyplot as plt
    folder = path.Path(__file__).abspath()
    sys.path.append(folder.parent.parent)

    from config import cfg
    from models import init_model
    from losses import make_losses
    from optimizers import make_optimizer

    num_classes = 751
    model = init_model(name=cfg.MODEL.NAME, cfg=cfg, num_classes=num_classes)
    nformer = init_model(name=cfg.MODEL.NF_NAME, cfg=cfg, num_classes=num_classes) if cfg.MODEL.NF else None
    criterion = make_losses(cfg, num_classes, model.out_features)
    optimizer = make_optimizer(cfg, model, nformer, criterion)
    start_epoch = 0
    scheduler = WarmupMultiStepLR(cfg, optimizer['model'], start_epoch - 1)
    # scheduler = WarmupCosineLR(cfg, optimizer['model'], start_epoch-1)
    # scheduler = WarmupPolyLR(cfg, optimizer['model'], start_epoch - 1)


    epoch = []
    sch = []
    for i in range(start_epoch, cfg.TRAIN.MAX_EPOCHS):
        epoch.append(i)
        # sch.append(optimizer['model'].param_groups[0]['lr'])
        sch.append(scheduler.get_lr()[0])
        # if i<40:
        #     print("i={}, Base Lr: {:.2e}".format(i,scheduler.get_lr()[0]))
        optimizer['model'].step()
        scheduler.step()

    plt.plot(epoch, sch, "b:")  # "b"为蓝色, "o"为圆点, ":"为点线
    plt.show()
