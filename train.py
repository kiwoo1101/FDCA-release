import numpy as np
import random
from config import cfg, logger, waiting_gpu
import torch
from data import make_data_loader
from engines import *
from models import init_model
from losses import make_losses
from optimizers import build_optimizer
from lr_scheduler import build_scheduler
from utils import metric


def train():
    set_seed(cfg.TRAIN.SEED, cfg.SYS.CUDNN)
    train_loader, val_loader, num_query, num_classes, num_pose = make_data_loader(cfg) # prepare dataset
    if 'wuqi' in cfg.SYS.ROOT_PATH: waiting_gpu(cfg, logger)
    model = init_model(name=cfg.MODEL.NAME, cfg=cfg, num_classes=num_classes, num_pose=num_pose)
    criterion = make_losses(cfg, num_classes, num_pose, model.out_features)
    optimizer = build_optimizer(cfg, model, criterion)
    start_epoch, best_state = model_pretrain(cfg, model, optimizer, criterion)
    scheduler = build_scheduler(cfg, optimizer, start_epoch)
    logger.set_pr_time()
    eval('do_train_'+cfg.MODEL.NAME)(
        cfg,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        metric,
        num_query,
        start_epoch,
        best_state
    )


def model_pretrain(cfg, model, optimizer, criterion):
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        try:
            checkpoint = torch.load(cfg.MODEL.SELF_PRETRAIN_MODEL)
            model.load_state_dict(checkpoint['model'])
            optimizer['model'].load_state_dict(checkpoint['optimizer_model'])
            start_epoch = checkpoint['epoch']
            best_state = checkpoint['best_state']
            if 'cent' in optimizer.keys():
                optimizer['cent'].load_state_dict(checkpoint['optimizer_cent'])
                criterion['cent'].load_state_dict(checkpoint['criterion_cent'])
            if 'cent_po' in optimizer.keys():
                optimizer['cent_po'].load_state_dict(checkpoint['optimizer_cent_po'])
                criterion['cent_po'].load_state_dict(checkpoint['criterion_cent_po'])
            if 'cuda' in cfg.SYS.DEVICE:
                put_cuda(optimizer)
            rank1_str = '(rank1:{:.2%})'.format(checkpoint['rank1']) if checkpoint['rank1'] != -1 else ''
            print('Started with epoch {}{} --- best_state is rank1:{:.2%}({}), mAP:{:.2%}({}), '
                  'mINP:{:.2%}({})'.format(start_epoch, rank1_str, best_state[0][0], best_state[1][0],
                                           best_state[0][1], best_state[1][1], best_state[0][2], best_state[1][2]))
        except Exception:
            raise Exception('Error(load checkpoint)!!!')
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        print('Started with imagenet model')
        start_epoch, best_state = 0, [[float("-inf"), float("-inf"), float("-inf")], [0, 0, 0]]
    else:
        raise Exception('Only support imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))
    return start_epoch, best_state


def set_seed(seed, cudnn_enable=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def put_cuda(optimizers):
    for optimizer in optimizers.values():
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()


def demo():
    print(torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    train()
    # demo()
    pass
