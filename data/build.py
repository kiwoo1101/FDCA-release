import torch
import time
from torch.utils.data import DataLoader
try:
    from data_manager import init_dataset
    from sampler import RandomIdentitySampler, RandomIdentitySampler_pose
    from transform import build_transforms
    from data_loader import ImageDataset
except:
    from .data_manager import init_dataset
    from .sampler import RandomIdentitySampler, RandomIdentitySampler_pose
    from .transform import build_transforms
    from .data_loader import ImageDataset


def train_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def train_collate_fn_pos(batch):
    imgs, pids, camids, poseids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    poseids = torch.tensor(poseids, dtype=torch.int64)
    # camids = torch.tensor(camids, dtype=torch.int64)
    # torch.stack可以将多个n维数据组成的序列拼成n+1维 --> (batch_size,C,H,W)
    return torch.stack(imgs, dim=0), pids, poseids

def val_collate_fn(batch):
    imgs, pids, camids, poseid = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids # 注意这里pids, camids并不是张量


def make_data_loader(cfg):
    print('Loading data...')
    time_st = time.perf_counter()
    train_transforms = build_transforms(cfg, is_train=True, randomErasing=cfg.INPUT.REA)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, cluster_k=cfg.DATASETS.CLUSTER_K,
                           flip=cfg.INPUT.FLIP, miss_value=cfg.INPUT.MISS_VALUE, poseid=(cfg.MODEL.NAME == 'CA'),
                           root_path=cfg.SYS.ROOT_PATH)
    num_classes = dataset.num_train_pids
    num_pose = dataset.num_train_pose
    num_query = dataset.num_query_imgs
    train_set = ImageDataset(dataset.train, train_transforms)
    collate_fn = train_collate_fn_pos if cfg.MODEL.NAME=='CA' else train_collate_fn
    if 'cons' in cfg.LOSS.LOSS_TYPE:
        train_loader = DataLoader(
            train_set, batch_size=cfg.TRAIN.BATCH_SIZE,
            sampler=RandomIdentitySampler_pose(dataset.train, cfg.TRAIN.BATCH_SIZE, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=collate_fn
        )
    elif 'htri' in cfg.LOSS.LOSS_TYPE:
        train_loader = DataLoader(
            train_set, batch_size=cfg.TRAIN.BATCH_SIZE,
            sampler=RandomIdentitySampler(dataset.train, cfg.TRAIN.BATCH_SIZE, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=num_workers,
            collate_fn=collate_fn, drop_last=True
        )
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn, drop_last=False
    )
    reset_print_freq = False
    while cfg.SYS.DEVICE != 'cpu' and len(train_loader)//cfg.TRAIN.PRINT_FREQ > 8:
        cfg.TRAIN.PRINT_FREQ *= 2
        reset_print_freq = True
    if reset_print_freq:
        print('Reset print_freq to {}'.format(cfg.TRAIN.PRINT_FREQ))
    loading_time = time.perf_counter() - time_st
    print('Loading data takes {} m {:.1f} s.'.format(loading_time//60, loading_time%60))
    return train_loader, val_loader, num_query, num_classes, num_pose


if __name__=='__main__':
    pass
