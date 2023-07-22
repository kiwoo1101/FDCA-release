from config import cfg, logger, waiting_gpu
import torch
from data import make_data_loader
from engines import inference
from models import init_model
from utils import metric, save_heatmap


def test():
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    # cfg.MODEL.NECK_FEAT = 'before'
    train_loader, val_loader, num_query, num_classes, num_pose = make_data_loader(cfg) # prepare dataset
    # if 'wuqi' in cfg.SYS.ROOT_PATH: waiting_gpu(cfg, logger)
    model=get_model(cfg, num_classes, num_pose)
    logger.set_pr_time()
    # grad_cam(model, target_layer='block4')
    inference(
        cfg,
        model,
        val_loader,
        metric,
        num_query
    )


def heatmap():
    records={}
    img_dir="/home/wuqi/kr/dataset/market/bounding_box_train/"
    # imgs=['1101_c6s3_011242_01.jpg','1123_c2s2_153902_01.jpg','0068_c6s1_012476_01.jpg','0159_c6s1_030301_01.jpg','0810_c3s2_107303_05.jpg']
    imgs=['1101_c6s3_011242_01.jpg','0159_c6s1_030301_01.jpg','0810_c3s2_107303_05.jpg']
    # img_dir = "/home/wuqi/kr/dataset/cuhk03/images_labeled/"
    # imgs=['2_123_1_01.png','2_097_2_10.png','1_293_2_07.png','4_006_1_05.png']
    model_lists=[]
    cfg.MODEL.PRETRAIN_CHOICE = 'self'

    cfg.merge_from_file("/home/wuqi/kr/kr1/CA/configs/bot.yml")
    cfg.MODEL.SELF_PRETRAIN_MODEL = cfg.SYS.ROOT_PATH + cfg.MODEL.SELF_PRETRAIN_MODEL
    train_loader, val_loader, num_query, num_classes, num_pose = make_data_loader(cfg)
    model = get_model(cfg, num_classes, num_pose)
    model_lists.append([model, 'layer4', 'bot'])

    # cfg.merge_from_file("/home/wuqi/kr/kr1/CA/configs/agw.yml")
    # cfg.MODEL.SELF_PRETRAIN_MODEL=cfg.SYS.ROOT_PATH + cfg.MODEL.SELF_PRETRAIN_MODEL
    # train_loader, val_loader, num_query, num_classes, num_pose = make_data_loader(cfg)
    # model = get_model(cfg, num_classes, num_pose)
    # model_lists.append([model, 'block3', 'agw'])

    cfg.merge_from_file("/home/wuqi/kr/kr1/CA/configs/ca.yml")
    cfg.MODEL.SELF_PRETRAIN_MODEL = cfg.SYS.ROOT_PATH + cfg.MODEL.SELF_PRETRAIN_MODEL
    train_loader, val_loader, num_query, num_classes, num_pose = make_data_loader(cfg)
    model = get_model(cfg, num_classes, num_pose)
    model_lists.append([model, 'block4', 'ca'])
    for img in imgs:
        records[img_dir+img] = model_lists

    save_heatmap(records)


def get_model(cfg, num_classes, num_pose):
    model = init_model(name=cfg.MODEL.NAME, cfg=cfg, num_classes=num_classes, num_pose=num_pose)
    try:
        checkpoint = torch.load(cfg.MODEL.SELF_PRETRAIN_MODEL, map_location=cfg.SYS.DEVICE)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_state = checkpoint['best_state']
        rank1_str = '(rank1:{:.2%})'.format(checkpoint['rank1']) if checkpoint['rank1'] != -1 else ''
        print('Inference with epoch {}{} --- best_state is rank1:{:.2%}({}), mAP:{:.2%}({}), '
              'mINP:{:.2%}({})'.format(start_epoch, rank1_str, best_state[0][0], best_state[1][0],
                                       best_state[0][1], best_state[1][1], best_state[0][2], best_state[1][2]))
    except Exception:
        raise Exception('Error(load checkpoint)!!!')
    return model


if __name__ == '__main__':
    test()
    # heatmap()
    pass
