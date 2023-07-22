import os, errno, torch, shutil
import os.path as osp
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from .inferencer import create_supervised_evaluator, inference
import time


def create_supervised_trainer(model, optimizer, criterion, cfg):
    device = cfg.SYS.DEVICE
    if 'cuda' in device:
        if torch.cuda.device_count() > 1:
            print("The model will be loaded onto multiple GPUs")
            model = nn.DataParallel(model)
        else:
            # print("The model will be loaded onto GPU"+cfg.SYS.DEVICE_IDS)
            print("The model will be loaded onto GPU")
    model.to(device)
    xent_w = cfg.LOSS.ID_LOSS_WEIGHT
    htri_w = cfg.LOSS.TRI_LOSS_WEIGHT
    cent_w = cfg.LOSS.CENTER_LOSS_WEIGHT

    def _update(engine, batch):
        model.train()
        for opti in optimizer.values():
            opti.zero_grad()
        img, pid = batch
        if device == 'cuda' and torch.cuda.device_count() >= 1:
            img = img.to(device)
            pid = pid.to(device)
        score, feat = model(img)

        loss_xent = xent_w*criterion['xent'](score, pid)
        loss_htri = htri_w*criterion['htri'](feat, pid)
        loss_cent = cent_w*criterion['cent'](feat, pid)

        loss = loss_xent+loss_htri+loss_cent
        loss.backward()
        if 'cent' in optimizer.keys():
            for param in criterion['cent'].parameters():
                param.grad.data *= (1. / cent_w)
        for opti in optimizer.values():
            opti.step()

        # compute acc
        acc = (score.max(1)[1] == pid).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def do_train(
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
        best_state,
):
    trainer = create_supervised_trainer(model, optimizer, criterion, cfg)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'r1_mAP':
                                                         metric['R1_mAP'](num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                                                                minp=cfg.TEST.METRIC_MINP, dist_type=cfg.TEST.DIST_TYPE)},
                                            cfg=cfg)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'acc')

    best = best_state[0]
    best_epoch = best_state[1]

    time_st = [time.perf_counter()]

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch
        print('Start training')

    @trainer.on(Events.COMPLETED)
    def nf_start_training(engine):
        print('End training')
        if cfg.TEST.RE_RANKING:
            inference(cfg, model, val_loader, metric, num_query, only_reranking=True)
            print('End inferencing')

    @trainer.on(Events.EPOCH_STARTED)
    def save_time(engine):
        time_st[0] = time.perf_counter()

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        # if cfg.MODEL.SAVE:
        #     save_checkpoint(model, optimizer, criterion, -1, engine.state.epoch, best_state,
        #                     False, osp.join(cfg.SYS.OUTPUT_DIR, 'checkpoint.pth.tar'))
        scheduler.step()
        trainer.state.iteration = 0
        epoch_time = time.perf_counter() - time_st[0]
        print('-' * 50 + "Epoch[{}] is done. Take {} m {:.1f} s".format(engine.state.epoch, epoch_time//60, epoch_time%60))

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.TRAIN.PRINT_FREQ))
    def log_training_loss(engine):
        loss_str = ("Loss: {:.3f}"
                    .format(engine.state.metrics['loss']))
        print("Epoch: {}/{}  Iteration: {:>3}/{:<3}  {}  id_Acc: {:.3f}  Base Lr: {:.2e}"
              .format(engine.state.epoch, cfg.TRAIN.MAX_EPOCHS, trainer.state.iteration, len(train_loader),
                      loss_str, engine.state.metrics['acc'],
                      scheduler.get_lr()[0]))

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.TEST.VALID_FREQ))
    def log_validation_results(engine):
        if engine.state.epoch<cfg.TEST.START_EPOCH:
            if cfg.MODEL.SAVE:
                save_checkpoint(model, optimizer, criterion, -1, engine.state.epoch, best_state,
                                False, osp.join(cfg.SYS.OUTPUT_DIR, 'checkpoint.pth.tar'))
            return
        st = time.perf_counter()
        print('Start evaluating')
        evaluator.run(val_loader)
        cmc, mAP, mINP = evaluator.state.metrics['r1_mAP']
        print("Validation Results - Epoch: {}".format(engine.state.epoch))
        print("mAP: {:.2%}".format(mAP))
        print("mINP: {:.2%}".format(mINP))
        for r in [1, 5, 10]:
            print("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
        if cmc[0] > best[0]:
            best[0] = cmc[0]
            best_epoch[0] = engine.state.epoch
            isbest = True
        else:
            isbest = False
        if mAP > best[1]:
            best[1] = mAP
            best_epoch[1] = engine.state.epoch
        if mINP > best[2]:
            best[2] = mINP
            best_epoch[2] = engine.state.epoch
        print('The best Rank-1 {:.2%} is implemented in Epoch {}'.format(best[0], best_epoch[0]))
        print('The best mAP {:.2%} is implemented in Epoch {}'.format(best[1], best_epoch[1]))
        print('The best mINP {:.2%} is implemented in Epoch {}'.format(best[2], best_epoch[2]))
        # if isbest:
        #     fpath = osp.join(cfg.SYS.OUTPUT_DIR, 'checkpoint.pth.tar')
        #     shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))
        #     print('The best model is saved.')
        if cfg.MODEL.SAVE:
            save_checkpoint(model, optimizer, criterion, cmc[0], engine.state.epoch, best_state,
                            isbest, osp.join(cfg.SYS.OUTPUT_DIR, 'checkpoint.pth.tar'))
        val_time = time.perf_counter()-st
        print('-' * 10 + "Validation epoch: {} takes {} m {:.1f} s.".format(engine.state.epoch, val_time//60, val_time%60))

    trainer.run(train_loader, max_epochs=cfg.TRAIN.MAX_EPOCHS)


def save_checkpoint(model, optimizer, criterion, rank1, epoch, best_state, is_best, fpath='checkpoint.pth.tar'):
    state={}
    state['model']=model.state_dict()
    state['optimizer_model']=optimizer['model'].state_dict()
    if 'cent' in optimizer.keys():
        state['optimizer_cent'] = optimizer['cent'].state_dict()
        state['criterion_cent'] = criterion['cent'].state_dict()
    state['rank1']=rank1
    state['epoch']=epoch
    state['best_state']=best_state
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    print('The checkpoint is saved.')
    # shutil.copy用于复制文件
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))
        print('The best model is saved.')


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
