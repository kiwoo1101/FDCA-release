import torch
import torch.nn as nn
from ignite.engine import Engine


def create_supervised_evaluator(model, metrics, cfg):
    device = cfg.SYS.DEVICE
    if 'cuda' in device and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    def _inference(engine, batch):
        # torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        metric,
        num_query,
        only_reranking=False
):
    def test(evaluator, val_loader):
        evaluator.run(val_loader)
        cmc, mAP, mINP = evaluator.state.metrics['r1_mAP']
        print('Validation Results - ')
        print("mAP: {:.2%}".format(mAP))
        print("mINP: {:.2%}".format(mINP))
        for r in [1, 5, 10]:
            print("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))

    if not only_reranking:
        print('-'*30)
        print("Inferencing...")
        evaluator = create_supervised_evaluator(model,
                                                metrics={'r1_mAP':
                                                             metric['R1_mAP'](num_query, max_rank=50,
                                                             feat_norm=cfg.TEST.FEAT_NORM, minp=cfg.TEST.METRIC_MINP,
                                                             dist_type=cfg.TEST.DIST_TYPE)},
                                                cfg=cfg)
        test(evaluator, val_loader)

    if cfg.TEST.RE_RANKING:
        print('-'*30)
        print("Re_ranking...")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': metric['R1_mAP'](num_query, max_rank=50,
                    feat_norm=cfg.TEST.FEAT_NORM, minp=cfg.TEST.METRIC_MINP, re_rank=cfg.TEST.RE_RANKING, dist_type=cfg.TEST.DIST_TYPE)}, cfg=cfg)
        test(evaluator, val_loader)


