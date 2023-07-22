import numpy as np
import torch
from ignite.metrics import Metric
from tqdm import tqdm
from .re_ranking import re_ranking
import torch.nn.functional as F


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm=True, minp=True, re_rank=False, dist_type='euclidean'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.minp = minp
        self.re_rank = re_rank
        self.dist_type = dist_type

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # print(feats.shape)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        if self.re_rank:
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        else:
            if self.dist_type == 'euclidean':
                print('using euclidean distance')
                m, n = qf.shape[0], gf.shape[0]
                distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
            elif self.dist_type == 'cosine':
                print('using cosine distance')
                qf = F.normalize(qf, p=2, dim=1)
                gf = F.normalize(gf, p=2, dim=1)
                distmat = 1 - torch.mm(qf, gf.t())
            else:
                raise NameError

            distmat = distmat.cpu().numpy()
        cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, minp=self.minp)

        return cmc, mAP, mINP


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, minp=True):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1) # 返回排序后的索引数组，axis=1代表按行排序
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    if minp:
        all_INP = []
    num_valid_q = 0.  # number of valid query

    # for q_idx in tqdm(range(num_q), desc='Test'):
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        if minp:
            pos_idx = np.where(orig_cmc == 1)
            max_pos_idx = np.max(pos_idx)
            inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
            all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    #     sys.stdout.write('\r'+'progress: {:.2%}'.format((q_idx+1)/num_q))
    # print('\n')

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    if minp:
        mINP = np.mean(all_INP)
        return all_cmc, mAP, mINP
    else:
        return all_cmc, mAP, -0.01
    # return all_cmc, mAP # all_cmc[0]为rank1精度，all_cmc[1]为rank2精度；mAP是一个浮点数


metric = {}
metric['R1_mAP'] = R1_mAP


if __name__=='__main__':
    distmat=np.array([[3,1,2,4,5],
                     [2,5,1,4,3]])
    q_pids=np.asarray([5,7])
    g_pids=np.asarray([3,4,5,6,7])
    q_camids=np.asarray([2,1])
    g_camids=np.asarray([2,1,1,3,2])

    eval_func(distmat,q_pids, g_pids, q_camids, g_camids)
    # from IPython import embed;embed()
    pass
