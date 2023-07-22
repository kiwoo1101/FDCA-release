import copy, random, torch
import sys
from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import Sampler


# RandomIdentitySampler：得到几乎所有图像（12000左右）的配对
# RandomIdentitySampler_low：每个epoch只能获取每个pid的四张图片，严重减小了数据规模

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        # 打乱每个pid对应的图像顺序
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        # 此时的batch_idxs_dict：1:[[1,1,1,1],[1,1,1,1]...],2:[[2,2,2,2],[2,2,2,2]...],...
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        # 打乱pid顺序
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0) # 弹出列表首元素
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySampler_low(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        # index_dic是一个字典，每个pid对应一个列表(该列表中存储该pid对应的所有图像id)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)# 0-750随机打乱
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)# 对每个pid,随机选4个图片
            ret.extend(t) # 在列表末尾一次性追加另一个序列中的多个值
        # 返回一个迭代器,[0,23,5432,7,8,...,65],1-4张图片属于一个ID,其余类推
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


class RandomIdentitySampler_pose(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.leng = len(data_source)
        assert self.leng % 2 == 0
        self.batch_size = batch_size
        self.num_instances = num_instances
        assert self.num_instances % 2 == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        n = self.leng//2
        # 打乱每个pid对应的图像顺序
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                try:
                    if idx >= n:
                        assert idx-n in idxs
                        continue
                    else:
                        assert idx + n in idxs
                except:
                    print(pid)
                    print(idx)
                    print(n)
                    print(idxs)
                    sys.exit(1)
                batch_idxs.append(idx)
                batch_idxs.append(idx+n)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        # 此时的batch_idxs_dict：1:[[1,1,1,1],[1,1,1,1]...],2:[[2,2,2,2],[2,2,2,2]...],...
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        # 打乱pid顺序
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0) # 弹出列表首元素
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


if __name__=='__main__':
    # from data_manager import Market1501
    # market=Market1501()
    # ran=RandomIdentitySampler(data_source=market.train)
    print(6//2)
    pass

