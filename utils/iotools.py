# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import errno
import json
import os
import re

import os.path as osp
import matplotlib
from matplotlib import pyplot as plt


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_log():
    # file = "/home/wuqi/kr/kr1/CA/logs/cuhk03_d_CA_resnet50_nl_resnet50-60/20230421/000256_CA_resnet50_nl_resnet50.log"
    # file = "/home/wuqi/kr/kr1/CA/logs/cuhk03_l_CA_resnet50_nl_resnet50-60/20230420/235423_CA_resnet50_nl_resnet50.log"
    file = "/home/wuqi/kr/kr1/CA/logs/market_CA_resnet50_nl_resnet50-60/20230302/123701_CA_resnet50_nl_resnet50.log"
    # file = "/home/wuqi/kr/kr1/CA/logs/market_CA_resnet50_nl_resnet50-60/20230421/125251_CA_resnet50_nl_resnet50.log"
    loss_id=[]
    loss_po=[]
    id_acc=[]
    po_acc=[]
    with open(file) as f:
        while True:
            lines = f.readline()
            if not lines:
                break
            head_loss_id = 'loss_id: '
            pattern_loss_id = re.compile(head_loss_id + '[0-9]+\.[0-9]+')
            loss_id_match=pattern_loss_id.search(lines)

            head_loss_po = 'loss_po: '
            pattern_loss_po = re.compile(head_loss_po + '[0-9]+\.[0-9]+')
            loss_po_match = pattern_loss_po.search(lines)

            head_id_acc = 'id_Acc: '
            pattern_id_acc = re.compile(head_id_acc + '[0-9]+\.[0-9]+')
            id_acc_match = pattern_id_acc.search(lines)

            head_po_acc = 'po_Acc: '
            pattern_po_acc = re.compile(head_po_acc + '[0-9]+\.[0-9]+')
            po_acc_match = pattern_po_acc.search(lines)
            if loss_id_match:
                assert loss_po_match is not None and id_acc_match is not None and po_acc_match is not None
                loss_id.append(float(loss_id_match.group().split(head_loss_id)[-1]))
                loss_po.append(float(loss_po_match.group().split(head_loss_po)[-1]))
                id_acc.append(float(id_acc_match.group().split(head_id_acc)[-1]))
                po_acc.append(float(po_acc_match.group().split(head_po_acc)[-1]))

    n=len(loss_id)
    # fig=plt.figure(dpi=300,figsize=(15,6))
    fig=plt.figure()
    font_size=16
    # plt.rcParams.update({"font.size": font_size})
    ax=fig.add_subplot(111)
    start = 1
    interval=4
    l=90*interval+start
    id_acc=id_acc[start:l:interval]
    po_acc=po_acc[start:l:interval]
    id_loss=loss_id[start:l:interval]
    po_loss=loss_po[start:l:interval]
    x=range(len(id_acc))
    # ax.plot(x, id_loss, '-.g', linewidth =2.0, label='id_loss')
    # ax.plot(x, po_loss, '--r', linewidth =2.0, label='po_loss')
    # ax.set_xlabel('Epoch',fontsize=font_size)
    # ax.set_ylabel('Loss',fontsize=font_size)
    # ax.legend(loc=(0.75, 0.13))

    # ax2=ax.twinx()
    ax2=ax
    ax2.plot(x, id_acc, '-c', linewidth =2.0, label='id_acc')
    ax2.plot(x, po_acc, '--b', linewidth =2.0, label='po_acc')
    ax2.set_xlabel('Epoch', fontsize=font_size)
    ax2.set_ylabel('Accuracy',fontsize=font_size)
    ax2.legend(loc=(0.75,0.77))

    # plt.title('Loss and accuracy curve of FDCA on Market1501',fontsize=font_size)
    plt.show()
    pass


if __name__ == '__main__':
    read_log()
    pass
