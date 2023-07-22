import os, errno, glob
import os.path as osp
import cv2
import numpy as np
import csv, json
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, pose = [], [], []
        for _, pid, camid, poseid in data:
            pids += [pid]
            cams += [camid]
            pose += [poseid]
        pids = set(pids)
        cams = set(cams)
        pose = set(pose)
        num_pids = len(pids)
        num_cams = len(cams)
        num_pose = len(pose)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams, num_pose

    # def print_dataset_statistics(self):
    #     raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self):
        print("Dataset statistics:")
        if self.poseid:
            print("  --------------------------------------------------")
            print("  subset   | # ids | # images | # cameras | # poses ")
            print("  --------------------------------------------------")
            print("  train    | {:^5d} | {:^8d} | {:^9d} | {:^7d}".
                  format(self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_pose))
            print("  query    | {:^5d} | {:^8d} | {:^9d} | {:^7d}".
                  format(self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_pose))
            print("  gallery  | {:^5d} | {:^8d} | {:^9d} | {:^7d}".
                  format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_pose))
            print("  --------------------------------------------------")
        else:
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # cameras ")
            print("  ----------------------------------------")
            print("  train    | {:^5d} | {:^8d} | {:^9d}".
                  format(self.num_train_pids, self.num_train_imgs, self.num_train_cams))
            print("  query    | {:^5d} | {:^8d} | {:^9d}".
                  format(self.num_query_pids, self.num_query_imgs, self.num_query_cams))
            print("  gallery  | {:^5d} | {:^8d} | {:^9d}".
                  format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams))
            print("  ----------------------------------------")

    def extract_keypoints(self, img_paths, root_path, keypoints_file='train_keypoints.csv'):
        if osp.exists(keypoints_file):
            return
    # ...


    def cluter(self, img_paths, root_path, keypoints_file, poseid_file, cluster_k, flip=True, miss_value=-1):
        if osp.exists(poseid_file):
            print('Reading poseids from {}...'.format(osp.split(poseid_file)[-1]))
            return json.load(open(poseid_file, "r"))
        self.extract_keypoints(img_paths, root_path, keypoints_file)
    # ...


if __name__=='__main__':
    pass
