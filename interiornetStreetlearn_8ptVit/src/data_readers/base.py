
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import csv
import os
import cv2
import math
import random
import json
import pickle
import os.path as osp
import time

from .augmentation import RGBDAugmentor

class RGBDDataset(data.Dataset):
    def __init__(self, name, datapath, reshape_size=[384,512], subepoch=None, \
                is_training=True, gpu=0, streetlearn_interiornet_type=None,
                use_mini_dataset=False, use_loftr_gating=False, from_saved_preds=None,
                solver='ransac'):
        """ Base class for RGBD dataset """
        self.root = datapath
        self.name = name
        self.streetlearn_interiornet_type = streetlearn_interiornet_type
        self.use_loftr_gating = use_loftr_gating
        self.from_saved_preds = from_saved_preds
        self.solver = solver
        
        self.aug = RGBDAugmentor(reshape_size=reshape_size, datapath=datapath)

        if 'StreetLearn' in self.name or 'InteriorNet' in self.name:
            self.use_mini_dataset = use_mini_dataset       
            self.scene_info = self._build_dataset(subepoch)     
        else:
            print("not currently setup in case have other dataset type!")
            import pdb; pdb.set_trace()

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    def __getitem__(self, index):
        """ return training video """
        local_index = index
        # in case index fails
        while True:
            try:
                images_list = self.scene_info['images'][local_index]
                poses = self.scene_info['poses'][local_index]
                intrinsics = self.scene_info['intrinsics'][local_index]

                images = []
                for i in range(2):
                    images.append(self.__class__.image_read(images_list[i]))

                poses = np.stack(poses).astype(np.float32)
                intrinsics = np.stack(intrinsics).astype(np.float32)
                
                images = np.stack(images).astype(np.float32)
                images = torch.from_numpy(images).float()
                images = images.permute(0, 3, 1, 2)

                poses = torch.from_numpy(poses)
                intrinsics = torch.from_numpy(intrinsics)

                images, poses, intrinsics = self.aug(images, poses, intrinsics)

                loftr_num_corr = self.scene_info['loftr_num_corr'][local_index]
                loftr_preds = self.scene_info['loftr_preds'][local_index]
                return images, poses, intrinsics, loftr_num_corr, loftr_preds
            except:
                local_index += 1
                continue

    def __len__(self):
        return len(self.scene_info['poses'])
