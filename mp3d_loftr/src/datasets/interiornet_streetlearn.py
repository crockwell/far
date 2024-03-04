from os import path as osp

import numpy as np
import torch
import pickle as pkl
import torch.utils as utils
from src.utils.dataset import (
    read_scannet_gray,
    get_interiornet_streetlearn_intrinsics,
    get_interiornet_streetlearn_T_0to1,
)
import os
import random

random.seed(0)
np.random.seed(0)

class InteriornetStreetlearnDataset(utils.data.Dataset):
    def __init__(self, numpy_path, dset_name, mode='train', split='train',
                 load_predictions_path=None, 
                 from_saved_preds=None, 
                 full_train_set=False):
        super().__init__()

        self.data = np.load(os.path.join('/home/cnris/vl/ExtremeRotation_code',numpy_path), allow_pickle=True)
        self.data = np.array(self.data, ndmin=1)[0]

        self.img_root = os.path.join("/home/cnris/vl/ExtremeRotation_code/data",dset_name)
        
        if (mode == "test" and "train" in numpy_path and not full_train_set) or mode == 'val':
            # use only 1% of the data for testing
            data_new = {}
            idx=0
            for i in np.arange(0,len(self.data),100):
                data_new[idx] = self.data[i]
                idx += 1
            self.data = data_new
            print("subsampling, only one in every 100")
        elif mode == "test" and "test" in numpy_path:
            dset = sorted(self.data.items())[:1000]
            self.data = {}
            for i, it in dset:
                self.data[i] = it
            print("subsampling, only first 1000")
        else:
            print("no subsampling!!")

        self.mode = mode
        if mode == 'test':
            self.split = split
        else:
            self.split = mode
        
        self.depth_dir = None

        self.K = get_interiornet_streetlearn_intrinsics()

        if load_predictions_path is not None:
            self.loaded_predictions = pkl.load(open(load_predictions_path,"rb"))
        else:
            self.loaded_predictions = None

        self.from_saved_preds = from_saved_preds
            
        print("\n\n using " + f"{self.split} split, {numpy_path} json" + "\n\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # read the grayscale image which will be resized to (1, 480, 640)
        img_name0 = osp.join(self.img_root, sample['img1']['path'])
        img_name1 = osp.join(self.img_root, sample['img2']['path'])
        
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0 = read_scannet_gray(img_name0, resize=(640, 480), augment_fn=None)
        image1 = read_scannet_gray(img_name1, resize=(640, 480), augment_fn=None)

        depth0 = depth1 = torch.tensor([])

        # read the intrinsic of depthmap
        K_0 = K_1 = self.K

        # read and compute relative poses
        T_0to1 = get_interiornet_streetlearn_T_0to1(sample)
        T_1to0 = T_0to1.inverse()

        loaded_preds = torch.tensor([])

        data = {
            'image0': image0,   # (1, h, w)
            'depth0': depth0,   # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'dataset_name': 'interiornet_streetlearn',
            'pair_id': idx,
            'pair_names': (img_name0, img_name1),
            'loaded_predictions': loaded_preds,
            'lightweight_numcorr': torch.tensor([0]),
        }

        if self.from_saved_preds is not None:
            preds_path = os.path.join(self.from_saved_preds, self.split, 'loftr_preds', str(idx)+'.pt')
            feats_path = os.path.join(self.from_saved_preds, self.split, 'coarse_features', str(idx)+'.pt')
            corres_path = os.path.join(self.from_saved_preds, self.split, 'loftr_num_correspondences', str(idx)+'.pt')
            featmaps = torch.load(feats_path)
            data.update({
                'loftr_rt': torch.load(preds_path),
                'featmap0': featmaps[0],
                'featmap1': featmaps[1],
                'num_correspondences': torch.load(corres_path),
            })
        
        return data
