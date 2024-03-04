from os import path as osp

import numpy as np
import torch
import pickle as pkl
import torch.utils as utils
from src.utils.dataset import (
    read_scannet_gray,
    get_mp3d_intrinsics,
    read_mp3d_depth, 
    get_mp3d_T_0to1,
)
import json
import os
from tqdm import tqdm
import random

random.seed(0)
np.random.seed(0)

class Mp3dDataset(utils.data.Dataset):
    def __init__(self, json_path, depth_dir, data_dir, mode='train', split='train',
                 load_predictions_path=None, 
                 from_saved_preds=None, 
                 from_saved_corr=None, full_train_set=False, 
                 use_large_dset=False, 
                 use_40pct_dset=False, load_prior_ransac=None):
        super().__init__()

        if use_large_dset:
            json_path = json_path[:-5] + "_large.json" # train is 83k
            '''
            generated on megaflop via https://github.com/jinlinyi/p-sparse-plane/blob/d8d9514ba3c3b23a0f7de21e972173010fc652be/planeRecon/utils/gt_correspondence_coco_ply.py#L266-L293
            on conda env sp3, using pytorch3d v0.2.5 and not using the replaced files described in the p-sparse-plane repo
            command: sh run.sh in folder /Pool1/users/cnris/samir/p-sparse-plane_fixed 
            '''

        print("using json path", json_path)

        with open(json_path, 'r') as f:
            self.data = json.load(f)['data']
        
        if mode == "test" and "train" in json_path and not full_train_set:
            # use only 10% of the data for testing
            self.data = self.data[::10]

        if use_40pct_dset and ("train" in json_path or "val" in json_path) and mode == 'train':
            indices = np.round(np.arange(0, len(self.data), 2.5)).astype(np.int32)
            self.data = np.array(self.data)[indices].tolist()
            print("using small dataset, size", len(self.data))

        self.mode = mode
        if mode == 'test':
            self.split = split
        else:
            self.split = mode
        
        self.depth_dir = depth_dir
        self.data_dir = data_dir

        self.K = get_mp3d_intrinsics()

        if load_predictions_path is not None:
            self.loaded_predictions = pkl.load(open(load_predictions_path,"rb"))
        else:
            self.loaded_predictions = None

        self.from_saved_preds = None
        if from_saved_preds is not None:
            self.from_saved_preds = from_saved_preds
            self.from_saved_corr = self.from_saved_preds

            if from_saved_corr is not None:
                self.from_saved_corr = from_saved_corr

            print(f"loading preds: {self.from_saved_preds}\ncorr: {self.from_saved_corr}")

        print("\n\n using " + f"{self.split} split, {json_path} json" + "\n\n")

        self.prior_ransac = None
        if load_prior_ransac is not None:
            self.prior_ransac = load_prior_ransac

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        passed = False

        while not passed:
            sample = self.data[idx]
            scene_name = osp.basename(osp.dirname(sample['0']['file_name']))

            # read the grayscale image which will be resized to (1, 480, 640)
            img_name0 = osp.join(self.data_dir, "/".join(sample['0']['file_name'].split("/")[-3:]))
            img_name1 = osp.join(self.data_dir, "/".join(sample['1']['file_name'].split("/")[-3:]))

            try:
                # TODO: Support augmentation & handle seeds for each worker correctly.
                image0 = read_scannet_gray(img_name0, resize=(640, 480), augment_fn=None)
                image1 = read_scannet_gray(img_name1, resize=(640, 480), augment_fn=None)
            except:
                idx = (idx + 1) % len(self.data)
                continue

            passed = True

            # read the depthmap which is stored as (480, 640)
            if self.mode in ['train', 'val']:
                depth0 = read_mp3d_depth(self.depth_dir, scene_name, img_name0)
                depth1 = read_mp3d_depth(self.depth_dir, scene_name, img_name1)
            else:
                depth0 = depth1 = torch.tensor([])

            # read the intrinsic of depthmap
            K_0 = K_1 = self.K

            # read and compute relative poses
            T_0to1 = get_mp3d_T_0to1(sample['rel_pose'])
            T_1to0 = T_0to1.inverse()

            if self.loaded_predictions is not None:
                loaded_preds = get_mp3d_T_0to1({"position": self.loaded_predictions['camera']['preds']['tran'][idx], 
                                "rotation": self.loaded_predictions['camera']['preds']['rot'][idx]})
            else:
                loaded_preds = torch.tensor([])

            lightweight_numcorr = torch.tensor([0])

            data = {
                'image0': image0,   # (1, h, w)
                'depth0': depth0,   # (h, w)
                'image1': image1,
                'depth1': depth1,
                'T_0to1': T_0to1,   # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'dataset_name': 'mp3d',
                'scene_id': scene_name,
                'pair_id': idx,
                'pair_names': (sample['0']['file_name'], sample['1']['file_name']),
                'loaded_predictions': loaded_preds,
                'lightweight_numcorr': lightweight_numcorr,
            }

            if self.prior_ransac is not None:
                prior_ransac = torch.load(os.path.join(self.prior_ransac, str(idx)+'.pt'))
                data.update({
                    'priorRT': prior_ransac,
                })

            if self.from_saved_preds is not None:
                preds_path = os.path.join(self.from_saved_preds, self.split, 'loftr_preds', str(idx)+'.pt')
                corres_path = os.path.join(self.from_saved_corr, self.split, 'loftr_num_correspondences', str(idx)+'.pt')
                data.update({
                    'loftr_rt': torch.load(preds_path),
                    'num_correspondences': torch.load(corres_path),
                })
        
        return data

class Mp3dLightDataset(utils.data.Dataset):
    def __init__(self, json_path, mode='train', split='train', correspondences_use_fit_only=False,
                 correspondence_transformer_load_feats=False, 
                 max_correspondences=2000, use_pred_corr=False, outlier_pct=0,
                 noise_pix=0, missing_pct=0, use_large_dset=False, 
                 corr_dropout=0.0,from_saved_preds=None, no_use_loftr_preds=False):
        super().__init__()

        if use_large_dset:
            json_path = json_path[:-5] + "_large.json" # train is 83k

        with open(json_path, 'r') as f:
            self.data = json.load(f)['data']
        
        if mode == "test" and "train" in json_path:
            # use only 10% of the data for testing
            self.data = self.data[::10]
        
        self.mode = mode
        if mode == 'test':
            self.split = split
        else:
            self.split = mode
            if 'val' in json_path:
                self.split = 'val'
        self.K = get_mp3d_intrinsics()

        self.correspondence_setting = "correspondences"
        if correspondences_use_fit_only:
            self.correspondence_setting = "correspondences_fit_only"

        self.from_saved_preds = from_saved_preds

        print(f"starting from saved preds {from_saved_preds}")

        self.no_use_loftr_preds = no_use_loftr_preds
        self.use_pred_corr = use_pred_corr
        if use_pred_corr:
            if correspondence_transformer_load_feats:
                self.correspondence_details_parent = os.path.join(self.from_saved_preds, \
                                                              self.split, 'hard_'+self.correspondence_setting)
            else:
                self.correspondence_details_parent = os.path.join(self.from_saved_preds, \
                                                            self.split, 'loftr_fine_correspondences')
        else:
            self.correspondence_details_parent = os.path.join('/home/cnris/data/mp3d_rpnet_v4_sep20/loftr_preds/ground_truth', self.split, self.correspondence_setting)

        indices_path = os.path.join(self.correspondence_details_parent, 'indices.pt')
        
        if self.split == 'test' and mode == 'test':
            self.existing_indices = np.arange(len(self.data))
        else:
            if not os.path.exists(indices_path):
                existing_indices = []
                for i in tqdm(range(len(self.data))):
                    if use_pred_corr:
                        thispath = os.path.join(self.correspondence_details_parent, str(i)+'.pt')
                        if os.path.exists(thispath):
                            corrs = torch.load(thispath)
                            if len(corrs) > 5:
                                existing_indices.append(i)
                    elif os.path.exists(os.path.join(self.correspondence_details_parent, str(i)+'.pt')):
                        existing_indices.append(i)
                torch.save(existing_indices, indices_path)

            self.existing_indices = torch.load(indices_path)
        print("\n\n"+f"{mode} split on {json_path} json, using {len(self.existing_indices)} of a total possible {len(self.data)} examples"+"\n\n")
        self.data = list(np.array(self.data)[self.existing_indices])

        self.correspondence_transformer_load_feats = correspondence_transformer_load_feats

        self.max_correspondences = max_correspondences

        self.outlier_pct = outlier_pct
        self.noise_pix = noise_pix
        self.missing_pct = missing_pct
        self.corr_dropout = corr_dropout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        scene_name = osp.basename(osp.dirname(sample['0']['file_name']))

        # read the intrinsic of depthmap
        K_0 = K_1 = self.K

        # read and compute relative poses
        T_0to1 = get_mp3d_T_0to1(sample['rel_pose'])
        T_1to0 = T_0to1.inverse()

        load_path = os.path.join(self.correspondence_details_parent, str(self.existing_indices[idx])+'.pt')

        if self.use_pred_corr and not self.no_use_loftr_preds:
            preds_path = os.path.join(self.from_saved_preds, self.split, 'loftr_preds', str(idx)+'.pt')
            corres_path = os.path.join(self.from_saved_preds, self.split, 'loftr_num_correspondences', str(idx)+'.pt')
            load_correspondences = os.path.exists(preds_path)
        else:
            preds_path = None
            corres_path = None
            load_correspondences = True
        
        if os.path.exists(load_path) and load_correspondences:
            correspondence_details = torch.load(load_path)

            if self.missing_pct > 0:
                # randomly remove pct of correspondences
                num_missing = int(self.missing_pct / 100 * len(correspondence_details))
                missing_indices = random.sample(list(np.arange(len(correspondence_details))), num_missing)
                correspondence_details = correspondence_details[~np.isin(np.arange(len(correspondence_details)), missing_indices)]
            
            if self.noise_pix > 0:
                noise = np.random.randn(correspondence_details.shape[0],2) * self.noise_pix
                correspondence_details[:,1,:2] += torch.from_numpy(noise)
            
            if self.outlier_pct > 0:
                # randomly take pct of correspondences and set them to outliers
                num_outliers = int(self.outlier_pct / 100 * len(correspondence_details))
                outlier_indices = random.sample(list(np.arange(len(correspondence_details))), num_outliers)
                replacement_y = np.random.randint(480, size=(num_outliers))
                replacement_x = np.random.randint(640, size=(num_outliers))
                correspondence_details[outlier_indices,1,1] = torch.from_numpy(replacement_y).float()
                correspondence_details[outlier_indices,1,0] = torch.from_numpy(replacement_x).float()

            if self.corr_dropout > 0.0:
                num_drop = int(self.corr_dropout * len(correspondence_details))
                num_drop = min(num_drop, len(correspondence_details) - 5)
                drop_indices = random.sample(list(np.arange(len(correspondence_details))), num_drop)
                possible_correspondence_details = correspondence_details[~np.isin(np.arange(len(correspondence_details)), drop_indices)]
                if len(possible_correspondence_details) >= 5:
                    correspondence_details = possible_correspondence_details
                if correspondence_details.shape[0] <= 5 or correspondence_details.shape[1] < 2:
                    correspondence_details = torch.load(os.path.join(self.correspondence_details_parent, str(self.existing_indices[idx])+'.pt'))

            correspondence_details = correspondence_details[:self.max_correspondences]

            if not self.correspondence_transformer_load_feats:
                correspondence_details = correspondence_details[...,:2]

            # normalize to [0, 1]
            correspondence_details[...,:2] /= 640 # max height or width
            
            if self.use_pred_corr and not self.no_use_loftr_preds:
                preds = torch.load(preds_path)
                corr = torch.load(corres_path)
            else:
                preds = None
                corr = None

        elif not self.correspondence_transformer_load_feats:
            correspondence_details = torch.zeros(0,2,2)
            preds = torch.zeros(0,3,4)
            corr = torch.zeros(0)
        else:
            correspondence_details = torch.zeros(0,2,258)
            preds = torch.zeros(0,3,4)
            corr = torch.zeros(0)

        data = {
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'dataset_name': 'mp3d',
            'scene_id': scene_name,
            'pair_id': self.existing_indices[idx],
            'pair_names': (sample['0']['file_name'], sample['1']['file_name']),
            'correspondence_details': correspondence_details,
            'lightweight_numcorr': torch.tensor([0]),
        }

        if self.use_pred_corr and not self.no_use_loftr_preds: 
            data.update({
                'loftr_rt': preds,
                'num_correspondences': corr,
            })
        else:
            data.update({
                'mkpts0_f': correspondence_details[:,0,:2] * 640,
                'mkpts1_f': correspondence_details[:,1,:2] * 640,
                'm_bids': torch.zeros(len(correspondence_details)),
            })

        return data
