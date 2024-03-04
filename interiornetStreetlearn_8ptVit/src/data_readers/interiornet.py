
import numpy as np
import torch
import cv2
import os
import os.path as osp
from .base import RGBDDataset

from scipy.spatial.transform import Rotation as R

class InteriorNet(RGBDDataset):

    def __init__(self, mode='training', **kwargs):
        self.mode = mode

        super(InteriorNet, self).__init__(name='InteriorNet', **kwargs)

    def compute_rotation_matrix_from_two_matrices(self, m1, m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
        return m

    def compute_rotation_matrix_from_viewpoint(self, rotation_x, rotation_y, batch):
        rotax = rotation_x.view(batch, 1).type(torch.FloatTensor)
        rotay = - rotation_y.view(batch, 1).type(torch.FloatTensor)

        c1 = torch.cos(rotax).view(batch, 1)  # batch*1
        s1 = torch.sin(rotax).view(batch, 1)  # batch*1
        c2 = torch.cos(rotay).view(batch, 1)  # batch*1
        s2 = torch.sin(rotay).view(batch, 1)  # batch*1

        # pitch --> yaw
        row1 = torch.cat((c2, s1 * s2, c1 * s2), 1).view(-1, 1, 3)  # batch*1*3
        row2 = torch.cat((torch.autograd.Variable(torch.zeros(s2.size())), c1, -s1), 1).view(-1, 1, 3)  # batch*1*3
        row3 = torch.cat((-s2, s1 * c2, c1 * c2), 1).view(-1, 1, 3)  # batch*1*3

        matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

        return matrix

    def compute_gt_rmat(self, rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size):
        gt_mtx1 = self.compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3)
        gt_mtx2 = self.compute_rotation_matrix_from_viewpoint(rotation_x2, rotation_y2, batch_size).view(batch_size, 3, 3)
        gt_rmat_matrix = self.compute_rotation_matrix_from_two_matrices(gt_mtx2, gt_mtx1).view(batch_size, 3, 3)
        return gt_rmat_matrix    


    def _build_dataset(self, subepoch):
        np.seterr(all="ignore")
        from tqdm import tqdm
        print("Building InteriorNet dataset")

        scene_info = {'images': [], 'poses': [], 'intrinsics': [], 'loftr_num_corr': [], 'loftr_preds': []}
        base_pose = np.array([0,0,0,0,0,0,1])
        
        if self.streetlearn_interiornet_type == '':
            path = 'metadata/interiornet/train_pair_rotation_overlap.npy'
            print('training with no translation')
        else:
            path = 'metadata/interiornetT/train_pair_translation_overlap.npy'
            print('training with translation')

        split = np.load(osp.join(self.root, path), allow_pickle=True)
        split = np.array(split, ndmin=1)[0]

        split_size = len(split.keys()) // 11

        start = split_size * (subepoch)
        end = split_size * (subepoch+1)

        if self.use_mini_dataset:
            start = 0
            end = 32000
        
        total_skipped = 0

        if self.use_loftr_gating:
            loftr_preds_all = torch.load(os.path.join(self.from_saved_preds, 'train', 'loftr_preds.pt')).numpy()
            loftr_num_corr_all = torch.load(os.path.join(self.from_saved_preds, 'train', 'loftr_num_correspondences.pt')).numpy()

        sorted_keys = sorted(split.keys())[start:end]
        for i in tqdm(sorted_keys):  
            images = [os.path.join(self.root, 'data', 'interiornet', split[i]['img1']['path']),
                        os.path.join(self.root, 'data', 'interiornet', split[i]['img2']['path'])]
            
            x1, y1 = split[i]['img1']['x'], split[i]['img1']['y']
            x2, y2 = split[i]['img2']['x'], split[i]['img2']['y']

            # compute rotation matrix
            gt_rmat = self.compute_gt_rmat(torch.tensor([[x1]]), torch.tensor([[y1]]), torch.tensor([[x2]]), torch.tensor([[y2]]), 1)

            # get quaternions from rotation matrix
            r = R.from_matrix(gt_rmat)
            rotation = r.as_quat()[0]

            rel_pose = np.concatenate([np.array([0,0,0]), rotation]) # translation is 0

            poses = np.vstack([base_pose, rel_pose])

            intrinsics = np.array([[128,128,128,128], [128,128,128,128]]) # 256x256 imgs

            if self.use_loftr_gating:
                #breakpoint()
                loftr_preds = loftr_preds_all[i]
                loftr_num_corr = loftr_num_corr_all[i]
                if loftr_preds == torch.zeros(3):
                    total_skipped += 1 
                    continue

                '''
                pred_path = os.path.join(self.from_saved_preds, 'train', 'loftr_preds', str(i)+'.pt')
                corr_path = os.path.join(self.from_saved_preds, 'train', 'loftr_num_correspondences', str(i)+'.pt')
                if not os.path.exists(pred_path) or not os.path.exists(corr_path):
                    total_skipped += 1 
                    continue # skip this one
                loftr_preds = torch.load(pred_path).numpy()
                loftr_num_corr = torch.load(corr_path).numpy()
                '''

                #import pdb; pdb.set_trace()
                # undo coordinate conversion we performed in loftr
                T = np.eye(4)
                T[:3] = loftr_preds
                flip_axis = np.array([[1,0,0,0], [0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # mp3d
                T = flip_axis@T@np.linalg.inv(flip_axis)
                flip_axis = np.array([[0,1,0,0], [1,0,0,0],[0,0,-1,0],[0,0,0,1]]) # interiornet
                T = flip_axis@T@np.linalg.inv(flip_axis)
                loftr_preds = T.astype(np.double)
            else:
                loftr_num_corr = np.array([0,0])
                loftr_preds = np.array([0,0])

            scene_info['loftr_num_corr'] += [loftr_num_corr]
            scene_info['loftr_preds'] += [loftr_preds]

            scene_info['images'].append(images)
            scene_info['poses'] += [poses]
            scene_info['intrinsics'] += [intrinsics] 
            

        print(f'total skipped {total_skipped} of total {end-start}')

        return scene_info

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

