import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import argparse

import torch 
import torch.nn.functional as F

from src.model import ViTEss
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
import random

random.seed(0)
np.random.seed(0)

def compute_angle_from_r_matrices(m):
    batch = m.shape[0]

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    return theta

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    return theta

def compute_rotation_matrix_from_two_matrices(m1, m2):
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
        return m

def compute_rotation_matrix_from_viewpoint(rotation_x, rotation_y, batch):
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

def evaluation_metric_rotation(predict_rotation, gt_rotation, save_folder=None):
    geodesic_loss = compute_geodesic_distance_from_two_matrices(predict_rotation.view(-1, 3, 3),
                                                                gt_rotation.view(-1, 3, 3)) / np.pi * 180
    gt_distance = compute_angle_from_r_matrices(gt_rotation.view(-1, 3, 3))

    geodesic_loss_overlap_large = geodesic_loss[gt_distance.view(-1) < (np.pi / 4)]
    geodesic_loss_overlap_small = geodesic_loss[(gt_distance.view(-1) >= np.pi / 4) & (gt_distance.view(-1) < np.pi / 2)]

    all_rotation_err = geodesic_loss[gt_distance.view(-1) < (np.pi / 2)] 
    all_rotation_mags_gt = gt_distance[gt_distance.view(-1) < (np.pi / 2)] / np.pi * 180
    '''
    all_rotation_err = all_rotation_err.cpu().numpy().astype(np.float32)
    all_rotation_err_name = os.path.join(save_folder, 'all_rotation_err_degrees.csv')
    np.savetxt(all_rotation_err_name, all_rotation_err, delimiter=',', fmt='%1.5f')

    all_rotation_mags_gt = all_rotation_mags_gt.cpu().numpy().astype(np.float32)
    all_rotation_mags_gt_name = os.path.join(save_folder, 'all_gt_rot_degrees.csv')
    np.savetxt(all_rotation_mags_gt_name, all_rotation_mags_gt, delimiter=',', fmt='%1.5f')
    '''

    res_error = {
        "rotation_geodesic_error_overlap_large": geodesic_loss_overlap_large,
        "rotation_geodesic_error_overlap_small": geodesic_loss_overlap_small,
    }
    return res_error

def eval_camera(predictions, save_folder):

    # convert pred & gt to quaternion
    pred, gt = np.copy(predictions['camera']['preds']['rot']), np.copy(predictions['camera']['gts']['rot'])

    r = R.from_quat(pred)
    r_pred = r.as_matrix()

    r = R.from_quat(gt)
    r_gt = r.as_matrix()

    res_error = evaluation_metric_rotation(torch.from_numpy(r_pred).cuda(), torch.from_numpy(r_gt).cuda())

    all_res = {}
    # mean, median, 10deg
    for k, v in res_error.items():
        v = v.view(-1).detach().cpu().numpy()
        if v.size == 0:
            continue
        mean = np.mean(v)
        median = np.median(v)
        count_10 = (v <= 10).sum(axis=0)
        percent_10 = np.true_divide(count_10, v.shape[0])
        all_res.update({k + '/mean': mean, k + '/median': median, k + '/10deg': percent_10})

    return all_res

def compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size):
    gt_mtx1 = compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3)
    gt_mtx2 = compute_rotation_matrix_from_viewpoint(rotation_x2, rotation_y2, batch_size).view(batch_size, 3, 3)
    gt_rmat_matrix = compute_rotation_matrix_from_two_matrices(gt_mtx2, gt_mtx1).view(batch_size, 3, 3)
    return gt_rmat_matrix   


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument("--datapath")
    parser.add_argument("--weights")
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--exp")
    parser.add_argument("--ckpt")
    parser.add_argument('--dataset', default='interiornet', choices=("interiornet", 'streetlearn'))
    parser.add_argument('--streetlearn_interiornet_type', default='', choices=('',"nooverlap","T",'nooverlapT'))

    # model
    parser.add_argument('--fusion_transformer', action="store_true", default=False)
    parser.add_argument('--fc_hidden_size', type=int, default=512)
    parser.add_argument('--pool_size', type=int, default=60)
    parser.add_argument('--transformer_depth', type=int, default=6)

    parser.add_argument('--losson6d', action='store_true')
    parser.add_argument('--use_normalized_6d', action='store_true')

    # loftr
    parser.add_argument('--use_loftr_gating', action='store_true', default=False)
    parser.add_argument("--from_saved_preds")
    parser.add_argument("--solver", type=str, default="ransac")

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    # for normalizing 6D 
    if args.dataset == 'interiornet':
        if args.streetlearn_interiornet_type == 'T':
            global_pose_mean = torch.tensor([0,0,0,0.92456496, -0.00201821, -0.00987212, -0.00019313, 0.72139406, -0.00184757]).cuda()
            global_pose_std = torch.tensor([1,1,1,0.07689704, 0.17564303, 0.32912105, 0.1753406, 0.27482772, 0.6109926]).cuda()
    elif args.dataset == 'streetlearn':
        if args.streetlearn_interiornet_type == 'T':
            global_pose_mean = torch.tensor([0,0,0,0.828742, 0.00034936, -0.00100069, -0.00250733,  0.7001684, -0.00283758]).cuda()
            global_pose_std = torch.tensor([1,1,1,0.16392577, 0.2663457, 0.46407992, 0.26599622, 0.27905113, 0.60093635]).cuda()


    if args.dataset == 'interiornet':
        if args.streetlearn_interiornet_type == 'T':
            dset = np.load(os.path.join(args.datapath, 'metadata/interiornetT/test_pair_translation.npy'), allow_pickle=True)
            output_folder = 'interiornetT_test'
        else:
            raise NotImplementedError()
    else:
        if args.streetlearn_interiornet_type == 'T':
            dset = np.load(os.path.join(args.datapath, 'metadata/streetlearnT/test_pair_translation.npy'), allow_pickle=True)
            output_folder = 'streetlearnT_test'
            args.dataset = 'streetlearn_2016'
        else:
            raise NotImplementedError()

    dset = np.array(dset, ndmin=1)[0]

    print('performing evaluation on %s set using model %s' % (output_folder, args.ckpt))

    try:
        os.makedirs(os.path.join('output', args.exp, output_folder))
    except:
        pass

    T_pose_np  = np.array([[1,0,0],[0,1,0], [0,0,1]])
    T_pose = torch.FloatTensor(T_pose_np).cuda()
    args.T_pose = T_pose

    model = ViTEss(args, global_pose_mean=global_pose_mean, global_pose_std=global_pose_std)
    state_dict = OrderedDict([
        (k.replace("module.", ""), v) for (k, v) in torch.load(args.ckpt)['model'].items()])
    model.load_state_dict(state_dict)
    model = model.cuda().eval()
    
    train_val = ''
    predictions = {'camera': {'preds': {'tran': [], 'rot': []}, 'gts': {'tran': [], 'rot': []}}}

    sorted(dset.keys())

    r_gt, r_pred = [], []
    for i, dset_i in tqdm(sorted(dset.items())[:1000]):
        base_pose = np.array([0,0,0,0,0,0,1])

        images = [cv2.imread(os.path.join(args.datapath, 'data', args.dataset, dset[i]['img1']['path'])),
                    cv2.imread(os.path.join(args.datapath, 'data', args.dataset, dset[i]['img2']['path']))]
        
        x1, y1 = dset[i]['img1']['x'], dset[i]['img1']['y']
        x2, y2 = dset[i]['img2']['x'], dset[i]['img2']['y']

        # compute rotation matrix
        gt_rmat = compute_gt_rmat(torch.tensor([[x1]]), torch.tensor([[y1]]), torch.tensor([[x2]]), torch.tensor([[y2]]), 1).double()

        # get quaternions from rotation matrix
        r = R.from_matrix(gt_rmat)
        rotation = r.as_quat()[0]

        images = np.stack(images).astype(np.float32)
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)
        images = images.unsqueeze(0).cuda()

        intrinsics = np.stack([np.array([[128,128,128,128], [128,128,128,128]])]).astype(np.float32)
        intrinsics = torch.from_numpy(intrinsics).cuda()          
        
        if args.use_loftr_gating:
            pred_path = os.path.join(args.from_saved_preds, 'test', 'loftr_preds', str(i)+'.pt')
            num_corr_path = os.path.join(args.from_saved_preds, 'test', 'loftr_num_correspondences', str(i)+'.pt')
            if os.path.exists(pred_path) and os.path.exists(num_corr_path):
                loftr_preds = torch.load(pred_path).unsqueeze(0).cuda()
                loftr_num_corr = torch.load(num_corr_path).unsqueeze(0).cuda()
                # undo coordinate conversion we performed in loftr
                loftr_preds = loftr_preds.cpu().numpy()[0]
                T = np.eye(4)
                T[:3] = loftr_preds
                flip_axis = np.array([[1,0,0,0], [0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # mp3d
                T = flip_axis@T@np.linalg.inv(flip_axis)
                flip_axis = np.array([[0,1,0,0], [1,0,0,0],[0,0,-1,0],[0,0,0,1]]) # interiornet
                T = flip_axis@T@np.linalg.inv(flip_axis)
                loftr_preds = torch.from_numpy(T.astype(np.double)).unsqueeze(0).cuda()
            else:
                loftr_preds = torch.eye(4)[:3].unsqueeze(0).cuda()
                loftr_num_corr = torch.tensor([0]).cuda()
        else:
            loftr_preds, loftr_num_corr = None, None

        with torch.no_grad():
            tran_preds, rot_preds, rot_preds_mtx, rot_preds_6d = model(images, intrinsics=intrinsics, \
                                                                    loftr_num_corr=loftr_num_corr, \
                                                                        loftr_preds=loftr_preds)

        predictions['camera']['gts']['tran'].append(np.array([0,0,0]))
        predictions['camera']['gts']['rot'].append(rotation)

        if args.use_normalized_6d:
            rot_preds_6d = rot_preds_6d * global_pose_std[3:] + global_pose_mean[3:]
            tran_preds = tran_preds * global_pose_std[:3] + global_pose_mean[:3]
            rot_quats = R.as_quat(R.from_matrix(rotation_6d_to_matrix(rot_preds_6d[0]).data.cpu().numpy()))
            preds = np.concatenate([tran_preds[0].data.cpu().numpy(), rot_quats])
        else:
            if args.losson6d:
                rot_quats = R.as_quat(R.from_matrix(rotation_6d_to_matrix(rot_preds_6d[0]).data.cpu().numpy()))
            else:
                rot_quats = R.as_quat(R.from_matrix(rot_preds_mtx[0].data.cpu().numpy()))
            preds = np.concatenate([tran_preds[0].data.cpu().numpy(), rot_quats])
            pr_copy = np.copy(preds)

        predictions['camera']['preds']['tran'].append(preds[:3])
        predictions['camera']['preds']['rot'].append(preds[3:])

    full_output_folder = os.path.join('output', args.exp, output_folder)
    camera_metrics = eval_camera(predictions, full_output_folder)

    for k in camera_metrics:
        print(k, camera_metrics[k])

    with open(os.path.join(full_output_folder, 'results.txt'), 'w') as f:
        for k in camera_metrics:
            print(k, camera_metrics[k], file=f)
