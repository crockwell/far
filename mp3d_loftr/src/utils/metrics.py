import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from src.losses.loftr_loss import rotation_6d_to_matrix, pose_mean_6d, pose_std_6d
from scipy.interpolate import interpn
import random
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import numeric

random.seed(0)
np.random.seed(0)

# --- METRICS ---

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # absolute error between 2 vectors
    t_err_abs = np.linalg.norm(t - t_gt)

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err, t_err_abs


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d

def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f'].detach()
    pts1 = data['mkpts1_f'].detach()

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, 
                  translation_scale=None, 
                  solver='ransac', priorRT=None):
    if len(kpts0) < 5:
        print("less than 5 keypoints, returning none")
        return None, 0, 0, 0
    
    # normalize keypoints
    kpts0_norm = ((kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]).detach().cpu().numpy()
    kpts1_norm = ((kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]).detach().cpu().numpy()

    K0_numpy, K1_numpy = K0.cpu().numpy(), K1.cpu().numpy()

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0_numpy[0, 0], K1_numpy[1, 1], K0_numpy[0, 0], K1_numpy[1, 1]])

    inliers_best_tight, inliers_best_ultra_tight = 0, 0

    num_correspondences_after_ransac = 0

    if solver == 'prior_ransac' and priorRT is not None:
        from third_party.prior_ransac.ransac import RANSAC
        #import pdb; pdb.set_trace()
        random_pcl = np.random.uniform(low=-3.0, high=3.0, size=(300, 3)).astype(np.float32)
        prior_params = {
            'rotation_pcl_error': True, 
            'rotation_error' : False,
            'K1': K0.float(),
            'K2': K1.float(),
            'RT': torch.FloatTensor(priorRT).cuda(),
            'pcl': torch.FloatTensor(random_pcl).cuda(),
            'lambda': 0.3,
            'biased_sampling': 'biased'
        }
        ransac_model = RANSAC(
            model_type='essential_cv2',
            max_iter=1, 
            inl_th=3E-7, 
            prior_params=prior_params,
            max_lo_iters=0,
            batch_size=2048,
            use_noexp_prior_scoring=True,
            use_linear_bias_sampling=True,
            bias_sigma_sq=0.1,
        )   
        
        kp2 = torch.FloatTensor(kpts1_norm).cuda()
        kp1 = torch.FloatTensor(kpts0_norm).cuda()
        E, mask, inliers_best_tight, inliers_best_ultra_tight = ransac_model.forward(kp1=kp1, kp2=kp2)
        E = E.cpu().numpy()
        mask = mask.unsqueeze(1).cpu().numpy().astype(np.uint8)
        inliers_best_tight = inliers_best_tight.sum().cpu().item()
        inliers_best_ultra_tight = inliers_best_ultra_tight.sum().cpu().item()
    elif solver == 'prior_ransac_noprior':
        from third_party.prior_ransac.ransac import RANSAC
        random_pcl = np.random.uniform(low=-3.0, high=3.0, size=(300, 3)).astype(np.float32)
        ransac_model = RANSAC(
            model_type='essential_cv2',
            max_iter=1, 
            inl_th=3E-7, 
            prior_params={},
            max_lo_iters=0,
            batch_size=2048,
            use_noexp_prior_scoring=False,
            use_linear_bias_sampling=False,
        )   
        
        kp2 = torch.FloatTensor(kpts1_norm).cuda()
        kp1 = torch.FloatTensor(kpts0_norm).cuda()
        E, mask, inliers_best_tight, inliers_best_ultra_tight = ransac_model.forward(kp1=kp1, kp2=kp2)
        E = E.cpu().numpy()
        mask = mask.unsqueeze(1).cpu().numpy().astype(np.uint8)
        inliers_best_tight = inliers_best_tight.sum().cpu().item()
        inliers_best_ultra_tight = inliers_best_ultra_tight.sum().cpu().item()
    else:
        E, mask = cv2.findEssentialMat(
            kpts0_norm, kpts1_norm, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None, 0, 0, 0

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E.astype(np.float64), kpts0_norm, kpts1_norm, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            if translation_scale is not None:
                t *= translation_scale.cpu().numpy()
            ret = (torch.from_numpy(R).cuda(), torch.from_numpy(t[:, 0]).cuda(), mask.ravel() > 0, torch.from_numpy(_E))
            best_num_inliers = n

    num_correspondences_after_ransac = torch.from_numpy(mask).sum()

    return ret, num_correspondences_after_ransac, inliers_best_tight, inliers_best_ultra_tight

def compute_correspondences_feats(kpts0, kpts1, feats):
    kpts0_lookup = torch.stack([torch.clamp((kpts0/8)[:,0].long(), 0, 79), torch.clamp((kpts0/8)[:,1].long(), 0, 59)], dim=-1)
    kpts1_lookup = torch.stack([torch.clamp((kpts1/8)[:,0].long(), 0, 79), torch.clamp((kpts1/8)[:,1].long(), 0, 59)], dim=-1)
    kpts_feats0 = feats[0,0,kpts0_lookup[:,1],kpts0_lookup[:,0]]
    kpts_feats1 = feats[0,1,kpts1_lookup[:,1],kpts1_lookup[:,0]]

    correspondences_feats = torch.stack([kpts_feats0, kpts_feats1],dim=1) # N, 2, 256
    return correspondences_feats

def compute_correspondences_depths(kpts0, kpts1, depths):
    kpts0_lookup = torch.stack([torch.clamp(kpts0[:,1], 0, 479), torch.clamp(kpts0[:,0], 0, 639)], dim=-1).cpu().numpy()
    kpts1_lookup = torch.stack([torch.clamp(kpts1[:,1], 0, 479), torch.clamp(kpts1[:,0], 0, 639)], dim=-1).cpu().numpy()
    grid = (np.linspace(0,depths.shape[-2]-1, depths.shape[-2]), np.linspace(0,depths.shape[-1]-1,depths.shape[-1]))
    kpts_depth0 = interpn(grid, depths[0,0].cpu().numpy(), kpts0_lookup)
    kpts_depth1 = interpn(grid, depths[0,1].cpu().numpy(), kpts1_lookup)
    kpts_depth0 = torch.from_numpy(kpts_depth0).to(device=kpts0.device)
    kpts_depth1 = torch.from_numpy(kpts_depth1).to(device=kpts0.device)

    correspondences_depths = torch.stack([kpts_depth0, kpts_depth1], dim=1).unsqueeze(-1) # N, 2, 1

    return correspondences_depths

def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 
                 't_errs': [], 
                 't_errs_abs': [], 
                 'inliers': [], 
                 'successful_fits': [],
                 'pred_R': [],
                 'pred_t': [],
                 'num_correspondences_before_ransac': [],
                 'num_correspondences_after_ransac': []})

    K0 = data['K0']
    K1 = data['K1']
    T_0to1 = data['T_0to1'].cpu().numpy()

    if config.LOFTR.SOLVER == 'prior_ransac' and 'priorRT' in data:
        priorRT = data['priorRT']
        if torch.is_tensor(priorRT):
            priorRT = priorRT.cpu().numpy()[0]
    else:
        priorRT = None

    for bs in range(K0.shape[0]):           
        if 'regressed_rt' in data:
            R = data['regressed_rt'][:,3:].cpu() * pose_std_6d[3:] + pose_mean_6d[3:]
            t = data['regressed_rt'][0,:3].cpu().numpy()* pose_std_6d[:3].numpy() + pose_mean_6d[:3].numpy()
            R = rotation_6d_to_matrix(R)[0].numpy()
            inliers = 0
            data['successful_fits'].append(0)
        elif 'mkpts0_c' in data or 'mkpts0_f' in data:
            m_bids = data['m_bids']
            pts0 = data['mkpts0_f']
            pts1 = data['mkpts1_f']

            depths = None
            np.random.seed(0)

            mask = m_bids == bs

            ret, num_correspondences_after_ransac, inliers_best_tight, inliers_best_ultra_tight = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], 
                                pixel_thr, conf=conf, 
                                translation_scale=data['translation_scale'], 
                                solver=config.LOFTR.SOLVER,
                                priorRT=priorRT)
            # new code
            if ret is None:
                ret = (np.eye(3), np.random.rand(3)-.5, np.zeros(mask.shape[0]), np.eye(3))
                data['successful_fits'].append(0)
                print("failed fit")
            else:
                ret_new = (ret[0].cpu().numpy(), ret[1].cpu().numpy(), ret[2], ret[3].cpu().numpy())
                ret = ret_new
                data['successful_fits'].append(1)
                data['num_correspondences_before_ransac'].append(len(pts0[mask]))
                data['num_correspondences_after_ransac'].append(num_correspondences_after_ransac)

            R, t, inliers, E = ret

            if config.SAVE_PREDS is not None:
                if 'ground_truth' in config.SAVE_PREDS or config.SAVE_HARD_CORRES:
                    if config.SAVE_CORR_AFTER_RANSAC:
                        #print(pts0.shape,inliers.sum())
                        corr0, corr1 = pts0[mask][inliers], pts1[mask][inliers]
                    else:
                        corr0, corr1 = pts0[mask], pts1[mask]
                    feats = torch.stack([data['featmap0'], data['featmap1']],dim=1).reshape([1,2,60,80,256])
                    if 'ground_truth' in config.SAVE_PREDS:
                        data['correspondences_depths'] = compute_correspondences_depths(corr0, corr1, depths)
                    
                    data['correspondences_feats'] = compute_correspondences_feats(corr0, corr1, feats)
                    data['correspondences'] = torch.cat([corr0, corr1], dim=-1).reshape([-1,2,2])
                    #if data['pair_id'].item() == 241:
                    #    import pdb; pdb.set_trace()
                else:
                    data['correspondences'] = torch.cat([pts0[mask], pts1[mask], torch.from_numpy(inliers).float().unsqueeze(1).cuda()], dim=-1).cpu()
        else:
            ret = (np.eye(3), np.random.rand(3)-.5, np.zeros(0), np.eye(3))
            R, t, inliers, E = ret
            data['successful_fits'].append(0)
            inliers, optimal_error, error = 0, 0, 0
        
        t_err, R_err, t_err_abs = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)

        if False:
            print('err',R_err, t_err, t_err_abs)
            print('pred\n',R)
            print('gt\n',T_0to1[bs])
            #print('pred 6d\n',data['regressed_rt'])
            breakpoint()

        data['pred_R'] = R
        data['pred_t'] = t
        data['R_errs'].append(R_err)
        data['t_errs'].append(t_err)
        data['t_errs_abs'].append(t_err_abs)
        data['inliers'].append(inliers)

# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs

def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]
    pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    # matterport metrics
    metrics['t_errs'] = np.array(metrics['t_errs'])
    metrics['R_errs'] = np.array(metrics['R_errs'])
    metrics['t_errs_abs'] = np.array(metrics['t_errs_abs'])
    metrics['successful_fits'] = np.array(metrics['successful_fits'])
    matterport_metrics = {'tr rot mean err': np.round(np.mean(metrics['t_errs']), 2), 
                          'tr rot median err': np.round(np.median(metrics['t_errs']), 2),
                          'tr rot pct < 30': np.round(100 * np.mean(metrics['t_errs'] < 30), 1),
                          'tr abs mean err': np.round(np.mean(metrics['t_errs_abs']), 2), 
                          'tr abs median err': np.round(np.median(metrics['t_errs_abs']), 2),
                          'tr abs pct < 1': np.round(100 * np.mean(metrics['t_errs_abs'] < 1), 1),
                          'rot mean err': np.round(np.mean(metrics['R_errs']), 2), 
                          'rot median err': np.round(np.median(metrics['R_errs']), 2),
                          'rot pct < 30': np.round(100 * np.mean(metrics['R_errs'] < 30), 1),
                          'pct successful fits': np.round(100 * np.mean(metrics['successful_fits']), 1),
                          'dset size': len(metrics['t_errs']),
    }

    return {**matterport_metrics, **aucs, **precs, }

def compute_angle_from_r_matrices(m):
    batch = m.shape[0]

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch)) * -1)

    theta = torch.acos(cos)

    return theta

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch)) * -1)

    theta = torch.acos(cos)

    return theta

def aggregate_metrics_interiornet_streetlearn(metrics, epi_err_thr=5e-4):
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)

    # interiornet / streetlearn metrics
    gt_rotation = torch.stack(metrics['gt_R']).float()
    predict_rotation = torch.stack(metrics['pred_R']).float()

    #import pdb; pdb.set_trace()

    geodesic_loss = compute_geodesic_distance_from_two_matrices(predict_rotation.view([-1, 3, 3]),
                                                                gt_rotation.view([-1, 3, 3])) / np.pi * 180
    gt_distance = compute_angle_from_r_matrices(gt_rotation.view(-1, 3, 3))

    geodesic_loss_overlap_large = geodesic_loss[gt_distance.view(-1) < (np.pi / 4)].cpu().numpy()
    geodesic_loss_overlap_small = geodesic_loss[(gt_distance.view(-1) >= np.pi / 4) & (gt_distance.view(-1) < np.pi / 2)].cpu().numpy()

    metrics['successful_fits'] = np.array(metrics['successful_fits'])
    interiornet_streetlearn_metrics = {
                          'large overlap, rot mean err': np.round(np.mean(geodesic_loss_overlap_large), 2), 
                          'large overlap, rot median err': np.round(np.median(geodesic_loss_overlap_large), 2),
                          'large overlap, rot pct < 10': np.round(100 * np.mean(geodesic_loss_overlap_large < 10), 1),
                          'small overlap, rot mean err': np.round(np.mean(geodesic_loss_overlap_small), 2), 
                          'small overlap, rot median err': np.round(np.median(geodesic_loss_overlap_small), 2),
                          'small overlap, rot pct < 10': np.round(100 * np.mean(geodesic_loss_overlap_small < 10), 1),
                          'pct successful fits': np.round(100 * np.mean(metrics['successful_fits']), 1),
    }

    return {**interiornet_streetlearn_metrics, **precs, }
