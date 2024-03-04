from math import log
from loguru import logger
import numpy as np
import torch
from einops import repeat
from kornia.utils import create_meshgrid
import torch.nn.functional as F

try:
    from .geometry import warp_kpts
except:
    from geometry import warp_kpts
from src.utils.metrics import estimate_pose

##############  ↓  Coarse-Level supervision  ↓  ##############

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    if 'image0' in data:
        device = data['image0'].device
        N, _, H0, W0 = data['image0'].shape
        _, _, H1, W1 = data['image1'].shape
    else:
        import pdb; pdb.set_trace()
        device = data['depth0'].device
        N, _, H0, W0 = data['depth0'].shape
        _, _, H1, W1 = data['depth0'].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # normal case
    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    i_ids, j_ids, b_ids = None, None, None
    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i_gt = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    _, w_pt1_i_gt = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    
    w_pt0_c = w_pt0_i_gt / scale1
    w_pt1_c = w_pt1_i_gt / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    b_ids_gt, i_ids_gt = torch.where(correct_0to1 != 0)
    j_ids_gt = nearest_index1[b_ids_gt, i_ids_gt]

    w_pt0_i, w_pt1_i = w_pt0_i_gt, w_pt1_i_gt
    b_ids, i_ids, j_ids = b_ids_gt, i_ids_gt, j_ids_gt



    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    # print(conf_matrix_gt.sum())

    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })

def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth', 'mp3d']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


##############  ↓  Fine-Level supervision  ↓  ##############

@torch.no_grad()
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    scale = config['LOFTR']['RESOLUTION'][1]
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    # 3. compute gt
    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later

    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  # [M, 2]

    data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['mp3d']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError

##############  ↓  RT supervision  ↓  ##############
#@torch.no_grad()
def spvs_RT(data, config):
    """
    Update:
        data (dict):{
            "expec_rt": [expec_rt]}
    """
    # get predicted RT
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']
    K0 = data['K0']
    K1 = data['K1']

    if config.LOFTR.SOLVER == 'prior_ransac' and 'priorRT' in data:
        priorRT = data['priorRT']
    else:
        priorRT = None

    #print("prior RT", priorRT)

    np.random.seed(0)

    pred_rt = None
    pred_e = None
    for bs in range(K0.shape[0]):
        mask = m_bids == bs

        ret, num_correspondences_after_ransac, inliers_best_tight, inliers_best_ultra_tight = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], 
                            pixel_thr, conf=conf, 
                            translation_scale=data['translation_scale'], 
                            solver=config.LOFTR.SOLVER, 
                            priorRT=priorRT)
        if ret is not None:
            pred_rt = torch.cat([ret[0],ret[1].unsqueeze(1)],axis=1)
            pred_e = ret[3]
        else:
            pred_rt = torch.cat([torch.eye(3),torch.zeros([3,1])],axis=1).cuda()
            pred_e = torch.eye(3).cuda()

    data.update({"loftr_rt": pred_rt,
                "expec_rt": pred_rt,
                "expec_e": pred_e,
                'num_correspondences_before_ransac': torch.tensor([len(pts0[mask])]).cuda(),
                'num_correspondences_after_ransac': num_correspondences_after_ransac,
                'num_correspondences': torch.tensor([num_correspondences_after_ransac]).cuda(),
                'inliers_best_tight': torch.tensor([inliers_best_tight]).cuda(),
                'inliers_best_ultra_tight': torch.tensor([inliers_best_ultra_tight]).cuda(),})

def compute_supervision_RT(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['mp3d', 'interiornet_streetlearn']:
        spvs_RT(data, config)
    else:
        raise NotImplementedError
    