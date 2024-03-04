import torch
import numpy as np
import cv2

import sys
sys.path.append("etc/feature_matching_baselines")

from LoFTR.src.loftr import LoFTR, default_cfg
from SuperGlue.models.utils import read_image
from SuperGlue.models.matching import Matching

torch.set_grad_enabled(False)

def compute_correspondences_feats(kpts0, kpts1, feats):
    kpts0_lookup = torch.stack([torch.clamp((kpts0/8)[:,0].long(), 0, 89), torch.clamp((kpts0/8)[:,1].long(), 0, 67)], dim=-1)
    kpts1_lookup = torch.stack([torch.clamp((kpts1/8)[:,0].long(), 0, 89), torch.clamp((kpts1/8)[:,1].long(), 0, 67)], dim=-1)
    kpts_feats0 = feats[0,0,kpts0_lookup[:,1],kpts0_lookup[:,0]]
    kpts_feats1 = feats[0,1,kpts1_lookup[:,1],kpts1_lookup[:,0]]

    correspondences_feats = torch.stack([kpts_feats0, kpts_feats1],dim=1) # N, 2, 256
    return correspondences_feats

class LoFTR_matcher:
    def __init__(self, resize, outdoor=False):
        # Initialize LoFTR
        print("started loading model")
        matcher = LoFTR(config=default_cfg)
        weights_path = "etc/feature_matching_baselines/LoFTR/weights/outdoor_ot.ckpt" if outdoor else "LoFTR/weights/indoor_ot.ckpt"
        matcher.load_state_dict(torch.load(weights_path)['state_dict'], strict=False)
        matcher = matcher.eval()#.cuda()
        self.matcher = matcher
        print("model loaded")
        self.resize = resize

    def match(self, pair_path, save_feats=False):
        '''retrurn correspondences between images (w/ path pair_path)'''

        input_path0, input_path1 = pair_path
        resize = self.resize
        resize_float = True
        rot0, rot1 = 0, 0
        device = 'cuda'

        # using resolution [640, 480] (default for 7Scenes, re-scale Scannet)
        image0, inp0, scales0 = read_image(
            input_path0, device, resize, rot0, resize_float)

        image1, inp1, scales1 = read_image(
            input_path1, device, resize, rot1, resize_float)

        #print(inp0.shape)

        # LoFTR needs resolution multiple of 8. If that is not the case, we pad 0's to get to a multiple of 8
        if inp0.size(2) % 8 != 0 or inp0.size(1) % 8 != 0:
            pad_bottom = inp0.size(2) % 8
            pad_right = inp0.size(3) % 8
            pad_fn = torch.nn.ConstantPad2d((0, pad_right, 0, pad_bottom), 0)
            inp0 = pad_fn(inp0)
            inp1 = pad_fn(inp1)

        #print(inp0.shape)
        #import pdb; pdb.set_trace()

        with torch.no_grad():
            batch = {'image0': inp0, 'image1': inp1}
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f']
            mkpts1 = batch['mkpts1_f']

        # todo: get number of correspondences, predictions from a solver, as well as features associated with each
        # though, will this be too big? Currently 1k,1k,4 -- becomes 1k,1k,520 -- each entry
        # total dset size is ~128G. So divided by 500 videos is 500MB. Seems ok! Might be kinda slow
        # may want to just run with corr only, first

        if mkpts0.shape[0] > 0:
            if save_feats:
                feats = torch.stack([batch['featmap0'], batch['featmap1']],dim=1).reshape([1,2,68,90,256])
                cfeats = compute_correspondences_feats(mkpts0, mkpts1, feats).cpu().numpy()
                pts = np.stack([mkpts0.cpu().numpy(), mkpts1.cpu().numpy()], axis=1)
                pts = np.concatenate([pts, cfeats], axis=2)
                return pts
            else:
                pts = np.concatenate([mkpts0.cpu().numpy(), mkpts1.cpu().numpy()], axis=1)
                return pts
        else:
            print("no correspondences")
            if save_feats:
                return np.full((1, 2, 258), np.nan)
            else:
                return np.full((1, 4), np.nan)

    def match_no_load(self, inp0, inp1):
        with torch.no_grad():
            batch = {'image0': inp0, 'image1': inp1}
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f']
            mkpts1 = batch['mkpts1_f']

        if mkpts0.shape[0] > 0:
            return mkpts0.cpu().numpy(), mkpts1.cpu().numpy()
        else:
            print("no correspondences")
            return np.zeros((1, 2)), np.zeros((1, 2))

class SuperGlue_matcher:
    def __init__(self, resize, outdoor=False):
        # copied default values
        nms_radius = 4
        keypoint_threshold = 0.005
        max_keypoints = 1024

        superglue_weights = 'outdoor' if outdoor else 'indoor'  # indoor trained on scannet
        sinkhorn_iterations = 20
        match_threshold = 0.2

        # Load the SuperPoint and SuperGlue models.
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #print('Running inference on device \"{}\"'.format(device))
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue_weights,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self.matching = Matching(config).eval()#.to(device)
        #self.device = device
        print('SuperGlue model loaded')
        self.resize = resize

    def match(self, pair_path, save_feats=False):
        '''retrurn correspondences between images (w/ path pair_path)'''

        input_path0, input_path1 = pair_path
        resize = self.resize
        resize_float = True
        rot0, rot1 = 0, 0

        image0, inp0, scales0 = read_image(
            input_path0, self.device, resize, rot0, resize_float)
        image1, inp1, scales1 = read_image(
            input_path1, self.device, resize, rot1, resize_float)

        #print(inp0.shape)
        #import pdb; pdb.set_trace()

        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        if mkpts0.shape[0] > 0:
            pts = np.concatenate([mkpts0, mkpts1], axis=1)
            return pts
        else:
            print("no correspondences")
            return np.full((1, 4), np.nan)

    def match_no_load(self, inp0, inp1):
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0] for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches = pred['matches0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        if mkpts0.shape[0] > 0:
            return mkpts0, mkpts1
        else:
            print("no correspondences")
            return np.zeros((1, 2)), np.zeros((1, 2))


class SIFT_matcher:
    def __init__(self, resize, outdoor=False):
        self.resize = resize

    def root_sift(self, descs):
        '''Apply the Hellinger kernel by first L1-normalizing, taking the square-root, and then L2-normalizing'''

        eps = 1e-7
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        return descs

    def match(self, pair_path):
        '''
        Given path to im1, im2, extract correspondences using OpenCV SIFT.
        Returns: pts (N x 4) array containing (x1, y1, x2, y2) correspondences; returns nan array if no correspondences.
        '''

        im1_path, im2_path = pair_path

        # hyper-parameters
        ratio_test_threshold = 0.8
        n_features = 2048
        sift = cv2.SIFT_create(n_features)

        # Read images in grayscale
        img0 = cv2.imread(im1_path, 0)
        img1 = cv2.imread(im2_path, 0)

        # Resize
        img0 = cv2.resize(img0, self.resize)
        img1 = cv2.resize(img1, self.resize)

        # get SIFT key points and descriptors
        kp0, des0 = sift.detectAndCompute(img0, None)
        kp1, des1 = sift.detectAndCompute(img1, None)

        # Apply normalisation (rootSIFT)
        des0, des1 = self.root_sift(des0), self.root_sift(des1)

        # Get matches using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des0, des1, k=2)

        pts1 = []
        pts2 = []
        good_matches = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < ratio_test_threshold * n.distance:
                pts2.append(kp1[m.trainIdx].pt)
                pts1.append(kp0[m.queryIdx].pt)
                good_matches.append(m)

        pts1 = np.float32(pts1).reshape(-1, 2)
        pts2 = np.float32(pts2).reshape(-1, 2)

        if pts1.shape[0] > 0:
            pts = np.concatenate([pts1, pts2], axis=1)
            return pts
        else:
            print("no correspondences")
            return np.full((1, 4), np.nan)
