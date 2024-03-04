from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np
from transforms3d.quaternions import qinverse, qmult, rotate_vector, quat2mat

from lib.datasets.utils import read_color_image, read_depth_image, correct_intrinsic_scale, read_scannet_gray
from etc.feature_matching_baselines.SuperGlue.models.utils import read_image


class MapFreeScene(data.Dataset):
    def __init__(
            self, scene_root, resize, sample_factor=1, overlap_limits=None, transforms=None,
            estimated_depth=None, use_loftr_preds=False, use_superglue_preds=False, resize_matcher=(540,720)):
        super().__init__()

        self.scene_root = Path(scene_root)
        self.resize = resize
        self.resize_matcher = resize_matcher
        self.sample_factor = sample_factor
        self.transforms = transforms
        self.estimated_depth = estimated_depth

        # load absolute poses
        self.poses = self.read_poses(self.scene_root)

        # read intrinsics
        self.K = self.read_intrinsics(self.scene_root, resize)
        self.K_loftr = self.read_intrinsics(self.scene_root, resize_matcher)

        # load pairs
        self.pairs = self.load_pairs(self.scene_root, overlap_limits, self.sample_factor)
        self.use_loftr_preds = use_loftr_preds
        self.use_superglue_preds = use_superglue_preds

    @staticmethod
    def read_intrinsics(scene_root: Path, resize=None):
        Ks = {}
        with (scene_root / 'intrinsics.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue

                line = line.strip().split(' ')
                img_name = line[0]
                fx, fy, cx, cy, W, H = map(float, line[1:])

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                if resize is not None:
                    K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H)
                Ks[img_name] = K
        return Ks

    @staticmethod
    def read_poses(scene_root: Path):
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """
        poses = {}
        with (scene_root / 'poses.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue

                line = line.strip().split(' ')
                img_name = line[0]
                qt = np.array(list(map(float, line[1:])))
                poses[img_name] = (qt[:4], qt[4:])
        return poses

    def load_pairs(self, scene_root: Path, overlap_limits: tuple = None, sample_factor: int = 1):
        """
        For training scenes, filter pairs of frames based on overlap (pre-computed in overlaps.npz)
        For test/val scenes, pairs are formed between keyframe and every other sample_factor query frames.
        If sample_factor == 1, all query frames are used. Note: sample_factor applicable only to test/val
        Returns:
        pairs: nd.array [Npairs, 4], where each column represents seaA, imA, seqB, imB, respectively
        """
        overlaps_path = scene_root / 'overlaps.npz'

        if overlaps_path.exists():
            f = np.load(overlaps_path, allow_pickle=True)
            idxs, overlaps = f['idxs'], f['overlaps']
            if overlap_limits is not None:
                min_overlap, max_overlap = overlap_limits
                mask = (overlaps > min_overlap) * (overlaps < max_overlap)
                idxs = idxs[mask]
                return idxs.copy()
        else:
            idxs = np.zeros((len(self.poses) - 1, 4), dtype=np.uint16)
            idxs[:, 2] = 1
            idxs[:, 3] = np.array([int(fn[-9:-4])
                                  for fn in self.poses.keys() if 'seq0' not in fn], dtype=np.uint16)
            return idxs[::sample_factor]

    def get_pair_path(self, pair):
        seqA, imgA, seqB, imgB = pair
        return (f'seq{seqA}/frame_{imgA:05}.jpg', f'seq{seqB}/frame_{imgB:05}.jpg')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # image paths (relative to scene_root)
        im1_path, im2_path = self.get_pair_path(self.pairs[index])

        # load color images
        image0_reg = read_color_image(self.scene_root / im1_path,
                                  self.resize, augment_fn=self.transforms)
        image1_reg = read_color_image(self.scene_root / im2_path,
                                  self.resize, augment_fn=self.transforms)

        if self.use_loftr_preds or self.use_superglue_preds:
            rot0, rot1 = 0, 0
            _, image0, _ = read_image(
                self.scene_root / im1_path, 'cpu', self.resize_matcher, rot0, resize_float=True)

            _, image1, _ = read_image(
                self.scene_root / im2_path, 'cpu', self.resize_matcher, rot1, resize_float=True)
            
            if self.use_loftr_preds and image0.size(2) % 8 != 0 or image0.size(1) % 8 != 0:
                pad_bottom = image0.size(2) % 8
                pad_right = image0.size(3) % 8
                pad_fn = torch.nn.ConstantPad2d((0, pad_right, 0, pad_bottom), 0)
                image0 = pad_fn(image0)
                image1 = pad_fn(image1)

            image0, image1 = image0[0], image1[0]

            K0 = self.K_loftr[im1_path].copy()
            K1 = self.K_loftr[im2_path].copy()
        else:
            image0, image1 = torch.tensor([]), torch.tensor([])
            K0 = self.K[im1_path].copy()
            K1 = self.K[im2_path].copy()

        # load depth maps
        if self.estimated_depth is not None:
            dim1_path = str(self.scene_root / im1_path).replace('.jpg',
                                                                f'.{self.estimated_depth}.png')
            dim2_path = str(self.scene_root / im2_path).replace('.jpg',
                                                                f'.{self.estimated_depth}.png')
            depth1 = read_depth_image(dim1_path)
            depth2 = read_depth_image(dim2_path)
        else:
            depth1 = depth2 = torch.tensor([]), torch.tensor([])

        # get absolute pose of im0 and im1
        # quaternion and translation vector that transforms World-to-Cam
        q1, t1 = self.poses[im1_path]
        # quaternion and translation vector that transforms World-to-Cam
        q2, t2 = self.poses[im2_path]
        c1 = rotate_vector(-t1, qinverse(q1))  # center of camera 1 in world coordinates)
        c2 = rotate_vector(-t2, qinverse(q2))  # center of camera 2 in world coordinates)

        # get 4 x 4 relative pose transformation matrix (from im1 to im2)
        # for test/val set, q1,t1 is the identity pose, so the relative pose matches the absolute pose
        q12 = qmult(q2, qinverse(q1))
        t12 = t2 - rotate_vector(t1, q12)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quat2mat(q12)
        T[:3, -1] = t12
        T = torch.from_numpy(T)

        data = {
            'image0_reg': image0_reg,  # (3, h, w)
            'depth0': depth1,  # (h, w)
            'image1_reg': image1_reg,
            'depth1': depth2,
            'image0': image0, # for matcher
            'image1': image1, # for matcher
            'K_color0': K0, # matcher if in use else reg
            'K_color1': K1, # matcher if in use else reg
            'T_0to1': T,  # (4, 4)  # relative pose
            'abs_q_0': q1,
            'abs_c_0': c1,
            'abs_q_1': q2,
            'abs_c_1': c2,
            'dataset_name': 'Mapfree',
            'scene_id': self.scene_root.stem,
            'scene_root': str(self.scene_root),
            'pair_id': index*self.sample_factor,
            'pair_names': (im1_path, im2_path),
            'sim': 0.  # needed for 7Scenes eval compatibility
        }

        return data


class MapFreeDataset(data.ConcatDataset):
    def __init__(self, cfg, mode, transforms=None, use_loftr_preds=False, use_superglue_preds=False):
        assert mode in ['train', 'val', 'test'], 'Invalid dataset mode'

        scenes = cfg.DATASET.SCENES
        data_root = Path(cfg.DATASET.DATA_ROOT) / mode
        resize = (cfg.DATASET.WIDTH, cfg.DATASET.HEIGHT)
        # If None, no depth. Otherwise, loads depth map with name `frame_00000.suffix.png` where suffix is estimated_depth
        estimated_depth = cfg.DATASET.ESTIMATED_DEPTH
        overlap_limits = (cfg.DATASET.MIN_OVERLAP_SCORE, cfg.DATASET.MAX_OVERLAP_SCORE)
        sample_factor = {'train': 1, 'val': 5, 'test': 5}[mode]

        if scenes is None:
            # Locate all scenes of the current dataset
            scenes = [s.name for s in data_root.iterdir() if s.is_dir()]

        # Init dataset objects for each scene
        data_srcs = [
            MapFreeScene(
                data_root / scene, resize, sample_factor, overlap_limits, transforms,
                estimated_depth, use_loftr_preds=use_loftr_preds, use_superglue_preds=use_superglue_preds) for scene in scenes]
        super().__init__(data_srcs)
