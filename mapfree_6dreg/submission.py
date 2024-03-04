import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import torch
import numpy as np
from tqdm import tqdm

from config.default import cfg
from lib.datasets.datamodules import DataModule
from lib.models.builder import build_model
from lib.utils.data import data_to_model_device
from transforms3d.quaternions import mat2quat
import torch.nn.functional as F

import sys
sys.path.append("etc/feature_matching_baselines")
sys.path.append("third_party/prior_ransac")

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

@dataclass
class Pose:
    image_name: str
    q: np.ndarray
    t: np.ndarray
    inliers: float

    def __str__(self) -> str:
        formatter = {'float': lambda v: f'{v:.6f}'}
        max_line_width = 1000
        q_str = np.array2string(self.q, formatter=formatter, max_line_width=max_line_width)[1:-1]
        t_str = np.array2string(self.t, formatter=formatter, max_line_width=max_line_width)[1:-1]
        return f'{self.image_name} {q_str} {t_str} {self.inliers}'

def predict(loader, model, use_matcher_preds=False, rot_loss=None):
    results_dict = defaultdict(list)

    for data in tqdm(loader):
        # run inference
        data = data_to_model_device(data, model)
        with torch.no_grad():
            R, t = model(data)
        if use_matcher_preds:
            R = rotation_6d_to_matrix(R) # from 6D to matrix
        
        R = R.detach().cpu().numpy()
        t = t.reshape(-1).detach().cpu().numpy()
        inliers = data['inliers']
        scene = data['scene_id'][0]
        query_img = data['pair_names'][1][0]

        # ignore frames without poses (e.g. not enough feature matches)
        if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
            continue

        if torch.is_tensor(inliers):
            inliers = inliers[0,0].cpu().item()

        # populate results_dict
        estimated_pose = Pose(image_name=query_img,
                              q=mat2quat(R).reshape(-1),
                              t=t.reshape(-1),
                              inliers=inliers)
        results_dict[scene].append(estimated_pose)

    return results_dict


def save_submission(results_dict: dict, output_path: Path):
    with ZipFile(output_path, 'w') as zip:
        for scene, poses in results_dict.items():
            poses_str = '\n'.join((str(pose) for pose in poses))
            zip.writestr(f'pose_{scene}.txt', poses_str.encode('utf-8'))

def eval(args):
    # Load configs
    cfg.set_new_allowed(True)
    cfg.merge_from_file('config/mapfree.yaml')
    cfg.merge_from_file(args.config)

    # Create dataloader
    if args.split == 'test':
        dataloader = DataModule(cfg, args.use_loftr_preds, use_superglue_preds=args.use_superglue_preds).test_dataloader()
    elif args.split == 'val' or args.split == 'train':
        cfg.TRAINING.BATCH_SIZE = 1
        cfg.TRAINING.NUM_WORKERS = 1
        dataloader = DataModule(cfg, args.use_loftr_preds, use_superglue_preds=args.use_superglue_preds).val_dataloader()
    else:
        raise NotImplemented(f'Invalid split: {args.split}')

    # Create model
    model = build_model(cfg, args.checkpoint, use_loftr_preds=args.use_loftr_preds, use_superglue_preds=args.use_superglue_preds, args=args) 

    # Get predictions from model
    results_dict = predict(dataloader, model, use_matcher_preds=args.use_loftr_preds or args.use_superglue_preds, rot_loss=cfg.TRAINING.ROT_LOSS)

    # Save predictions to txt per scene within zip
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_submission(results_dict, args.output_root / 'submission.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument(
        '--checkpoint', help='path to model checkpoint (models with learned parameters)',
        default='')
    parser.add_argument('--output_root', '-o', type=Path, default=Path('results/'))
    parser.add_argument(
        '--split', choices=('val', 'test', 'train'),
        default='test',
        help='Dataset split to use for evaluation. Choose from test or val. Default: test')
    parser.add_argument('--use_loftr_preds', action='store_true',)
    parser.add_argument('--use_superglue_preds', action='store_true',)
    parser.add_argument('--use_vanilla_transformer', action='store_true',)
    parser.add_argument("--d_model", default=32, type=int)
    parser.add_argument("--max_steps", default=200_000, type=int)
    parser.add_argument('--use_prior', action='store_true',)

    args = parser.parse_args()
    eval(args)
