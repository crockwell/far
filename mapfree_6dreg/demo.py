import argparse
from pathlib import Path
import torch
import numpy as np
from config.default import cfg
import torch.nn.functional as F
import cv2
import sys

from lib.models.builder import build_model
from lib.datasets.utils import read_color_image, correct_intrinsic_scale

sys.path.append("third_party/LoFTR")
sys.path.append("etc/feature_matching_baselines")
sys.path.append("third_party/prior_ransac")

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new

def read_image(path, resize):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    image = cv2.resize(image.astype('float32'), (w_new, h_new))
    inp = torch.from_numpy(image/255.).float()[None, None]
    return inp, w, h

def eval(args):
    # Load configs
    cfg.set_new_allowed(True)
    cfg.merge_from_file('config/mapfree.yaml')
    cfg.merge_from_file(args.config)

    args.use_loftr_preds = True
    args.use_prior = True
    args.use_vanilla_transformer = True
    args.d_model = 256
    args.use_superglue_preds = False
    args.one_cycle_lr = False
    args.max_steps = False
    args.use_one_inlier_int = False
    resize=(270,360)
    resize_matcher=(540,720)
    K_color0 = torch.tensor([
        [float(args.k0[0]), 0, float(args.k0[2])],
        [0, float(args.k0[1]), float(args.k0[3])],
        [0, 0, 1]
    ]).unsqueeze(0)
    K_color1 = torch.tensor([
        [float(args.k1[0]), 0, float(args.k1[2])],
        [0, float(args.k1[1]), float(args.k1[3])],
        [0, 0, 1]
    ]).unsqueeze(0)

    # Create model
    model = build_model(cfg, args.checkpoint, use_loftr_preds=args.use_loftr_preds, use_superglue_preds=args.use_superglue_preds, args=args) 

    im1_path = Path(args.img_path0)
    im2_path = Path(args.img_path1)
    image0_reg = read_color_image(im1_path, resize).unsqueeze(0)
    image1_reg = read_color_image(im2_path, resize).unsqueeze(0)
    image0, w0, h0 = read_image(im1_path, resize_matcher)
    image1, w1, h1 = read_image(im2_path, resize_matcher)
    
    # scale intrinsics appropriately
    K_color0 = correct_intrinsic_scale(K_color0.numpy(), resize_matcher[0] / w0, resize_matcher[1] / h0)
    K_color1 = correct_intrinsic_scale(K_color1.numpy(), resize_matcher[0] / w1, resize_matcher[1] / h1)

    if image0.size(2) % 8 != 0 or image0.size(1) % 8 != 0:
        pad_bottom = image0.size(2) % 8
        pad_right = image0.size(3) % 8
        pad_fn = torch.nn.ConstantPad2d((0, pad_right, 0, pad_bottom), 0)
        image0 = pad_fn(image0)
        image1 = pad_fn(image1)

    data = {
        'image0_reg': image0_reg.cuda(),  # (3, h, w)
        'image1_reg': image1_reg.cuda(),
        'image0': image0.cuda(), # for matcher
        'image1': image1.cuda(), # for matcher
        'K_color0': torch.from_numpy(K_color0).cuda(),
        'K_color1': torch.from_numpy(K_color1).cuda(),
    }

    with torch.no_grad():
        R, t = model(data)
        R = rotation_6d_to_matrix(R) # from 6D to matrix
    
    R = R.detach().cpu().numpy()
    t = t.detach().cpu().numpy()

    print(np.round(np.concatenate([R[0],np.swapaxes(t, 0, 1)], axis=1), 4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--checkpoint', help='path to model checkpoint', default='')
    parser.add_argument('--img_path0', type=str, help='img path 0')
    parser.add_argument('--img_path1', type=str, help='img path 1')
    parser.add_argument('--k0', type=str, nargs='+')
    parser.add_argument('--k1', type=str, nargs='+')
    args = parser.parse_args()
    eval(args)
