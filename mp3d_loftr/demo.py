import pytorch_lightning as pl
import argparse

from src.config.default import get_cfg_defaults

from src.lightning.lightning_loftr import PL_LoFTR
import sys
import torch
import cv2
import numpy as np
sys.path.append('third_party/prior_ransac')

def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, default="", help='path to the checkpoint')
    parser.add_argument(
        '--exp_name', type=str, default="", help='exp_name, if not present use ckpt')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=2)
    parser.add_argument(
        '--fx', type=float, default=517.97, help='focal length x')
    parser.add_argument(
        '--fy', type=float, default=517.97, help='focal length y')
    parser.add_argument(
        '--cx', type=float, default=320, help='principal point x')
    parser.add_argument(
        '--cy', type=float, default=240, help='principal point y')
    parser.add_argument(
        '--h', type=int, default=640, help='img height x')
    parser.add_argument(
        '--w', type=int, default=480, help='img width y')
    parser.add_argument(
        '--img_path0', type=str, default="test", help='img path 0')
    parser.add_argument(
        '--img_path1', type=str, default="test", help='img path 1')
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # not relevant
    config.LOFTR.FEAT_SIZE = 128
    config.LOFTR.NUM_ENCODER_LAYERS = 6
    config.LOFTR.NUM_BANDS = 10
    config.LOFTR.CORRESPONDENCE_TRANSFORMER_USE_POS_ENCODING = False
    config.LOFTR.CORRESPONDENCE_TF_USE_GLOBAL_AVG_POOLING = False
    config.LOFTR.CORRESPONDENCE_TF_USE_FEATS = False
    config.LOFTR.CTF_CAT_FEATS = False
    config.LOFTR.NUM_HEADS = 8
    config.USE_PRED_CORR = False
    config.STRICT_FALSE = False
    config.EVAL_FIT_ONLY = False


    # relevant
    config.LOFTR.PREDICT_TRANSLATION_SCALE = False
    config.LOFTR.REGRESS_RT = True
    config.LOFTR.REGRESS_LOFTR_LAYERS = 1
    config.LOFTR.REGRESS.USE_POS_EMBEDDING = True
    config.LOFTR.REGRESS.REGRESS_USE_NUM_CORRES = True
    config.LOFTR.COARSE.LAYER_NAMES = ['self', 'cross'] * 3

    # pass into config
    config.LOFTR.FROM_SAVED_PREDS = None
    config.LOFTR.SOLVER = "prior_ransac"
    config.LOFTR.USE_MANY_RANSAC_THR = True

    config.LOFTR.FINE_PRED_STEPS = 2
    config.LOFTR.REGRESS.SAVE_MLP_FEATS = False
    config.LOFTR.REGRESS.USE_SIMPLE_MOE = True
    config.LOFTR.REGRESS.USE_2WT = True
    config.LOFTR.REGRESS.USE_5050_WEIGHT = False
    config.LOFTR.REGRESS.USE_1WT = False
    config.LOFTR.REGRESS.SCALE_8PT = True
    config.LOFTR.REGRESS.SAVE_GATING_WEIGHTS = False
    config.LOFTR.TRAINING = False

    config.LOAD_PREDICTIONS_PATH = None
    config.EXP_NAME = args.exp_name
    config.USE_CORRESPONDENCE_TRANSFORMER = False
    config.CORRESPONDENCES_USE_FIT_ONLY = False
    config.EVAL_SPLIT = "test"
    config.PL_VERSION = pl.__version__

    # load data
    image0 = cv2.imread(args.img_path0, cv2.IMREAD_GRAYSCALE)
    image0 = cv2.resize(image0, (args.w, args.h))
    image0 = torch.from_numpy(image0).float()[None].unsqueeze(0).cuda() / 255
    image1 = cv2.imread(args.img_path1, cv2.IMREAD_GRAYSCALE)
    image1 = cv2.resize(image1, (args.w, args.h))
    image1 = torch.from_numpy(image1).float()[None].unsqueeze(0).cuda() / 255

    def get_intrinsics(fx, fy, cx, cy):
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        K = np.array(K)
        K = torch.from_numpy(K.astype(np.double)).unsqueeze(0).cuda()
        return K, K
        
    K_0, K_1 = get_intrinsics(float(args.fx), float(args.fy), float(args.cx), float(args.cy))

    # unused
    depth0 = depth1 = torch.tensor([]).unsqueeze(0).cuda()
    T_0to1 = T_1to0 = torch.tensor([]).unsqueeze(0).cuda()
    scene_name = torch.tensor([]).unsqueeze(0).cuda()
    loaded_preds = torch.tensor([]).unsqueeze(0).cuda()
    lightweight_numcorr = torch.tensor([0]).unsqueeze(0).cuda()

    batch = {
        'image0': image0,   # (1, h, w)
        'image1': image1,
        'K0': K_0,  # (3, 3)
        'K1': K_1,
        # below is unused
        'depth0': depth0,   # (h, w)
        'depth1': depth1,
        'T_0to1': T_0to1,   # (4, 4)
        'T_1to0': T_1to0,
        'dataset_name': ['mp3d'],
        'scene_id': scene_name,
        'pair_id': 0,
        'pair_names': (args.img_path0, args.img_path1),
        'loaded_predictions': loaded_preds,
        'lightweight_numcorr': lightweight_numcorr,
    }

    # lightning module
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, split="test").eval().cuda()
    
    # forward pass
    batch = model.test_step(batch, batch_idx=0, skip_eval=True)

    # output
    print("predicted pose is:\n", np.round(batch['loftr_rt'].cpu().numpy(),4))
