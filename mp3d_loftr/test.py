import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger

from src.config.default import get_cfg_defaults
from src.utils.profiler import build_profiler

from src.lightning.data import Mp3dDataModule, Mp3dLightDataModule, InteriornetStreetlearnDataModule
from src.lightning.lightning_loftr import PL_LoFTR
from pytorch_lightning.plugins import DDPPlugin

import sys
import os
sys.path.append('third_party/prior_ransac')

def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
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
        '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    parser.add_argument(
        '--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--corr_dropout', type=float, default=0.0, help='prob of dropping out correspondences for transformer')
    parser.add_argument(
        '--num_workers', type=int, default=2)
    parser.add_argument(
        '--thr', type=float, default=None, help='modify the coarse-level matching threshold.')
    parser.add_argument(
        "--num_coarse_loftr_layers", default=4, type=int, help="num coarse loftr layers")
    parser.add_argument(
        '--eval_split', type=str, default="test", help='split on which to evaluate')
    parser.add_argument(
        '--numpy_data_path', type=str,  help='numpy datapath for streetlearn/interiornet')
    parser.add_argument(
        '--dset_name', type=str, default='', help='dset name for loading imgs (streetlearn/interiornet)')
    parser.add_argument(
        '--regress_use_num_corres', action='store_true', help='if true, encode num corres to use during LoFTR prediction')
    parser.add_argument(
        '--save_preds', default=None, type=str, help='if is not none gives file path to save coarse features & LoFTR predictions')
    parser.add_argument(
        '--save_corr', action='store_true', help='save correspondences')
    parser.add_argument(
        '--no_save_feats', action='store_true', help='if true, dont save feats')
    parser.add_argument(
        '--no_save_preds', action='store_true', help='if true, dont save preds')
    parser.add_argument(
        '--correspondence_tf_use_feats', action='store_true', help='if true, correspondence transformer uses features as input')
    parser.add_argument(
        '--ctf_cat_feats', action='store_true', help='if true, correspondence transformer uses features cat as input')
    parser.add_argument(
        '--use_pos_embedding', action='store_true', help='if true, emm pose regressor uses use_pos_embedding')
    parser.add_argument(
        "--load_loftr_feats", default=False, action="store_true", help="load loftr feats")
    parser.add_argument(
        '--correspondence_tf_use_global_avg_pooling', action='store_true', help='if true, use global avg pooling for corresondence transformer')
    parser.add_argument(
        '--correspondence_transformer_use_pos_encoding', action='store_true', help='if true, use positional encoding for corresondence transformer')
    parser.add_argument(
        '--save_corr_after_ransac', action='store_true', help='if true, save gt correspondences after ransac')
    parser.add_argument(
        "--max_correspondences", default=2000, type=int, help="max correspondences input to correspondence transformer")
    parser.add_argument(
        "--save_hard_corres", default=False, action="store_true", help="save hard correspondences only")
    parser.add_argument(
        "--after_ransac_path", default=None, type=str, help="if we train after ransac, path to load these")
    parser.add_argument(
        "--num_bands", default=10, type=int, help="num pos encoding bands")
    parser.add_argument(
        '--use_correspondence_transformer', action='store_true', help='if true, use transformer mapping correspondences to RT (no scale)')
    parser.add_argument(
        '--correspondences_use_fit_only', action='store_true', help='if true, only train on cases where there are >5 ground truth correspondences')
    parser.add_argument(
        "--predict_translation_scale", default=False, action="store_true", help="predict scale of translation")
    parser.add_argument(
        "--scale_8pt", default=False, action="store_true", help="if true, scale 8 point translation before gated")
    parser.add_argument(
        "--num_encoder_layers", default=6, type=int, help="num encoder layers in corr transformer")
    parser.add_argument(
        "--num_heads", default=8, type=int, help="num heads in corr transformer")
    parser.add_argument(
        "--fine_pred_steps", default=1, type=int, help="number of times to run fine pred corr transformer")
    parser.add_argument(
        "--outlier_pct", default=0, type=int, help="percentage of random correspondences that replace gt correspondences")
    parser.add_argument(
        "--noise_pix", default=0, type=int, help="stdev of noise (in pixels) added to gt correspondences")
    parser.add_argument(
        "--use_many_ransac_thr", action='store_true', help="use many sampson thresholds")
    parser.add_argument(
        "--missing_pct", default=0, type=int, help="percentage of missing correspondences that replace gt correspondences")
    parser.add_argument(
        "--regress_rt", default=False, action="store_true", help="regress RT (no scale) with transformer")
    parser.add_argument(
        "--no_save_numcorr", default=False, action="store_true", help="")
    parser.add_argument(
        "--regress_loftr_layers", default=0, type=int, help="regress RT (no scale) with LoFTR + EMM")    
    parser.add_argument(
        "--from_saved_corr", default=None, type=str, help="path from which to load correspondences, if doing so")
    parser.add_argument(
        "--feat_size", default=128, type=int, help="num features in corr transformer")
    parser.add_argument(
        '--strict_false', action='store_true', help='if true, load ckpt with strict=False')
    parser.add_argument(
        "--use_pred_depth", default=False, action="store_true", help="if true, use DPT predicted depth instead of ground truth")
    parser.add_argument(
        "--use_pred_corr", default=False, action="store_true", help="use predicted correspondences input to corr transformer")
    parser.add_argument(
        "--load_prior_ransac", default=None, type=str, help="")
    parser.add_argument(
        "--eval_fit_only", default=False, action="store_true", help="eval only examples where the model fits")   
    parser.add_argument(
        "--use_5050_weight", default=False, action="store_true", help="equally weighted")   
    parser.add_argument(
        "--use_old_slow_warmup", default=False, action="store_true", help="if true, use old slow warmup")    
    parser.add_argument(
        "--save_mlp_feats", default=False, action="store_true", help="if true, save MLP feats")
    parser.add_argument(
        "--use_1wt", default=False, action="store_true", help="if true, pred 1wt")
    parser.add_argument(
        "--use_2wt", default=False, action="store_true", help="if true, pred 2wt")
    parser.add_argument(
        "--use_simple_moe", default=False, action="store_true", help="if true, use simple mixture of experts")
    parser.add_argument(
        "--use_ransac_prior", default=False, action="store_true", help="if true, use ransac prior")
    parser.add_argument(
        '--save_gating_weights', action='store_true', default=False, help='if true, saves gating weights to disk.')
    parser.add_argument(
        "--load_predictions_path", type=str, default=None, help="path to load predictions from to initalize")
    parser.add_argument(
        "--solver", default='ransac', type=str, help="optimizer to solve E")
    parser.add_argument(
        "--from_saved_preds", default=None, type=str, help="path from which to load predictions / features, if doing so")
    parser.add_argument(
        "--ransac_thr", default=None, type=float, help="threshold for ransac / lms")
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

    # pass into config
    config.LOFTR.PREDICT_TRANSLATION_SCALE = args.predict_translation_scale
    config.LOFTR.REGRESS_RT = args.regress_rt
    config.LOFTR.REGRESS_LOFTR_LAYERS = args.regress_loftr_layers
    config.LOFTR.FROM_SAVED_PREDS = args.from_saved_preds
    config.LOFTR.SAVE_PREDS = args.save_preds # needed to avoid empty loftr... kinda messy implementation
    config.LOFTR.CORRESPONDENCE_TF_USE_GLOBAL_AVG_POOLING = args.correspondence_tf_use_global_avg_pooling
    config.LOFTR.CORRESPONDENCE_TRANSFORMER_USE_POS_ENCODING = args.correspondence_transformer_use_pos_encoding
    config.LOFTR.FEAT_SIZE = args.feat_size
    config.LOFTR.NUM_BANDS = args.num_bands
    config.LOFTR.NUM_ENCODER_LAYERS = args.num_encoder_layers
    config.LOFTR.NUM_HEADS = args.num_heads
    config.LOFTR.CORRESPONDENCE_TF_USE_FEATS = args.correspondence_tf_use_feats
    config.LOFTR.CTF_CAT_FEATS = args.ctf_cat_feats
    config.LOFTR.SOLVER = args.solver
    config.LOFTR.USE_MANY_RANSAC_THR = args.use_many_ransac_thr

    config.LOFTR.REGRESS.USE_POS_EMBEDDING = args.use_pos_embedding
    config.LOFTR.REGRESS.REGRESS_USE_NUM_CORRES = args.regress_use_num_corres
    config.LOFTR.FINE_PRED_STEPS = args.fine_pred_steps
    config.LOFTR.REGRESS.SAVE_MLP_FEATS = args.save_mlp_feats
    config.LOFTR.REGRESS.USE_SIMPLE_MOE = args.use_simple_moe
    config.LOFTR.REGRESS.USE_2WT = args.use_2wt
    config.LOFTR.REGRESS.USE_5050_WEIGHT = args.use_5050_weight
    config.LOFTR.REGRESS.USE_1WT = args.use_1wt
    config.LOFTR.REGRESS.SCALE_8PT = args.scale_8pt
    config.LOFTR.REGRESS.SAVE_GATING_WEIGHTS = args.save_gating_weights
    config.LOFTR.TRAINING = False

    config.SAVE_PREDS = args.save_preds
    config.SAVE_CORR = args.save_corr
    config.SAVE_CORR_AFTER_RANSAC = args.save_corr_after_ransac
    config.SAVE_HARD_CORRES = args.save_hard_corres
    config.EVAL_FIT_ONLY = args.eval_fit_only
    config.LOAD_PREDICTIONS_PATH = args.load_predictions_path
    config.EXP_NAME = args.exp_name
    config.USE_CORRESPONDENCE_TRANSFORMER = args.use_correspondence_transformer
    config.CORRESPONDENCES_USE_FIT_ONLY = args.correspondences_use_fit_only
    config.MAX_CORRESPONDENCES = args.max_correspondences
    config.OUTLIER_PCT = args.outlier_pct
    config.NOISE_PIX = args.noise_pix
    config.MISSING_PCT = args.missing_pct
    config.STRICT_FALSE = args.strict_false
    config.DSET_NAME = args.dset_name
    config.NO_SAVE_FEATS = args.no_save_feats
    config.NO_SAVE_PREDS = args.no_save_preds
    config.NO_SAVE_NUMCORR = args.no_save_numcorr
    config.EVAL_SPLIT = args.eval_split
    config.USE_PRED_CORR = args.use_pred_corr
    config.PL_VERSION = pl.__version__

    if args.num_coarse_loftr_layers < 4:
        config.LOFTR.COARSE.LAYER_NAMES = ['self', 'cross'] * args.num_coarse_loftr_layers

    if args.regress_loftr_layers > 1:
        config.LOFTR.REGRESS.LAYER_NAMES = ['self', 'cross'] * args.regress_loftr_layers

    if args.ransac_thr is not None:
        config.TRAINER.RANSAC_PIXEL_THR = args.ransac_thr

    # tune when testing
    if args.thr is not None:
        config.LOFTR.MATCH_COARSE.THR = args.thr

    loguru_logger.info(f"Args and config initialized!")

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir, split=args.eval_split)
    loguru_logger.info(f"LoFTR-lightning initialized!")

    # lightning data
    if config.USE_CORRESPONDENCE_TRANSFORMER:
        if config.DATASET.TRAINVAL_DATA_SOURCE == 'interiornet_streetlearn':
            raise NotImplementedError()
        else:
            data_module = Mp3dLightDataModule(args, config)
    else:
        if config.DATASET.TRAINVAL_DATA_SOURCE == 'interiornet_streetlearn':
            data_module = InteriornetStreetlearnDataModule(args, config)
            loguru_logger.info(f"InteriornetStreetlearnDataModule DataModule initialized!")
        else:
            data_module = Mp3dDataModule(args, config)
        loguru_logger.info(f"LoFTR Mp3d DataModule initialized!")

    

    # lightning trainer
    trainer = pl.Trainer.from_argparse_args(args, replace_sampler_ddp=False, logger=False,
                                            plugins=DDPPlugin(find_unused_parameters=False,))

    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)
