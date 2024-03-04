import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import Mp3dDataModule, Mp3dLightDataModule
from src.lightning.lightning_loftr import PL_LoFTR

loguru_logger = get_rank_zero_only_logger(loguru_logger)
import warnings 
warnings.filterwarnings('error', category=RuntimeWarning)

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
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    parser.add_argument(
        '--find_unused_parameters', action='store_true', help='if true, debug ddp')
    parser.add_argument(
        "--rt_weight_rot", default=0.0, type=float, help="weight for the rotation loss")
    parser.add_argument(
        "--rt_weight_tr", default=0.0, type=float, help="weight for the translation loss")
    parser.add_argument(
        "--scale_weight", default=0.0, type=float, help="weight for the scale loss")
    parser.add_argument(
        "--fine_weight", default=1.0, type=float, help="weight for fine correspondences loss")
    parser.add_argument(
        "--coarse_weight", default=1.0, type=float, help="weight for coarse correspondences loss")
    parser.add_argument(
        "--predict_translation_scale", default=False, action="store_true", help="predict scale of translation")
    parser.add_argument(
        "--use_40pct_dset", default=False, action="store_true", help="use tiny dset")
    parser.add_argument(
        '--ckpt_every_n_epochs', type=int, default=1, help='how frequently to checkpoint')
    parser.add_argument(
        '--num_warmup_steps', type=int, default=None, help='num warmup steps')
    parser.add_argument(
        '--lr', type=float, default=None, help='learning rate')
    parser.add_argument(
        "--use_one_cycle_lr", default=False, action="store_true", help="one cycle lr")
    parser.add_argument(
        "--use_pred_corr", default=False, action="store_true", help="use predicted correspondences input to corr transformer")
    parser.add_argument(
        '--correspondences_use_fit_only', action='store_true', help='if true, only train on cases where there are >5 ground truth correspondences')
    parser.add_argument(
        "--regress_rt", default=False, action="store_true", help="regress RT (no scale) with transformer")    
    parser.add_argument(
        "--regress_loftr_layers", default=0, type=int, help="regress RT (no scale) with LoFTR + EMM")    
    parser.add_argument(
        "--use_old_slow_warmup", default=False, action="store_true", help="if true, use old slow warmup")
    parser.add_argument(
        "--thr", default=0.2, type=float, help="threshold for conf_matrix in coarse correspondences")
    parser.add_argument(
        "--max_scale_loss", default=999, type=float, help="threshold for scale loss")
    parser.add_argument(
        '--use_pos_embedding', action='store_true', help='if true, emm pose regressor uses use_pos_embedding')
    parser.add_argument(
        "--use_many_ransac_thr", action='store_true', help="use many sampson thresholds")
    parser.add_argument(
        '--regress_use_num_corres', action='store_true', help='if true, encode num corres to use during LoFTR prediction')
    parser.add_argument(
        '--no_save_feats', action='store_true', help='if true, dont save feats')
    parser.add_argument(
        '--no_save_preds', action='store_true', help='if true, dont save preds')
    parser.add_argument(
        '--correspondence_tf_use_global_avg_pooling', action='store_true', help='if true, use global avg pooling for corresondence transformer')
    parser.add_argument(
        '--use_correspondence_transformer', action='store_true', help='if true, use transformer mapping correspondences to RT (no scale)')
    parser.add_argument(
        '--strict_false', action='store_true', help='if true, load ckpt with strict=False')
    parser.add_argument(
        "--use_large_dset", default=False, action="store_true", help="if true, use bigger train set")
    parser.add_argument(
        "--num_coarse_loftr_layers", default=4, type=int, help="num coarse loftr layers")
    parser.add_argument(
        "--num_bands", default=10, type=int, help="num pos encoding bands")
    parser.add_argument(
        "--feat_size", default=128, type=int, help="num features in corr transformer")
    parser.add_argument(
        "--num_encoder_layers", default=6, type=int, help="num encoder layers in corr transformer")
    parser.add_argument(
        "--num_heads", default=8, type=int, help="num heads in corr transformer")
    parser.add_argument(
        "--outlier_pct", default=0, type=int, help="percentage of random correspondences that replace gt correspondences")
    parser.add_argument(
        "--missing_pct", default=0, type=int, help="percentage of missing correspondences that replace gt correspondences")
    parser.add_argument(
        "--noise_pix", default=0, type=int, help="stdev of noise (in pixels) added to gt correspondences")
    parser.add_argument(
        '--corr_dropout', type=float, default=0.0, help='prob of dropping out correspondences for transformer')
    parser.add_argument(
        '--correspondence_tf_use_feats', action='store_true', help='if true, correspondence transformer uses features as input')
    parser.add_argument(
        '--ctf_cat_feats', action='store_true', help='if true, correspondence transformer uses features cat as input')
    parser.add_argument(
        '--correspondence_transformer_use_pos_encoding', action='store_true', help='if true, use positional encoding for corresondence transformer')
    parser.add_argument(
        "--max_correspondences", default=2000, type=int, help="max correspondences input to correspondence transformer")
    parser.add_argument(
        "--use_ransac_prior", default=False, action="store_true", help="if true, use ransac prior")
    parser.add_argument(
        "--use_1wt", default=False, action="store_true", help="if true, pred 1wt")
    parser.add_argument(
        "--use_2wt", default=False, action="store_true", help="if true, pred 2wt")
    parser.add_argument(
        "--use_simple_moe", default=False, action="store_true", help="if true, use simple mixture of experts")
    parser.add_argument(
        "--no_save_numcorr", default=False, action="store_true", help="")
    parser.add_argument(
        "--solver", default='ransac', type=str, help="optimizer to solve E")
    parser.add_argument(
        "--loss_fine_type", default='l2_with_std', type=str, help="loss fine type")
    parser.add_argument(
        "--use_pred_depth", default=False, action="store_true", help="if true, use DPT predicted depth instead of ground truth")
    parser.add_argument(
        "--use_l1_rt_loss", default=False, action="store_true", help="if true, use l1 rt loss")
    parser.add_argument(
        "--fine_pred_steps", default=1, type=int, help="number of times to run fine pred corr transformer")
    parser.add_argument(
        "--scale_8pt", default=False, action="store_true", help="if true, scale 8 point translation before gated")
    parser.add_argument(
        "--from_saved_preds", default=None, type=str, help="path from which to load predictions / features, if doing so")
    parser.add_argument(
        "--from_saved_corr", default=None, type=str, help="path from which to load correspondences, if doing so")
    parser.add_argument(
        "--after_ransac_path", default=None, type=str, help="if we train after ransac, path to load these")
    parser.add_argument(
        "--load_loftr_feats", default=False, action="store_true", help="load loftr feats")
    parser.add_argument(
        '--save_preds', default=None, type=str, help='not relevant to train')
    parser.add_argument(
        "--use_sparse_spvs", default=False, action="store_true", help="if true, use sparse spvs option for supervision (no negatives)")
    
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation
    
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    
    if args.num_warmup_steps is not None:
        print("overwrote num warmup steps, using:",args.num_warmup_steps)
        config.TRAINER.WARMUP_STEP = args.num_warmup_steps
    else:
        config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP)
    if args.lr is not None:
        print("overwrote lr, using:",args.lr)
        config.TRAINER.TRUE_LR = args.lr
    else:
        config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
        print("using default learning rate", config.TRAINER.TRUE_LR)
    
    config.LOFTR.PREDICT_TRANSLATION_SCALE = args.predict_translation_scale
    config.LOFTR.REGRESS_RT = args.regress_rt
    config.LOFTR.REGRESS_LOFTR_LAYERS = args.regress_loftr_layers
    config.LOFTR.MATCH_COARSE.SPARSE_SPVS = args.use_sparse_spvs
    config.LOFTR.MATCH_COARSE.THR = args.thr
    config.LOFTR.FROM_SAVED_PREDS = args.from_saved_preds
    config.LOFTR.CORRESPONDENCE_TF_USE_GLOBAL_AVG_POOLING = args.correspondence_tf_use_global_avg_pooling
    config.LOFTR.CORRESPONDENCE_TRANSFORMER_USE_POS_ENCODING = args.correspondence_transformer_use_pos_encoding
    config.LOFTR.NUM_BANDS = args.num_bands
    config.LOFTR.FEAT_SIZE = args.feat_size
    config.LOFTR.NUM_ENCODER_LAYERS = args.num_encoder_layers
    config.LOFTR.NUM_HEADS = args.num_heads
    config.LOFTR.FINE_PRED_STEPS = args.fine_pred_steps
    config.LOFTR.CORRESPONDENCE_TF_USE_FEATS = args.correspondence_tf_use_feats
    config.LOFTR.CTF_CAT_FEATS = args.ctf_cat_feats
    args.load_prior_ransac = None
    config.LOFTR.SOLVER = args.solver
    config.LOFTR.USE_MANY_RANSAC_THR = args.use_many_ransac_thr
    config.LOFTR.TRAINING = True

    config.LOFTR.REGRESS.USE_POS_EMBEDDING = args.use_pos_embedding
    config.LOFTR.REGRESS.REGRESS_USE_NUM_CORRES = args.regress_use_num_corres
    config.LOFTR.REGRESS.USE_SIMPLE_MOE = args.use_simple_moe
    config.LOFTR.REGRESS.SAVE_MLP_FEATS = False
    config.LOFTR.REGRESS.USE_2WT = args.use_2wt
    config.LOFTR.REGRESS.USE_1WT = args.use_1wt
    config.LOFTR.REGRESS.USE_5050_WEIGHT = False
    config.LOFTR.REGRESS.SCALE_8PT = args.scale_8pt
    config.LOFTR.REGRESS.SAVE_GATING_WEIGHTS = False

    config.LOFTR.LOSS.RT_WEIGHT_ROT = args.rt_weight_rot
    config.LOFTR.LOSS.RT_WEIGHT_TR = args.rt_weight_tr
    config.LOFTR.LOSS.SCALE_WEIGHT = args.scale_weight
    config.LOFTR.LOSS.COARSE_WEIGHT = args.coarse_weight
    config.LOFTR.LOSS.FINE_WEIGHT = args.fine_weight
    config.LOFTR.LOSS.MAX_SCALE_LOSS = args.max_scale_loss
    config.LOFTR.LOSS.FINE_TYPE = args.loss_fine_type
    config.LOFTR.LOSS.USE_L1_RT_LOSS = args.use_l1_rt_loss

    config.USE_CORRESPONDENCE_TRANSFORMER = args.use_correspondence_transformer
    config.CORRESPONDENCES_USE_FIT_ONLY = args.correspondences_use_fit_only
    config.MAX_CORRESPONDENCES = args.max_correspondences
    config.OUTLIER_PCT = args.outlier_pct
    config.NOISE_PIX = args.noise_pix
    config.MISSING_PCT = args.missing_pct
    config.SAVE_PREDS = None
    config.STRICT_FALSE = args.strict_false
    config.NO_SAVE_FEATS = args.no_save_feats
    config.NO_SAVE_PREDS = args.no_save_preds
    config.NO_SAVE_NUMCORR = args.no_save_numcorr
    config.USE_PRED_CORR = args.use_pred_corr
    config.PL_VERSION = pl.__version__

    if args.solver == 'prior_ransac':
        import os
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

    if args.num_coarse_loftr_layers < 4:
        config.LOFTR.COARSE.LAYER_NAMES = ['self', 'cross'] * args.num_coarse_loftr_layers

    if args.regress_loftr_layers > 1:
        config.LOFTR.REGRESS.LAYER_NAMES = ['self', 'cross'] * args.regress_loftr_layers

    if args.use_one_cycle_lr:
        config.TRAINER.SCHEDULER = 'OneCycleLR'
        if args.use_large_dset:
            config.TRAINER.STEPS = args.max_epochs * 8740
        elif args.use_40pct_dset:
            config.TRAINER.STEPS = args.max_epochs * 1700
        else:
            config.TRAINER.STEPS = args.max_epochs * 3665
        print(f"using {config.TRAINER.STEPS} steps")
        config.TRAINER.PCT_WARMUP = config.TRAINER.WARMUP_STEP / config.TRAINER.STEPS
        config.TRAINER.WARMUP_TYPE = 'constant'

    if args.use_old_slow_warmup:
        config.TRAINER.WARMUP_STEP /= _scaling

    print(f"using {config.TRAINER.WARMUP_STEP} warmup steps")

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"LoFTR LightningModule initialized!")

    # lightning data
    if config.USE_CORRESPONDENCE_TRANSFORMER:
        data_module = Mp3dLightDataModule(args, config)
    else:
        data_module = Mp3dDataModule(args, config)
        loguru_logger.info(f"LoFTR Mp3d DataModule initialized!")
    
    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    
    print("about to setup model")

    

    if pl.__version__ == '1.6.0':
        ckpt_callback = ModelCheckpoint(monitor='rot_mean_err', verbose=True, save_top_k=5, mode='min',
                                        save_last=True,
                                        dirpath=str(ckpt_dir),
                                        filename='{epoch}-{rot_mean_err:.2f}-{rot_median_err:.2f}-{tr_abs_mean_err:.2f}')
    else:
        ckpt_callback = ModelCheckpoint(monitor='rot_mean_err', verbose=True, save_top_k=5, mode='min',
                                        save_last=True,
                                        dirpath=str(ckpt_dir),
                                        filename='{epoch}-{rot_mean_err:.2f}-{rot_median_err:.2f}-{tr_abs_mean_err:.2f}',
                                        every_n_val_epochs=args.ckpt_every_n_epochs)

    print("about to setup learning rate monitor")

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    
    print("about to initialize training")

    if pl.__version__ == '1.6.0':
        # Lightning Trainer
        trainer = pl.Trainer.from_argparse_args(
            args,
            gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
            callbacks=callbacks,
            logger=logger,
            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
            replace_sampler_ddp=False,  # use custom sampler
            profiler=profiler)        
    else:
        # Lightning Trainer
        trainer = pl.Trainer.from_argparse_args(
            args,
            plugins=DDPPlugin(find_unused_parameters=args.find_unused_parameters,
                            num_nodes=args.num_nodes,
                            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
            gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
            callbacks=callbacks,
            logger=logger,
            sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
            replace_sampler_ddp=False,  # use custom sampler
            reload_dataloaders_every_epoch=False,  # avoid repeated samples!
            weights_summary='full',
            profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()
