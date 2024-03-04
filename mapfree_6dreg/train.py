import argparse
import os
# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "1"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa: E402

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from config.default import cfg
from lib.datasets.datamodules import DataModule
from lib.models.regression.model import RegressionModel

import sys
sys.path.append("etc/feature_matching_baselines")
sys.path.append("third_party/prior_ransac")

import torch
torch.autograd.set_detect_anomaly(True)

def main(args):
    cfg.set_new_allowed(True)

    cfg.merge_from_file(args.dataset_config)
    cfg.merge_from_file(args.config)

    datamodule = DataModule(cfg, args.use_loftr_preds, args.use_superglue_preds) 
    model = RegressionModel(cfg, args.use_loftr_preds, args.use_superglue_preds, ckpt_path=args.resume, 
                            not_strict=args.not_strict, use_vanilla_transformer=args.use_vanilla_transformer,
                            d_model=args.d_model, max_steps=args.max_steps, use_prior=args.use_prior)

    logger = TensorBoardLogger(save_dir='weights', name=args.experiment)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True,
        save_top_k=5,
        verbose=True,
        monitor='val_loss/loss',
        mode='min'
    )

    lr_monitoring_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    if args.gpus > 1:
        trainer = pl.Trainer(gpus=args.gpus,
                    log_every_n_steps=cfg.TRAINING.LOG_INTERVAL,
                    val_check_interval=cfg.TRAINING.VAL_INTERVAL,
                    limit_val_batches=cfg.TRAINING.VAL_BATCHES,
                    max_epochs=cfg.TRAINING.EPOCHS,
                    logger=logger,
                    callbacks=[checkpoint_callback, lr_monitoring_callback],
                    num_sanity_val_steps=10,
                    gradient_clip_val=cfg.TRAINING.GRAD_CLIP,
                    accelerator='ddp', 
                    replace_sampler_ddp=False,
                    sync_batchnorm=args.gpus > 1,
                    plugins=DDPPlugin(find_unused_parameters=False),
                    track_grad_norm=-1)
    else:
        trainer = pl.Trainer(gpus=args.gpus,
                            log_every_n_steps=cfg.TRAINING.LOG_INTERVAL,
                            val_check_interval=cfg.TRAINING.VAL_INTERVAL,
                            limit_val_batches=cfg.TRAINING.VAL_BATCHES,
                            max_epochs=cfg.TRAINING.EPOCHS,
                            logger=logger,
                            callbacks=[checkpoint_callback, lr_monitoring_callback],
                            num_sanity_val_steps=10,
                            gradient_clip_val=cfg.TRAINING.GRAD_CLIP,
                            track_grad_norm=-1)

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file')
    parser.add_argument('dataset_config', help='path to dataset config file')
    parser.add_argument('--experiment', help='experiment name', default='default')
    parser.add_argument('--resume', help='resume from checkpoint path', default=None)
    parser.add_argument('--use_loftr_preds', action='store_true',)
    parser.add_argument('--use_superglue_preds', action='store_true',)
    parser.add_argument('--not_strict', action='store_true',)
    parser.add_argument("--gpus", default=1, type=int, help="numgpu")
    parser.add_argument('--use_vanilla_transformer', action='store_true',)
    parser.add_argument("--d_model", default=32, type=int)
    parser.add_argument("--max_steps", default=200_000, type=int)
    parser.add_argument('--use_prior', action='store_true',)
    
    args = parser.parse_args()

    main(args)
