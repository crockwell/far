
from collections import defaultdict
from loguru import logger
from pathlib import Path
import pickle as pkl
import os

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.loftr import LoFTR
from src.baselines.simple_transformer import SimpleTransformer
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine, compute_supervision_RT
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics,
    aggregate_metrics_interiornet_streetlearn
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None, split=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        if config.USE_CORRESPONDENCE_TRANSFORMER:
            _config['loftr']['part2'] = False
            self.matcher = SimpleTransformer(config=_config['loftr'])
        else:
            # Matcher: LoFTR
            self.matcher = LoFTR(config=_config['loftr'])
            
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        self.pretrained_ckpt = None
        if pretrained_ckpt:
            self.pretrained_ckpt = pretrained_ckpt
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']

            if config.USE_CORRESPONDENCE_TRANSFORMER:
                state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}
            strict=True
            if config.STRICT_FALSE:
                strict=False
            if self.config.LOFTR.TRAINING and self.config.LOFTR.USE_MANY_RANSAC_THR:
                # need to take loftr_regress.moe_predictor.0.weight out of state dict
                del state_dict['matcher.loftr_regress.moe_predictor.0.weight']
                del state_dict['matcher.loftr_regress.moe_predictor.0.bias']

            if self.config.LOFTR.TRAINING and self.config.LOFTR.REGRESS.USE_1WT:
                # need to take moe_predictor.4 out of state dict
                del state_dict['matcher.loftr_regress.moe_predictor.4.weight']
                del state_dict['matcher.loftr_regress.moe_predictor.4.bias']

            self.matcher.load_state_dict(state_dict, strict=strict)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        else:
            print("\n\n\n WARNING, NOT LOADING PRETRAINED CHECKPOINT! \n\n\n")

        self.pl_version = config.PL_VERSION

        # Testing
        self.dump_dir = dump_dir
        self.split = split

    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        # check if gradients all valid
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print("detected inf or nan values in gradients. not updating model parameters")
            optimizer.zero_grad()

        optimizer.step(closure=optimizer_closure)

        optimizer.zero_grad()

    def _trainval_inference(self, batch, train=False):
        if self.config.LOFTR.FROM_SAVED_PREDS is None and not self.config.USE_CORRESPONDENCE_TRANSFORMER:
            if not self.config.DATASET.TRAINVAL_DATA_SOURCE == 'interiornet_streetlearn':
                with self.profiler.profile("Compute coarse supervision"):
                    compute_supervision_coarse(batch, self.config)

            with self.profiler.profile("LoFTR"):
                self.matcher(batch, train=train)

            if not self.config.DATASET.TRAINVAL_DATA_SOURCE == 'interiornet_streetlearn':
                with self.profiler.profile("Compute fine supervision"):
                    compute_supervision_fine(batch, self.config)
                        
            batch.update({
                'num_correspondences_before_ransac': 0,
                'num_correspondences_after_ransac': 0
            })
        else:
            batch.update({
                'num_correspondences_before_ransac': 0,
                'num_correspondences_after_ransac': 0
            })
        
        if self.config.LOFTR.REGRESS_RT:
            if (self.config.USE_CORRESPONDENCE_TRANSFORMER and not self.config.USE_PRED_CORR and self.config.LOFTR.REGRESS.USE_SIMPLE_MOE) or \
                (self.config.USE_CORRESPONDENCE_TRANSFORMER and (self.config.LOFTR.SOLVER == 'prior_ransac')) or \
                (not self.config.USE_CORRESPONDENCE_TRANSFORMER and self.config.LOFTR.REGRESS.USE_SIMPLE_MOE):   
                batch['translation_scale'] = None
                compute_supervision_RT(batch, self.config)

            for i in range(self.config.LOFTR.FINE_PRED_STEPS):
                if i < self.config.LOFTR.FINE_PRED_STEPS - 1 and 'prior_ransac' in self.config.LOFTR.SOLVER:
                    with torch.no_grad():
                        with self.profiler.profile("LoFTR"):
                            self.matcher.forward_rt_prediction(batch)
                        compute_supervision_RT(batch, self.config)
                else:
                    with self.profiler.profile("LoFTR"):
                        self.matcher.forward_rt_prediction(batch)
                    with self.profiler.profile("Compute losses"):
                        self.loss(batch)
        else:
            with self.profiler.profile("Compute losses"):
                self.loss(batch)
    
    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            if ((self.config.LOFTR.FROM_SAVED_PREDS is None or \
                ('SAVE_PREDS' in self.config.LOFTR and self.config.LOFTR.SAVE_PREDS is not None and 'ground_truth' in self.config.LOFTR.SAVE_PREDS)) \
                    and not self.config.USE_CORRESPONDENCE_TRANSFORMER):
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            else:
                batch.update({
                    'epi_errs': torch.tensor([[0]]),
                    'm_bids': torch.tensor([[0]])
                })
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            if self.config.USE_CORRESPONDENCE_TRANSFORMER:
                bs = batch['correspondence_details'].shape[0]
            else:
                bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                't_errs_abs': batch['t_errs_abs'],
                'inliers': batch['inliers'],
                'successful_fits': batch['successful_fits'],
                'gt_R': batch['T_0to1'][:,:3,:3].cpu(),
                'pred_R': torch.from_numpy(batch['pred_R']).unsqueeze(0).cpu(),
                'pred_t': torch.from_numpy(batch['pred_t']).unsqueeze(0).cpu(),
                'lightweight_numcorr': [x.cpu().numpy() for x in batch['lightweight_numcorr']],
            }
            if 'scene_root' in batch:
                metrics['scene_root'] = batch['scene_root']
                metrics['pair_names'] = batch['pair_names'][1]

            if self.config.LOFTR.REGRESS.SAVE_GATING_WEIGHTS:
                metrics['gating_reg_weights'] = batch['gating_reg_weights']
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch, train=True)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # net-params
            if self.config.LOFTR.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
                self.logger.experiment.add_scalar(
                    f'skh_bin_score', self.matcher.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures
            if self.config.TRAINER.ENABLE_PLOTTING and not self.config.LOFTR.REGRESS_RT:
                compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                ct = 0
                for k, v in figures.items():
                    if ct == 0:
                        self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)
                    else:
                        self.logger.experiment.add_figure(f'gt_match/{k}', v, self.global_step)
                    ct += 1

        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        ret_dict, _ = self._compute_metrics(batch)
        
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        if batch_idx % val_plot_interval == 0 and not (self.config.LOFTR.REGRESS_RT):
            figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        
    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and ((self.pl_version == '1.6.0' and self.trainer.sanity_checking) \
                or (self.pl_version != '1.6.0' and self.trainer.running_sanity_check)):
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            for _ls in _loss_scalars:
                for k in _loss_scalars[0]:
                    _ls[k] = torch.nan_to_num(_ls[k], nan=0.0, posinf=0.0, neginf=0.0)
                    if k in ['loss_f', 'loss_rot', 'loss_tr']:
                        _ls[k] = _ls[k].double()
                    elif k in ['num_correspondences_after_ransac']:
                        _ls[k] = _ls[k].long()
                    else:
                        _ls[k] = _ls[k].float()

            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            multi_val_metrics['rot_mean_err'].append(val_metrics_4tb['rot mean err'])
            multi_val_metrics['rot_median_err'].append(val_metrics_4tb['rot median err'])
            multi_val_metrics['tr_abs_mean_err'].append(val_metrics_4tb['tr abs mean err'])
            
            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).float().mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                
                for k, v in figures.items():
                    if self.trainer.global_rank == 0:
                        for plot_idx, fig in enumerate(v):
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        self.log(f'rot_median_err', torch.tensor(np.mean(multi_val_metrics['rot_median_err'])))
        self.log(f'rot_mean_err', torch.tensor(np.mean(multi_val_metrics['rot_mean_err'])))
        self.log(f'tr_abs_mean_err', torch.tensor(np.mean(multi_val_metrics['tr_abs_mean_err'])))

    def test_step(self, batch, batch_idx, skip_eval=False):
        if self.config.LOFTR.FROM_SAVED_PREDS is None and not self.config.USE_CORRESPONDENCE_TRANSFORMER:
            with self.profiler.profile("LoFTR"):
                self.matcher(batch)

        if self.config.LOFTR.REGRESS_RT or self.config.USE_CORRESPONDENCE_TRANSFORMER:
            if (self.config.USE_CORRESPONDENCE_TRANSFORMER and not self.config.USE_PRED_CORR and self.config.LOFTR.REGRESS.USE_SIMPLE_MOE) or \
                (self.config.USE_CORRESPONDENCE_TRANSFORMER and (self.config.LOFTR.SOLVER == 'prior_ransac')) or \
                (self.config.LOFTR.SOLVER == "prior_ransac_noprior") or \
                (self.config.LOFTR.FROM_SAVED_PREDS is None and self.config.LOFTR.REGRESS.USE_SIMPLE_MOE and not self.config.USE_CORRESPONDENCE_TRANSFORMER):
                batch['translation_scale'] = None
                compute_supervision_RT(batch, self.config)

            for i in range(self.config.LOFTR.FINE_PRED_STEPS):
                with self.profiler.profile("LoFTR"):
                    self.matcher.forward_rt_prediction(batch)

                if i < self.config.LOFTR.FINE_PRED_STEPS - 1 and ('prior_ransac' in self.config.LOFTR.SOLVER or self.config.LOFTR.SOLVER == "prior_ransac_noprior"):
                    compute_supervision_RT(batch, self.config)

        if not skip_eval:
            ret_dict, rel_pair_names = self._compute_metrics(batch)

            if self.config.SAVE_PREDS is not None:
                parent = self.config.SAVE_PREDS

                if not self.config.NO_SAVE_PREDS:
                    # preds
                    preds = torch.from_numpy(np.concatenate([batch['pred_R'], np.expand_dims(batch['pred_t'], axis=1)],axis=1))
                    predssavepath = os.path.join(parent, self.split, 'loftr_preds', str(batch['pair_id'].cpu().item())+'.pt')
                    torch.save(preds, predssavepath)

                if not self.config.NO_SAVE_NUMCORR and len(batch['num_correspondences_after_ransac']) > 0:
                    nc = torch.tensor(batch['num_correspondences_after_ransac'][0])
                    ncsavepath = os.path.join(parent, self.split, 'loftr_num_correspondences', str(batch['pair_id'].cpu().item())+'.pt')
                    torch.save(nc, ncsavepath)

                if self.config.SAVE_HARD_CORRES:
                    # save as hard correspodnences
                    
                    correspondence = batch['correspondences'].cpu().float()
                    correspondence_feats = batch['correspondences_feats'].cpu().float()
                    postfix = ''
                    if self.config.SAVE_CORR_AFTER_RANSAC:
                        postfix = '_after_ransac'

                    concat_correspondence = torch.cat([correspondence, correspondence_feats], dim=-1) # x,y,feat = 258

                    if self.config.EVAL_SPLIT == 'test':
                        save_path = os.path.join(parent, self.split, 'hard_correspondences'+postfix, str(batch['pair_id'].cpu().item())+'.pt')
                        torch.save(concat_correspondence, save_path)
                    elif concat_correspondence.shape[0] > 5:
                        save_path = os.path.join(parent, self.split, 'hard_correspondences_fit_only'+postfix, str(batch['pair_id'].cpu().item())+'.pt')
                        torch.save(concat_correspondence, save_path)
                elif self.config.LOFTR.REGRESS.SAVE_MLP_FEATS:
                        # mlp feats
                        featssavepath = os.path.join(parent, self.split, 'mlp_feats', str(batch['pair_id'].cpu().item())+'.pt')
                        torch.save(batch['mlp_feats'][0].cpu(), featssavepath)

                        # preds
                        preds = torch.from_numpy(np.concatenate([batch['pred_R'], np.expand_dims(batch['pred_t'], axis=1)],axis=1))
                        predssavepath = os.path.join(parent, self.split, 'loftr_preds', str(batch['pair_id'].cpu().item())+'.pt')
                        torch.save(preds, predssavepath)
                else:
                    if self.config.SAVE_CORR:
                        if len(batch['num_correspondences_after_ransac']) > 0:
                            if 'correspondences' in batch:
                                corrsavepath = os.path.join(parent, self.split, 'loftr_fine_correspondences', str(batch['pair_id'].cpu().item())+'.pt')
                                torch.save(batch['correspondences'], corrsavepath)
                        

            with self.profiler.profile("dump_results"):
                if self.dump_dir is not None:
                    # dump results for further analysis
                    keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                    pair_names = list(zip(*batch['pair_names']))
                    bs = batch['image0'].shape[0]
                    dumps = []
                    for b_id in range(bs):
                        item = {}
                        mask = batch['m_bids'] == b_id
                        item['pair_names'] = pair_names[b_id]
                        item['identifier'] = '#'.join(rel_pair_names[b_id])
                        for key in keys_to_save:
                            item[key] = batch[key][mask].cpu().numpy()
                        for key in ['R_errs', 't_errs', 'inliers']:
                            item[key] = batch[key][b_id]
                        dumps.append(item)
                    ret_dict['dumps'] = dumps

            if self.config.EVAL_FIT_ONLY and ret_dict['metrics']['successful_fits'][0] == 0:
                for key in ret_dict['metrics']:
                    ret_dict['metrics'][key] = []

            return ret_dict
        else:
            return batch
            

    def plot_errors(self, metrics, plot=False):
        for name, xlim in zip(['t_errs_abs', 't_errs', 'R_errs', 't_errs', 'R_errs'],
                              [10,90,180,20,20]):
                              
            if self.config.EXP_NAME != '' or self.pretrained_ckpt is None:
                bname = os.path.join("logs/tb_logs",self.config.EXP_NAME)
                dname = os.path.join("logs/tb_logs",self.config.EXP_NAME)
            else:
                dname = os.path.dirname(os.path.dirname(self.pretrained_ckpt))
                bname = os.path.basename(os.path.dirname(self.pretrained_ckpt))
                
            if plot:
                x = np.sort(metrics[name])
                y = np.arange(1, len(x) + 1) / len(x)
                plt.plot(x, y)
                plt.xlim(0, xlim)
                plt.xlabel(name)
                plt.ylabel('CDF')
                plt.title(name + ' CDF: ' + bname)
                plt.savefig(os.path.join(dname, self.split+"_"+name+"_thr_"+str(xlim)+".png"))
                plt.clf()

            # also save metrics[name] so we can compute this later
            np.save(os.path.join(dname, self.split+"_"+name+".npy"), metrics[name])

        # in map-free case, need to save dict with dict[scene] = np.array([range(n)]) where n=num in scene
        if 'scene_root' in metrics:
            out_dict = {}
            for i in range(len(metrics['scene_root'])):
                scene = metrics['scene_root'][i]
                if scene not in out_dict:
                    out_dict[scene] = {}
                idx = int(metrics['pair_names'][i].split('_')[-1].split(".")[0])
                out_dict[scene][idx] = {'pred_R': metrics['pred_t'][i], 'pred_t': metrics['pred_R'][i]}
            with open(os.path.join(dname, self.split+"_preds.pkl"),'wb') as f:
                pkl.dump(out_dict, f)
        else:
            np.save(os.path.join(dname, self.split+"_pred_R.npy"), np.array([x.numpy() for x in metrics['pred_R']]))
            np.save(os.path.join(dname, self.split+"_pred_t.npy"), np.array([x.numpy() for x in metrics['pred_t']]))

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        

        if self.trainer.global_rank == 0:
            if 'interiornet' in self.config.DSET_NAME or 'streetlearn' in self.config.DSET_NAME:
                val_metrics_4tb = aggregate_metrics_interiornet_streetlearn(metrics, self.config.TRAINER.EPI_ERR_THR)
            else:
                val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            # file to save to

            for key in val_metrics_4tb:
                if 'tr' in key or 'rot' in key or 'pct' in key or 'dset size' in key:
                    print(f"{key} {str(val_metrics_4tb[key])}")
            
            print("")
            for key in val_metrics_4tb:
                if 'auc' in key:
                    print(f"{key} {str(val_metrics_4tb[key])}")

            if self.config.EXP_NAME != '' or self.pretrained_ckpt is None:
                dirname = os.path.join("logs/tb_logs",self.config.EXP_NAME)
                os.makedirs(dirname, exist_ok=True)
            else:
                dirname = os.path.dirname(os.path.dirname(self.pretrained_ckpt))
            
            results_name = "results.txt" 
            if self.config.LOFTR.SOLVER != 'ransac':
                results_name = f"results_{self.config.LOFTR.SOLVER}.txt"
            elif self.config.TRAINER.RANSAC_PIXEL_THR != 0.5:
                results_name = f"results_ransac_thr_{self.config.TRAINER.RANSAC_PIXEL_THR}.txt"

            if self.config.CORRESPONDENCES_USE_FIT_ONLY:
                results_name = f"use_fit_only_{results_name}"

            if self.config.MAX_CORRESPONDENCES < 2000:
                results_name = f"max_corrs_{self.config.MAX_CORRESPONDENCES}_{results_name}"

            if self.config.OUTLIER_PCT > 0:
                results_name = f"outlier_pct_{self.config.OUTLIER_PCT}_{results_name}"

            if self.config.NOISE_PIX > 0:
                results_name = f"noise_pix_{self.config.NOISE_PIX}_{results_name}"

            if self.config.MISSING_PCT > 0:
                results_name = f"missing_pct_{self.config.MISSING_PCT}_{results_name}"

            with open(os.path.join(dirname, self.split + "_" + results_name), "w") as f:
                for key in val_metrics_4tb:
                    if 'tr' in key or 'rot' in key or 'pct' in key or 'dset size' in key:
                        f.write(f"{key} {str(val_metrics_4tb[key])} \n")

                print("")
                for key in val_metrics_4tb:
                    if 'auc' in key:
                        print(f"{key} {str(val_metrics_4tb[key])}")

            # plot CDF of errors
            if True:
                self.plot_errors(metrics)

            # also save the number of post-ransac correspondences (if exist)
            if np.array(metrics['inliers']).dtype != 'int64':
                np.save(os.path.join(dirname, self.split+"_num_correspondences_after_ransac.npy"), \
                        np.array([x.sum() for x in metrics['inliers']]))

                np.save(os.path.join(dirname, self.split+"_num_correspondences_before_ransac.npy"), \
                        np.array([x.shape[0] for x in metrics['inliers']]))

            if self.config.LOFTR.REGRESS.SAVE_GATING_WEIGHTS:
                np.save(os.path.join(dirname, self.split+"_gating_reg_weights.npy"), \
                        np.array([x.cpu().numpy() for x in metrics['gating_reg_weights']]))

            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)
