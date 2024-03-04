import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np

from lib.models.regression.aggregator import *
from lib.models.regression.head import *

from lib.utils.loss import *
from lib.utils.metrics import pose_error_torch, error_auc, A_metrics
from lib.models.matching.pose_solver import EssentialMatrixSolver

from yacs.config import CfgNode as CN
from etc.feature_matching_baselines.LoFTR.src.loftr import LoFTR, default_cfg
from etc.feature_matching_baselines.SuperGlue.models.matching import Matching

from lib.models.regression.encoder.resnet import ResNet
from lib.models.regression.encoder.resunet import ResUNet

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

class RegressionModel(pl.LightningModule):
    """Regresses Relative Pose between a pair of images"""

    def __init__(self, cfg, use_loftr_preds=False, use_superglue_preds=False, ckpt_path=None, 
                 not_strict=False, inference=False, use_vanilla_transformer=False,
                 d_model=32, max_steps=200_000, 
                 use_prior=False):
        super().__init__()

        self.cfg = cfg

        # initialise encoder
        try:
            encoder = eval(cfg.ENCODER.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid encoder {cfg.ENCODER.TYPE}')
        self.encoder = encoder(cfg.ENCODER)

        self.max_steps = max_steps
        self.use_prior = use_prior

        # head
        self.use_vanilla_transformer = use_vanilla_transformer

        if use_vanilla_transformer:
            num_heads = 8
            num_layers = 6
            d_im = 256
            self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_im, nhead=num_heads), num_layers=num_layers)

        # aggregator
        try:
            aggregator = eval(cfg.AGGREGATOR.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid aggregator {cfg.AGGREGATOR.TYPE}')
        self.aggregator = aggregator(cfg.AGGREGATOR, self.encoder.num_out_layers)

        try:
            head = eval(cfg.HEAD.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid head {cfg.HEAD.TYPE}')
        self.head = head(cfg, self.aggregator.num_out_layers, full_forward_pass=False)


        # initialise loss function
        try:
            self.rot_loss = eval(cfg.TRAINING.ROT_LOSS)
        except NameError:
            raise NotImplementedError(f'Invalid rotation loss {cfg.TRAINING.ROT_LOSS}')
        try:
            self.trans_loss = eval(cfg.TRAINING.TRANS_LOSS)
        except NameError:
            raise NotImplementedError(f'Invalid translation loss {cfg.TRAINING.TRANS_LOSS}')

        # set loss function weights
        # if LAMBDA is 0., use learnable weights from
        # Geometric loss functions for camera pose regression with deep learning (Kendal & Cipolla)
        self.LAMBDA = cfg.TRAINING.LAMBDA
        if cfg.TRAINING.LAMBDA == 0.:
            self.s_r = torch.nn.Parameter(torch.zeros(1))
            self.s_t = torch.nn.Parameter(torch.zeros(1))

        if use_loftr_preds or use_superglue_preds or self.use_vanilla_transformer:
            self.H2 = 512
            self.H = 256*12*9
            self.pose_size = 9

        if use_loftr_preds or use_superglue_preds:
            outdoor = True
            if use_loftr_preds:
                self.matcher = LoFTR(config=default_cfg)
                weights_path = "etc/feature_matching_baselines/LoFTR/weights/outdoor_ot.ckpt" if outdoor else "etc/feature_matching_baselines/LoFTR/weights/indoor_ot.ckpt"
                self.matcher.load_state_dict(torch.load(weights_path)['state_dict'], strict=False)
                self.matcher = self.matcher.eval()
            else:
                nms_radius = 4
                keypoint_threshold = 0.005
                max_keypoints = 1024

                superglue_weights = 'outdoor' if outdoor else 'indoor'  # indoor trained on scannet
                sinkhorn_iterations = 20
                match_threshold = 0.2

                # Load the SuperPoint and SuperGlue models.
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
                self.matching = Matching(config).eval()

            self.pose_solver = EssentialMatrixSolver(cfg.SOLVER, self.use_prior)
            
            self.num_corr_size = 3
            self.moe_predictor = nn.Sequential(
                nn.Linear(self.H+2*self.pose_size+self.num_corr_size, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(),
                nn.Linear(self.H2, 2),
                nn.Sigmoid(),
            )

        if self.use_vanilla_transformer:
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.pose_size),
            )

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            self.load_state_dict(state_dict, strict=not not_strict)

        self.use_loftr_preds = use_loftr_preds
        self.use_superglue_preds = use_superglue_preds
        if not inference and use_loftr_preds or use_superglue_preds:
            if use_loftr_preds:
                all_params = [x for x in self.matcher.parameters()]
            elif use_superglue_preds:
                all_params = [x for x in self.matching.parameters()]
            
            for param in all_params:
                param.requires_grad = False

    def match_no_load(self, inp0, inp1):
        batch = {'image0': inp0, 'image1': inp1}
        if self.use_loftr_preds:
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
        elif self.use_superglue_preds:
            pred = self.matching(batch)
            pred = {k: v[0] for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches = pred['matches0']

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid].cpu().numpy()
            mkpts1 = kpts1[matches[valid]].cpu().numpy()

        if mkpts0.shape[0] > 0:
            return mkpts0, mkpts1
        else:
            print("no correspondences")
            return np.zeros((1, 2)), np.zeros((1, 2))

    def transformer_head(self, features):
        B = features.shape[0]
        feats_in = features.reshape([B, -1])
        pred_reg_6d = self.pose_regressor(feats_in).float()
        t = pred_reg_6d[...,:3]
        R = pred_reg_6d[...,3:]
        return R, t

    def regression_mlp(self, features, loftr_preds, loftr_num_corr, R=None, t=None):
        # features are of size 6,67,92,68 -- need to be much smaller
        # loftr preds are only for batch size = 1 -- need to be on full batch

        B = features.shape[0]
        loftr_preds_6d_rot = compute_6d(loftr_preds[...,:3,:3].float())
        loftr_preds_6d = torch.cat([loftr_preds[..., 3], loftr_preds_6d_rot], dim=-1)
        if self.use_prior:
            num_correspondences = loftr_num_corr.detach().float() / 500 # make similar scale to 6d
        else:
            num_correspondences = loftr_num_corr.detach().float() / 500 # make similar scale to 6d
            if len(loftr_num_corr.shape) == 1:
                num_correspondences = num_correspondences.unsqueeze(0)
            num_correspondences = torch.cat([num_correspondences, num_correspondences/10, num_correspondences/100],dim=-1).cuda() 
            # vanilla RANSAC filler value before prior returns 3 values
        loftr_preds_6d = torch.cat([loftr_preds_6d, num_correspondences],dim=-1).cuda()
        
        if (self.use_vanilla_transformer):
            feats_in = features.reshape([B, -1])
            pred_reg_6d = self.pose_regressor(feats_in).float()
        else:
            pred_reg_6d = torch.cat([t[:,0], compute_6d(R[...,:3,:3].float())], dim=-1)

        normalized_t = loftr_preds_6d[...,:3] * torch.clamp(
            (torch.linalg.norm(pred_reg_6d[...,:3], dim=-1) / 
             torch.clamp(torch.linalg.norm(loftr_preds_6d[...,:3], dim=-1),1e-2,1e2)).unsqueeze(1),1e-2, 1e2)
        loftr_preds_6d_out = torch.cat([normalized_t, loftr_preds_6d[...,3:]], dim=-1)

        feats_preds = torch.cat([features.reshape([B, -1]), pred_reg_6d, loftr_preds_6d_out],dim=-1)

        pred_RT_wt = self.moe_predictor(feats_preds)
        
        t = pred_RT_wt[...,:1] * pred_reg_6d[...,:3] + (1-pred_RT_wt[...,:1]) * loftr_preds_6d_out[...,:3]
        R = pred_RT_wt[...,1:] * pred_reg_6d[...,3:] + (1-pred_RT_wt[...,1:]) * loftr_preds_6d_out[...,3:-self.num_corr_size]

        return R, t
    
    def forward(self, data):
        priorRT = None
        num_loops = 1
        if self.use_prior:
            num_loops = 2
        
        for loop in range(num_loops):
            if self.use_loftr_preds or self.use_superglue_preds:
                # run matcher + 8pt
                all_inliers, all_loftr_rt = [], []
                for i in range(len(data['image0'])):
                    with torch.no_grad():
                        data['mkpts0_f'], data['mkpts1_f'] = self.match_no_load(data['image0'][i:i+1], data['image1'][i:i+1])
                        data2 = {'K_color0': data['K_color0'][i:i+1].cpu(), 'K_color1': data['K_color1'][i:i+1].cpu()}
                        if priorRT is not None:
                            this_priorRT = priorRT[i]
                        else:
                            this_priorRT = None

                        ret, inliers_best_tight, inliers_best_ultra_tight = self.pose_solver.estimate_pose(data['mkpts0_f'], data['mkpts1_f'], 
                                                                                                                    data2, this_priorRT)

                        R_, t_, inliers = ret
                        if self.use_prior:
                            if inliers is not None and inliers_best_tight is not None and \
                                type(inliers) is int and type(inliers_best_tight) is int:
                                inliers = torch.tensor([[inliers, inliers_best_tight, inliers_best_ultra_tight]]).cuda()
                            else:
                                inliers = torch.tensor([[0., 0., 0.]]).cuda()
                        else:
                            inliers = torch.tensor([[inliers]]).cuda()
                        if R_ is not None and t_ is not None and R_.shape[0] > 0 and t_.shape[0] > 0:
                            loftr_rt = torch.cat([torch.from_numpy(R_.copy()).unsqueeze(0).float(), 
                                                        torch.from_numpy(t_.copy()).unsqueeze(-1).unsqueeze(0).float()], axis=-1).cuda()
                        else:
                            loftr_rt = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0]]).unsqueeze(0).float().cuda()
                        
                        all_inliers.append(inliers)
                        all_loftr_rt.append(loftr_rt)

                data['loftr_rt'] = torch.cat(all_loftr_rt)
                data['inliers'] = torch.cat(all_inliers)

            with torch.set_grad_enabled(loop==num_loops-1):

                # img is B, 3, 360, 270 for regression
                # img is B, 3, 544, 720 for matching (padded for network)
                vol0 = self.encoder(data['image0_reg'])
                vol1 = self.encoder(data['image1_reg']) # (B, 32, 92, 68)

                global_volume = self.aggregator(vol0, vol1) # (B, 27, 92, 68)
                R, t, feats = self.head(global_volume, data)

                if self.use_vanilla_transformer:
                    B, C, H, W = feats.shape
                    feats_in = feats.reshape([B,C,H*W])
                    feats = self.transformer(feats_in.permute([2,0,1])).permute([1,2,0])

                if self.use_loftr_preds or self.use_superglue_preds:
                    R, t = self.regression_mlp(feats, data['loftr_rt'].float(), data['inliers'], R, t)
                    
                    if self.use_prior and loop < num_loops-1:
                        # need to convert this from 6d to 3x3 matrix
                        convertedR = rotation_6d_to_matrix(R)
                        priorRT = torch.cat([convertedR, t.unsqueeze(2)],dim=-1)
                elif self.use_vanilla_transformer:
                    R, t = self.transformer_head(feats)
                else:
                    data['inliers'] = 0

            
        data['R'] = R
        data['t'] = t
        return R, t

    def loss_fn(self, data):
        if self.use_loftr_preds or self.use_superglue_preds or self.use_vanilla_transformer:
            R_loss = self.rot_loss(data['R'], data['T_0to1'][..., :3, :3])
            t_loss = self.trans_loss(data['t'], data['T_0to1'][..., :3, 3])
        else:
            R_loss = self.rot_loss(data)
            t_loss = self.trans_loss(data)

        if self.LAMBDA == 0:
            # PoseNet (Kendal & Cipolla) lernable loss scaling
            loss = R_loss * torch.exp(-self.s_r) + t_loss * torch.exp(-self.s_t) + self.s_r + self.s_t
        else:
            loss = R_loss + self.LAMBDA * t_loss

        return R_loss, t_loss, loss

    def training_step(self, batch, batch_idx):
        if self.use_loftr_preds:
            self.matcher.eval()
        elif self.use_superglue_preds:
            self.matching.eval()

        self(batch)
        R_loss, t_loss, loss = self.loss_fn(batch)

        self.log('train/R_loss', R_loss)
        self.log('train/t_loss', t_loss)
        self.log('train/loss', loss)
        if self.LAMBDA == 0.:
            self.log('train/s_R', self.s_r)
            self.log('train/s_t', self.s_t)
        
        return loss

    def validation_step(self, batch, batch_idx):
        Tgt = batch['T_0to1']
        R, t = self(batch)
        R_loss, t_loss, loss = self.loss_fn(batch)

        if self.use_loftr_preds or self.use_superglue_preds or self.use_vanilla_transformer:
            R = rotation_6d_to_matrix(R) # from 6D to matrix

        # validation metrics
        outputs = pose_error_torch(R, t, Tgt, reduce=None)
        outputs['R_loss'] = R_loss
        outputs['t_loss'] = t_loss
        outputs['loss'] = loss
        return outputs

    def validation_epoch_end(self, outputs):
        # aggregates metrics/losses from all validation steps
        aggregated = {}
        for key in outputs[0].keys():
            aggregated[key] = torch.stack([x[key] for x in outputs])

        # compute stats
        median_t_ang_err = aggregated['t_err_ang'].median()
        median_t_scale_err = aggregated['t_err_scale'].median()
        median_t_euclidean_err = aggregated['t_err_euc'].median()
        median_R_err = aggregated['R_err'].median()
        mean_R_loss = aggregated['R_loss'].mean()
        mean_t_loss = aggregated['t_loss'].mean()
        mean_loss = aggregated['loss'].mean()

        # a1, a2, a3 metrics of the translation vector norm
        a1, a2, a3 = A_metrics(aggregated['t_err_scale_sym'])

        # compute AUC of Euclidean translation error for 10cm, 50cm and 1m thresholds
        AUC_euc_10, AUC_euc_50, AUC_euc_100 = error_auc(
            aggregated['t_err_euc'].view(-1).detach().cpu().numpy(),
            [0.1, 0.5, 1.0]).values()

        # compute AUC of pose error (max of rot and t ang. error) for 5, 10 and 20 degrees thresholds
        pose_error = torch.maximum(
            aggregated['t_err_ang'].view(-1),
            aggregated['R_err'].view(-1)).detach().cpu()
        AUC_pos_5, AUC_pos_10, AUC_pos_20 = error_auc(pose_error.numpy(), [5, 10, 20]).values()

        # compute AUC of rotation error 5, 10 and 20 deg thresholds
        rot_error = aggregated['R_err'].view(-1).detach().cpu()
        AUC_rot_5, AUC_rot_10, AUC_rot_20 = error_auc(rot_error.numpy(), [5, 10, 20]).values()

        # compute AUC of translation angle error 5, 10 and 20 deg thresholds
        t_ang_error = aggregated['t_err_ang'].view(-1).detach().cpu()
        AUC_tang_5, AUC_tang_10, AUC_tang_20 = error_auc(t_ang_error.numpy(), [5, 10, 20]).values()

        #if self.trainer.global_rank == 0:
        # log stats
        self.log('val_loss/R_loss', mean_R_loss)
        self.log('val_loss/t_loss', mean_t_loss)
        self.log('val_loss/loss', mean_loss)
        self.log('val_metrics/t_ang_err', median_t_ang_err)
        self.log('val_metrics/t_scale_err', median_t_scale_err)
        self.log('val_metrics/t_euclidean_err', median_t_euclidean_err)
        self.log('val_metrics/R_err', median_R_err)
        self.log('val_auc/euc_10', AUC_euc_10)
        self.log('val_auc/euc_50', AUC_euc_50)
        self.log('val_auc/euc_100', AUC_euc_100)
        self.log('val_auc/pose_5', AUC_pos_5)
        self.log('val_auc/pose_10', AUC_pos_10)
        self.log('val_auc/pose_20', AUC_pos_20)
        self.log('val_auc/rot_5', AUC_rot_5)
        self.log('val_auc/rot_10', AUC_rot_10)
        self.log('val_auc/rot_20', AUC_rot_20)
        self.log('val_auc/tang_5', AUC_tang_5)
        self.log('val_auc/tang_10', AUC_tang_10)
        self.log('val_auc/tang_20', AUC_tang_20)
        self.log('val_t_scale/a1', a1)
        self.log('val_t_scale/a2', a2)
        self.log('val_t_scale/a3', a3)

        print('mean R loss', mean_R_loss, 'mean t loss', mean_t_loss, 'mean loss', mean_loss, '\n', \
            'median t ang err', median_t_ang_err, 'median t scale err', median_t_scale_err, '\n', \
                'median t euclidean err', median_t_euclidean_err, 'median R err', median_R_err, '\n', \
                'AUC euc 10', AUC_euc_10, 'AUC euc 50', AUC_euc_50, 'AUC euc 100', AUC_euc_100, '\n', \
                'AUC pose 5', AUC_pos_5, 'AUC pose 10', AUC_pos_10, 'AUC pose 20', AUC_pos_20, '\n',\
                'AUC rot 5', AUC_rot_5, 'AUC rot 10', AUC_rot_10, 'AUC rot 20', AUC_rot_20, '\n', \
                'AUC tang 5', AUC_tang_5, 'AUC tang 10', AUC_tang_10, 'AUC tang 20', AUC_tang_20)

        return mean_loss

    def configure_optimizers(self):
        tcfg = self.cfg.TRAINING
        opt = torch.optim.Adam(self.parameters(), lr=tcfg.LR, eps=1e-6)
        if tcfg.LR_STEP_INTERVAL:
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, tcfg.LR_STEP_INTERVAL, tcfg.LR_STEP_GAMMA)
            return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        return opt
