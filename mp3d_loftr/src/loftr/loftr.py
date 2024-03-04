import torch
import torch.nn as nn
import numpy as np
from einops.einops import rearrange
import torch.nn.functional as F

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess, LocalFeatureTransformerRegressor
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from src.losses.loftr_loss import compute_normalized_6d, rotation_6d_to_matrix, pose_mean_6d, pose_std_6d

class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        if self.config['from_saved_preds'] is None or ('save_preds' in self.config and self.config['save_preds'] is not None and 'ground_truth' in self.config['save_preds']):
            # Modules
            self.backbone = build_backbone(config)
            self.pos_encoding = PositionEncodingSine(
                config['coarse']['d_model'],
                temp_bug_fix=config['coarse']['temp_bug_fix'])
            self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
            self.coarse_matching = CoarseMatching(config['match_coarse'])
            self.fine_preprocess = FinePreprocess(config)
            self.loftr_fine = LocalFeatureTransformer(config["fine"])
            self.fine_matching = FineMatching(config)
            if 'predict_translation_scale' in self.config and self.config['predict_translation_scale']:
                H = 512
                self.translation_scale_predictor1 = nn.Sequential( # 2, 256, 60, 80
                    nn.MaxPool2d(2),
                    nn.Conv2d(256, 128, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 64, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 16, 1),
                    nn.ReLU(),
                )
                
                self.translation_scale_predictor2 = nn.Sequential( # 2, 256, 60, 80
                    nn.Linear(15*20*16*2, H), 
                    nn.ReLU(), 
                    nn.Linear(H, H), 
                    nn.ReLU(), 
                    nn.Linear(H, 1),
                )
            
        if self.config['regress_rt']:
            self.loftr_regress = LocalFeatureTransformerRegressor(config)
        

    def forward_feature_extraction(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        data.update({
            'featmap0': feat_c0,
            'featmap1': feat_c1,
            'featmap_f0': feat_f0,
            'featmap_f1': feat_f1,
            'feats_c': feats_c,
        })

    def forward_correspondence_prediction(self, data, train=False):
        feat_c0 = data['featmap0']
        feat_c1 = data['featmap1']
        feat_f0 = data['featmap_f0']
        feat_f1 = data['featmap_f1']
        feats_c = data['feats_c']

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        # mask is None, feat_c0 is 1,4800,256 before & after
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)

        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data, train=train)

        # 6b. transformer predicts scale from features (if desired)
        pred_tran_scale = None
        if 'predict_translation_scale' in self.config and self.config['predict_translation_scale']:
            shrunk_feats = self.translation_scale_predictor1(feats_c) # 2, 256, 60, 80
            
            pred_tran_scale = self.translation_scale_predictor2(shrunk_feats.reshape([data['bs'],-1]))[:,0]
                        
        data.update({
            'featmap0': feat_c0,
            'featmap1': feat_c1,
            'mask_c0': mask_c0,
            'mask_c1': mask_c1,
            'translation_scale': pred_tran_scale
        })

    def preprocess_helper(self, data):
        feat_c0, feat_c1 = data['featmap0'], data['featmap1']
        if 'mask_c0' not in data:
            mask_c0, mask_c1 = None, None
        else:
            mask_c0 = data['mask_c0']
            mask_c1 = data['mask_c1']

        # 6a. transformer predicts RT from coarse correspondence features (if desired)
        loftr_preds_6d, inv_loftr_preds_6d, F = None, None, None
        if self.config['regress']['use_simple_moe']:
            loftr_preds = data['loftr_rt'].detach() # no gradient to real 8Pt machinery through this branch
            if len(loftr_preds.shape) > 2:
                loftr_preds = loftr_preds.squeeze(0)

            # map to normalized translation + 6d coords
            loftr_preds_6d = compute_normalized_6d(loftr_preds.float()).unsqueeze(0)
            loftr_preds_6d_4x4 = torch.cat([loftr_preds,torch.tensor([[0,0,0,1.]]).cuda()],dim=0)
            inv_loftr_preds_6d = compute_normalized_6d(torch.linalg.inv(loftr_preds_6d_4x4)[:3,:4]).float().unsqueeze(0)

            if self.config['regress']['regress_use_num_corres']:
                num_correspondences = data['num_correspondences'].detach().float().unsqueeze(0) / 500 # make similar scale to 6d
                loftr_preds_6d = torch.cat([loftr_preds_6d, num_correspondences],dim=-1)
                inv_loftr_preds_6d = torch.cat([inv_loftr_preds_6d, num_correspondences], dim=-1)

            if self.config['use_many_ransac_thr']:
                num_correspondences = torch.cat([
                    data['num_correspondences_before_ransac'].detach().float().unsqueeze(0) / 500,
                    data['inliers_best_tight'].detach().float().unsqueeze(0) / 500,
                    data['inliers_best_ultra_tight'].detach().float().unsqueeze(0) / 500,
                ], dim=-1)
                loftr_preds_6d = torch.cat([loftr_preds_6d, num_correspondences],dim=-1)
                inv_loftr_preds_6d = torch.cat([inv_loftr_preds_6d, num_correspondences], dim=-1)
            
        return feat_c0, feat_c1, mask_c0, mask_c1, loftr_preds_6d, inv_loftr_preds_6d

    def forward_rt_prediction(self, data):
        if self.config['regress_rt']:
            feat_c0, feat_c1, mask_c0, mask_c1, loftr_preds_6d, \
                inv_loftr_preds_6d = self.preprocess_helper(data)
                
            pred_RT, mlp_features, pred_RT_wt = self.loftr_regress(feat_c0, feat_c1, mask0=mask_c0, mask1=mask_c1, 
                                            loftr_preds=loftr_preds_6d, inv_loftr_preds=inv_loftr_preds_6d, F=F)
            data.update({'regressed_rt': pred_RT,
                            'expec_rt': pred_RT[0]},)
            if self.config['regress']['save_mlp_feats']:
                data.update({'mlp_feats': mlp_features})
            
            if self.config['regress']['save_gating_weights']:
                data.update({'gating_reg_weights': pred_RT_wt})

            if self.config['solver'] == 'prior_ransac':
                R = data['regressed_rt'][:,3:].detach().cpu() * pose_std_6d[3:] + pose_mean_6d[3:]
                t = data['regressed_rt'][0,:3].detach().cpu().numpy()* pose_std_6d[:3].numpy() + pose_mean_6d[:3].numpy()
                R = rotation_6d_to_matrix(R)[0].numpy()
                data.update({'priorRT': np.concatenate([R,np.expand_dims(t,1)],axis=-1)})

    def forward(self, data, train=False):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        self.forward_feature_extraction(data)
        self.forward_correspondence_prediction(data, train=train)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
