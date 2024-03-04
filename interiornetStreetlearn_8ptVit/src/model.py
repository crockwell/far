import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .modules.extractor import ResidualBlock
from .modules.vision_transformer import _create_vision_transformer
from src.geom.RotationContinuity.sanity_test.code.tools import compute_rotation_matrix_from_ortho6d, compute_pose_from_rotation_matrix

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def compute_6d(pose_mtx):
    pose_6d = matrix_to_rotation_6d(pose_mtx[...,:3,:3])
    tr = pose_mtx[...,:3,3]
    pose_6d_norm = torch.cat([tr,pose_6d],dim=-1)
    return pose_6d_norm

def compute_normalized_6d(pose_mtx, global_pose_mean, global_pose_std):
    pose_6d = matrix_to_rotation_6d(pose_mtx[...,:3,:3])
    tr = pose_mtx[...,:3,3]
    pose_6d_norm = (torch.cat([tr,pose_6d],dim=-1) - global_pose_mean.to(device=pose_6d.device)) / global_pose_std.to(device=pose_6d.device)
    return pose_6d_norm

class ViTEss(nn.Module):
    def __init__(self, args, global_pose_mean=None, global_pose_std=None):
        super(ViTEss, self).__init__()

        # hyperparams
        self.total_num_features = 192
        self.feature_resolution = (24, 24)
        self.pose_size = 9
        self.num_patches = self.feature_resolution[0] * self.feature_resolution[1]
        extractor_final_conv_kernel_size = max(1, 28-self.feature_resolution[0]+1)
        self.pool_feat1 = min(96, 4 * args.pool_size)
        self.pool_feat2 = args.pool_size
        self.H2 = args.fc_hidden_size
        self.use_loftr_gating = args.use_loftr_gating
        self.global_pose_mean = global_pose_mean
        self.global_pose_std = global_pose_std
        self.use_normalized_6d = args.use_normalized_6d

        # layers
        self.flatten = nn.Flatten(0,1)
        self.resnet = models.resnet18(pretrained=True) # this will be overridden if we are loading pretrained model
        self.resnet.fc = nn.Identity()
        self.extractor_final_conv = ResidualBlock(128, self.total_num_features, 'batch', kernel_size=extractor_final_conv_kernel_size)

        self.fusion_transformer = None
        if args.fusion_transformer:
            self.num_heads = 3
            model_kwargs = dict(patch_size=16, embed_dim=self.total_num_features, depth=args.transformer_depth, 
                                num_heads=self.num_heads)
            self.fusion_transformer = _create_vision_transformer('vit_tiny_patch16_384', **model_kwargs)

            self.transformer_depth = args.transformer_depth
            self.fusion_transformer.blocks = self.fusion_transformer.blocks[:args.transformer_depth]
            self.fusion_transformer.patch_embed = nn.Identity()
            self.fusion_transformer.head = nn.Identity() 
            self.fusion_transformer.cls_token = None
            self.pos_encoding = None

            # we overwrite pos_embedding as we don't have class token
            self.fusion_transformer.pos_embed = nn.Parameter(torch.zeros([1,self.num_patches,self.total_num_features])) 
            # randomly initialize as usual 
            nn.init.xavier_uniform_(self.fusion_transformer.pos_embed) 

            pos_enc = 6
            self.H = int(self.num_heads*2*(self.total_num_features//self.num_heads + pos_enc) * (self.total_num_features//self.num_heads))
        else:
            self.H = self.pool_feat2 * self.feature_resolution[0] * self.feature_resolution[1]
            self.pool_transformer_output = nn.Sequential(
                nn.Conv2d(self.total_num_features, self.pool_feat1, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat1),
                nn.ReLU(),
                nn.Conv2d(self.pool_feat1, self.pool_feat2, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat2),
            )
        
        if self.use_loftr_gating:           
            self.moe_predictor = nn.Sequential(
                nn.Linear(self.H+2*self.pose_size+1, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(),
                nn.Linear(self.H2, 2),
                nn.Sigmoid(),
            )
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.pose_size),
            )
        else:
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.pose_size),
            )

        self.T_pose = args.T_pose

    def update_intrinsics(self, input_shape, intrinsics):
        sizey, sizex = self.feature_resolution
        scalex = sizex / input_shape[-1]
        scaley = sizey / input_shape[-2]
        xidx = np.array([0,2])
        yidx = np.array([1,3])
        intrinsics[:,:,xidx] = scalex * intrinsics[:,:,xidx]
        intrinsics[:,:,yidx] = scaley * intrinsics[:,:,yidx]
            
        return intrinsics

    def extract_features(self, images, intrinsics=None):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        if intrinsics is not None:
            intrinsics = self.update_intrinsics(images.shape, intrinsics)

        # for resnet, we need 224x224 images
        input_images = self.flatten(images)
        input_images = F.interpolate(input_images, size=224)

        x = self.resnet.conv1(input_images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) 
        x = self.resnet.layer1(x) # 64, 56, 56
        x = self.resnet.layer2(x) # 128, 28, 28       
        
        x = self.extractor_final_conv(x) # 192, 24, 24 

        x = x.reshape([input_images.shape[0], -1, self.num_patches])
        if self.fusion_transformer is None:
            features = x[:,:self.total_num_features//2]
        else:
            features = x[:,:self.total_num_features]
        features = features.permute([0,2,1])

        return features, intrinsics

    def forward(self, images, intrinsics=None, inference=False, loftr_num_corr=None, loftr_preds=None):
        """ Estimates SE3 between pair of frames """

        features, intrinsics = self.extract_features(images, intrinsics)
        B, _, _, _, _ = images.shape

        if self.fusion_transformer is not None:
            x = features[:,:,:self.total_num_features]
            x = self.fusion_transformer.patch_embed(x)
            x = x + self.fusion_transformer.pos_embed
            x = self.fusion_transformer.pos_drop(x)

            for layer in range(self.transformer_depth):
                x = self.fusion_transformer.blocks[layer](x, intrinsics=intrinsics)

            features = self.fusion_transformer.norm(x)
        else:
            reshaped_features = features.reshape([-1,self.feature_resolution[0],self.feature_resolution[1],self.total_num_features])
            features = self.pool_transformer_output(reshaped_features.permute(0,3,1,2))

        if self.use_loftr_gating:
            if self.use_normalized_6d:
                loftr_preds_6d = compute_normalized_6d(loftr_preds.float(), self.global_pose_mean, self.global_pose_std)
            else:
                loftr_preds_6d = compute_6d(loftr_preds.float())
            num_correspondences = loftr_num_corr.detach().float().unsqueeze(1) / 500 # make similar scale to 6d
            
            loftr_preds_6d = torch.cat([loftr_preds_6d, num_correspondences],dim=-1)
            pred_reg_6d = self.pose_regressor(features.reshape([B, -1]))
            feats_preds = torch.cat([features.reshape([B, -1]), pred_reg_6d, loftr_preds_6d],dim=-1)
            pred_RT_wt = self.moe_predictor(feats_preds)

            # only actually predict rotation, so don't worry about scaling
            pred_T = pred_RT_wt[...,:1] * pred_reg_6d[...,:3] + (1-pred_RT_wt[...,:1]) * loftr_preds_6d[...,:3]
            pred_R = pred_RT_wt[...,1:] * pred_reg_6d[...,3:] + (1-pred_RT_wt[...,1:]) * loftr_preds_6d[...,3:-1]
            pose_preds = torch.cat([pred_T, pred_R], dim=-1)
        
        else:
            pose_preds = self.pose_regressor(features.reshape([B, -1]))
        
        rot_preds_6d = pose_preds[:, 3:]
        tran_preds = pose_preds[:, :3]
        if self.use_normalized_6d:
            preds_6d_unnorm = rot_preds_6d*self.global_pose_std[3:]+self.global_pose_mean[3:]
            tran_preds_unnorm = tran_preds * self.global_pose_std[:3] + self.global_pose_mean[:3]
        else:
            preds_6d_unnorm = rot_preds_6d
            tran_preds_unnorm = tran_preds
        
        rot_preds_mtx = compute_rotation_matrix_from_ortho6d(preds_6d_unnorm)
        rot_preds = compute_pose_from_rotation_matrix(self.T_pose, rot_preds_mtx)
        
        return tran_preds_unnorm, rot_preds, rot_preds_mtx, rot_preds_6d
