"""Module containing RANSAC modules.
Borrows heavily from Kornia https://github.com/kornia/kornia"""
import math
from typing import Callable, Optional, Tuple, Dict, Any
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)

import numpy as np
import torch
import kornia.geometry.epipolar as epi
from kornia.core import Device, Module, Tensor, zeros
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.geometry import (
    find_fundamental,
    find_homography_dlt,
    find_homography_dlt_iterated,
    find_homography_lines_dlt,
    find_homography_lines_dlt_iterated,
    symmetrical_epipolar_distance,
    epipolar
)
from kornia.geometry.homography import (
    line_segment_transfer_error_one_way,
    oneway_transfer_error,
    sample_is_valid_for_homography,
)
import essential as essential_utils
import utils3d
from cv_geometry import run_8point, run_5point_our_kornia, run_5point_cv2

def get_RT_from_fundamental(F, K1, K2):
    '''
        F : B x 3 x 3
        K1 : 3 x 3
        K2 : 3 x 3
    '''
    K1 = K1[None].repeat(len(F), 1, 1)
    K2 = K2[None].repeat(len(F), 1, 1)
    E = essential_utils.essential_from_fundamental(F, K1 , K2)
    R1, R2, T = essential_utils.decompose_essential_matrix(E)
    return R1, R2, T

def get_RT_from_essential(E):
    '''
        F : B x 3 x 3
        K1 : 3 x 3
        K2 : 3 x 3
    '''
    R1, R2, T = essential_utils.decompose_essential_matrix(E)
    return R1, R2, T
    
def essential_from_RT(RT, K1, K2):
    RT1 = torch.eye(RT)
    R1, T1 = RT1[:3,:3], RT1[:3,3:]
    R2, T2 = RT[:3,:3], RT[:3,3:]
    E = epipolar.essential_from_Rt(R1, T1, R2, T2)
    return E

def fundamental_from_RT(RT, K1, K2):
    RT1 = torch.eye(4).to(RT)
    R1, T1 = RT1[:3,:3], RT1[:3,3:]
    R2, T2 = RT[:3,:3], RT[:3,3:]
    E = epipolar.essential_from_Rt(R1, T1, R2, T2)
    K1_inv = torch.inverse(K1)
    K2_inv = torch.inverse(K2)
    F = torch.matmul(K1_inv, torch.matmul(E, K2_inv))
    return E


class RANSAC(Module):
    """Module for robust geometry estimation with RANSAC. https://en.wikipedia.org/wiki/Random_sample_consensus.

    Args:
        model_type: type of model to estimate, e.g. "homography" or "fundamental".
        inliers_threshold: threshold for the correspondence to be an inlier.
        batch_size: number of generated samples at once.
        max_iterations: maximum batches to generate. Actual number of models to try is ``batch_size * max_iterations``.
        confidence: desired confidence of the result, used for the early stopping.
        max_local_iterations: number of local optimization (polishing) iterations.
    """

    def __init__(
        self,
        model_type: str = 'homography',
        inl_th: float = 2.0,
        batch_size: int = 2048,
        max_iter: int = 10,
        confidence: float = 0.99,
        max_lo_iters: int = 5,
        prior_params: Dict[str, Any] = {},
        use_noexp_prior_scoring: bool = False,
        use_linear_bias_sampling: bool = False,
        bias_sigma_sq: float = 1.0,
        compute_stopping_inlier_only: bool = False,
        perform_early_stopping: bool = False,
        l1_dist: bool = False,
        use_epipolar_error: bool = False,
        K: Optional[Tensor] = None,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.supported_models = ['homography', 'fundamental', 'homography_from_linesegments']
        self.inl_th = inl_th
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model_type = model_type
        self.confidence = confidence
        self.max_lo_iters = max_lo_iters
        self.use_noexp_prior_scoring = use_noexp_prior_scoring
        self.use_linear_bias_sampling = use_linear_bias_sampling
        self.bias_sigma_sq = bias_sigma_sq
        self.compute_stopping_inlier_only = compute_stopping_inlier_only
        self.perform_early_stopping = perform_early_stopping
        self.l1_dist = l1_dist
        self.use_epipolar_error = use_epipolar_error
        self.K = K
        self.normalize = normalize

        self.error_fn: Callable[..., Tensor]
        self.minimal_solver: Callable[..., Tensor]
        self.polisher_solver: Callable[..., Tensor]
        self.prior_params = prior_params

        self.setup_prior(self.prior_params)

        if model_type == 'homography':
            self.error_fn = oneway_transfer_error
            self.minimal_solver = find_homography_dlt
            self.polisher_solver = find_homography_dlt_iterated
            self.minimal_sample_size = 4
        elif model_type == 'homography_from_linesegments':
            self.error_fn = line_segment_transfer_error_one_way
            self.minimal_solver = find_homography_lines_dlt
            self.polisher_solver = find_homography_lines_dlt_iterated
            self.minimal_sample_size = 4
        elif model_type == 'fundamental':
            self.error_fn = symmetrical_epipolar_distance
            self.minimal_solver = run_8point
            self.minimal_sample_size = 8
            # https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fundam.cpp#L498
            self.polisher_solver = find_fundamental
        elif model_type == 'essential':
            self.error_fn = epi.sampson_epipolar_distance
            self.minimal_solver = run_5point_our_kornia
            self.minimal_sample_size = 5
            self.polisher_solver = find_fundamental
        elif model_type == 'essential_cv2':
            if self.use_epipolar_error:
                self.error_fn = epi.symmetrical_epipolar_distance
            else:
                self.error_fn = epi.sampson_epipolar_distance
            self.minimal_solver = run_5point_cv2
            self.minimal_sample_size = 6
        else:
            raise NotImplementedError(f"{model_type} is unknown. Try one of {self.supported_models}")

    def sample(self, sample_size: int, pop_size: int, batch_size: int, weight: torch.Tensor, device: Device = torch.device('cpu')) -> Tensor:
        """Minimal sampler, but unlike traditional RANSAC we sample in batches to get benefit of the parallel
        processing, esp. on GPU.
        """
        if weight is not None:
            weight = weight.cpu().numpy()
            weight = weight + 1E-4
            weight = weight/weight.sum()
            out = np.random.choice(pop_size, (batch_size, sample_size), replace=True, p=weight)
            out = torch.LongTensor(out).to(device)
            ## this does not ensure in a given batch have unique indexes
        else:
            rand = torch.rand(batch_size, pop_size, device=device)
            _, out = rand.topk(k=sample_size, dim=1)
        return out
    
    
    def setup_prior(self, prior_params):
        if prior_params: 
            self.use_prior = True
            self.prior_lambda = prior_params['lambda']
            # normalize point cloud
            self.prior_params['RT'][:,3] /= torch.linalg.norm(prior_params['RT'][:,3])
        else:
            self.use_prior = False
            self.prior_lambda = 1.0
        return

    @staticmethod
    def compute_RT_error(RT, pcl, target_pcl):
        rt_pcl = utils3d.transform_points(pcl.permute(0, 2,1), RT)
        error = torch.abs(rt_pcl - target_pcl).reshape(len(RT), -1).mean(1)
        return error
    
    @staticmethod
    def get_RT(R, T):
        RT = torch.zeros(R.shape[0], 4, 4)
        RT[:,:3,:3] = R
        RT[:,:3, 3:] = T
        RT[:, 3, 3] = 1
        return RT.to(R.device)
    
    def get_prior_estimate(self, model):
        if self.use_prior:
            K1 = self.prior_params['K1']
            K2 = self.prior_params['K2']
            prior_RT = self.prior_params['RT']
            random_pcl = self.prior_params['pcl']
            if self.model_type == 'fundamental':
                R1, R2, T = get_RT_from_fundamental(model, K1, K2)
            elif self.model_type == 'essential' or self.model_type == 'essential_cv2':
                R1, R2, T = get_RT_from_essential(model)
            else:
                assert False, 'what is wrong?'
            RT1 = self.get_RT(R1, T)
            RT2 = self.get_RT(R2, T)
            bsize = RT1.shape[0]

            #import pdb; pdb.set_trace()
            if self.normalize:
                prior_RT[...,3] = prior_RT[...,3]/prior_RT[...,3].norm(dim=-1, keepdim=True)

            target_pcl = utils3d.transform_points(random_pcl.permute(1,0)[None], prior_RT[None],)
            random_pcl = random_pcl[None].repeat(bsize, 1, 1)
            
            error1 = self.compute_RT_error(RT1, random_pcl, target_pcl)
            error2 = self.compute_RT_error(RT2, random_pcl, target_pcl)
            error = torch.min(error1, error2)
            return error
        else:
            return 0


    @staticmethod
    def max_samples_by_conf(n_inl: int, num_tc: int, sample_size: int, conf: float) -> float:
        """Formula to update max_iter in order to stop iterations earlier
        https://en.wikipedia.org/wiki/Random_sample_consensus."""
        if n_inl == num_tc:
            return 1.0
        
        n_inl = min(num_tc-1, n_inl)
        denom = math.log(1.0 - math.pow(n_inl / num_tc, sample_size) + 1E-4)
        if denom == 0:
            return math.log(1.0 - conf)
        try:
            realval = math.log(1.0 - conf) / denom
        except:
            import pdb; pdb.set_trace()
        return realval

    def estimate_model_from_minsample(self, kp1: Tensor, kp2: Tensor) -> Tensor:
        batch_size, sample_size = kp1.shape[:2]
        H = self.minimal_solver(kp1, kp2, torch.ones(batch_size, sample_size, dtype=kp1.dtype, device=kp1.device))
        return H

    def verify(self, kp1: Tensor, kp2: Tensor, models: Tensor, inl_th: float, prior_score: Tensor) -> Tuple[Tensor, Tensor, float]:
        if len(kp1.shape) == 2:
            kp1 = kp1[None]
        if len(kp2.shape) == 2:
            kp2 = kp2[None]
        batch_size = models.shape[0]

        if self.l1_dist:
            squared = False
        else:
            squared = True
        
        models_in = models

        if self.model_type == 'homography_from_linesegments':
            errors = self.error_fn(kp1.expand(batch_size, -1, 2, 2), kp2.expand(batch_size, -1, 2, 2), models_in)
        else:
            errors = self.error_fn(kp1.expand(batch_size, -1, 2), kp2.expand(batch_size, -1, 2), models_in, squared=squared)
        
        inl = errors <= inl_th 
        models_score = inl.to(kp1).sum(dim=1)
        models_score_plus = models_score + prior_score.to(kp1)
        best_model_idx = models_score_plus.argmax()
        best_model_score = models_score_plus[best_model_idx].item()
        best_model_score_inlier = models_score[best_model_idx].item()
        model_best = models[best_model_idx].clone()
        inliers_best = inl[best_model_idx]

        inl_tight = errors <= inl_th / 10.0
        inl_ultra_tight = errors <= inl_th / 100.0
        inliers_best_tight = inl_tight[best_model_idx]
        inliers_best_ultra_tight = inl_ultra_tight[best_model_idx]

        if self.compute_stopping_inlier_only:
            return model_best, inliers_best, best_model_score, inliers_best_tight, inliers_best_ultra_tight, best_model_score_inlier
        else:
            return model_best, inliers_best, best_model_score, inliers_best_tight, inliers_best_ultra_tight

    def remove_bad_samples(self, kp1: Tensor, kp2: Tensor) -> Tuple[Tensor, Tensor]:
        """"""
        # ToDo: add (model-specific) verification of the samples,
        # E.g. constraints on not to be a degenerate sample
        if self.model_type == 'homography':
            mask = sample_is_valid_for_homography(kp1, kp2)
            return kp1[mask], kp2[mask]
        return kp1, kp2

    def remove_bad_models(self, models: Tensor) -> Tensor:
        # ToDo: add more and better degenerate model rejection
        # For now it is simple and hardcoded
        main_diagonal = torch.diagonal(models, dim1=1, dim2=2)
        mask = main_diagonal.abs().min(dim=1)[0] > 1e-4
        return models[mask]

    def polish_model(self, kp1: Tensor, kp2: Tensor, inliers: Tensor) -> Tensor:
        # TODO: Replace this with MAGSAC++ polisher
        kp1_inl = kp1[inliers][None]
        kp2_inl = kp2[inliers][None]
        num_inl = kp1_inl.size(1)
        model = self.polisher_solver(
            kp1_inl, kp2_inl, torch.ones(1, num_inl, dtype=kp1_inl.dtype, device=kp1_inl.device)
        )
        return model

    def validate_inputs(self, kp1: Tensor, kp2: Tensor, weights: Optional[Tensor] = None) -> None:
        if self.model_type in ['homography', 'fundamental']:
            KORNIA_CHECK_SHAPE(kp1, ["N", "2"])
            KORNIA_CHECK_SHAPE(kp2, ["N", "2"])
            if not (kp1.shape[0] == kp2.shape[0]) or (kp1.shape[0] < self.minimal_sample_size):
                raise ValueError(
                    f"kp1 and kp2 should be \
                                 equal shape at at least [{self.minimal_sample_size}, 2], \
                                 got {kp1.shape}, {kp2.shape}"
                )
        if self.model_type == 'homography_from_linesegments':
            KORNIA_CHECK_SHAPE(kp1, ["N", "2", "2"])
            KORNIA_CHECK_SHAPE(kp2, ["N", "2", "2"])
            if not (kp1.shape[0] == kp2.shape[0]) or (kp1.shape[0] < self.minimal_sample_size):
                raise ValueError(
                    f"kp1 and kp2 should be \
                                 equal shape at at least [{self.minimal_sample_size}, 2, 2], \
                                 got {kp1.shape}, {kp2.shape}"
                )

    def forward(self, kp1: Tensor, kp2: Tensor, weights: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Main forward method to execute the RANSAC algorithm.

        Args:
            kp1: source image keypoints :math:`(N, 2)`.
            kp2: distance image keypoints :math:`(N, 2)`.
            weights: optional correspondences weights. Not used now.

        Returns:
            - Estimated model, shape of :math:`(1, 3, 3)`.
            - The inlier/outlier mask, shape of :math:`(1, N)`, where N is number of input correspondences.
        """
        self.validate_inputs(kp1, kp2, weights)
        best_score_total: float = float(self.minimal_sample_size)
        num_tc: int = len(kp1)
        best_model_total = zeros(3, 3, dtype=kp1.dtype, device=kp1.device)
        inliers_best_total: Tensor = zeros(num_tc, 1, device=kp1.device, dtype=torch.bool)
       
        if self.use_prior:
           
            F = fundamental_from_RT(self.prior_params['RT'], 
                                    self.prior_params['K1'],
                                    self.prior_params['K2'])
           
            sampson_errors = symmetrical_epipolar_distance(kp1[None,], kp2[None], F[None])
            if self.use_linear_bias_sampling:
                bias_weight = torch.exp(-sampson_errors/self.bias_sigma_sq)
                bias_weight = bias_weight.squeeze()
            else:
                bias_weight = torch.exp(-sampson_errors**2)
                bias_weight = bias_weight/(1E-4 + bias_weight.sum())
                bias_weight = bias_weight.squeeze()


        for i in range(self.max_iter):
            # print(i)
            # Sample minimal samples in batch to estimate models
            if (i % 2 == 0 and self.use_prior) and self.prior_params['biased_sampling']:
                idxs =  self.sample(self.minimal_sample_size, num_tc, self.batch_size, weight=bias_weight,  device=kp1.device,)
            else:
                idxs = self.sample(self.minimal_sample_size, num_tc, self.batch_size,  weight=None, device=kp1.device,)
            kp1_sampled = kp1[idxs]
            kp2_sampled = kp2[idxs]
            
            kp1_sampled, kp2_sampled = self.remove_bad_samples(kp1_sampled, kp2_sampled)
            if len(kp1_sampled) == 0:
                continue
            
            # Estimate models
            models = self.estimate_model_from_minsample(kp1_sampled, kp2_sampled)
            models = self.remove_bad_models(models)
            if (models is None) or (len(models) == 0):
                continue

            if self.use_prior:
                prior_estimate = self.get_prior_estimate(models)
                if self.use_noexp_prior_scoring:
                    prior_estimate = -prior_estimate**2 / self.prior_lambda
                else:
                    prior_estimate = torch.exp(-prior_estimate/0.1) * self.prior_lambda
            else:
                prior_estimate = torch.zeros(len(models))    

            # Score the models and select the best one
            if self.compute_stopping_inlier_only:
                model, inliers, model_score, inliers_best_tight, inliers_best_ultra_tight, best_model_score_inlier = self.verify(kp1, kp2, models, self.inl_th, 
                                    prior_estimate)
            else:
                model, inliers, model_score, inliers_best_tight, inliers_best_ultra_tight = self.verify(kp1, kp2, models, self.inl_th, 
                                    prior_estimate)
            # Store far-the-best model and (optionally) do a local optimization
            if model_score > best_score_total:
                # Local optimization
                for lo_step in range(self.max_lo_iters):
                    model_lo = self.polish_model(kp1, kp2, inliers)
                    if (model_lo is None) or (len(model_lo) == 0):
                        continue
                    _, inliers_lo, score_lo = self.verify(kp1, kp2, model_lo, self.inl_th, )
                    # print (f"Orig score = {best_model_score}, LO score = {score_lo} TC={num_tc}")
                    if score_lo > model_score:
                        model = model_lo.clone()[0]
                        inliers = inliers_lo.clone()
                        model_score = score_lo
                    else:
                        break
                # Now storing the best model
                best_model_total = model.clone()
                inliers_best_total = inliers.clone()
                best_score_total = model_score
                
                # # Should we already stop?
                if self.perform_early_stopping:
                    if self.compute_stopping_inlier_only:
                        new_max_iter = int(self.max_samples_by_conf(int(best_model_score_inlier), num_tc, self.minimal_sample_size, self.confidence))
                    else:
                        new_max_iter = int(self.max_samples_by_conf(int(best_score_total), num_tc, self.minimal_sample_size, self.confidence))
                    print (f"New max_iter = {new_max_iter}")
                    # Stop estimation, if the model is very good
                    if (i + 1) * self.batch_size >= new_max_iter:
                        break
                
        # local optimization with all inliers for better precision
        return best_model_total, inliers_best_total, inliers_best_tight, inliers_best_ultra_tight