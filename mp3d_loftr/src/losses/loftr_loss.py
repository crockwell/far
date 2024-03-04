from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

pose_mean_6d = torch.tensor([-0.34898765,  0.17085525, -0.87944315, 0.50275223, 0.03533648, -0.18179045, -0.03533648, 0.98189617, 0.09313615])
pose_std_6d = torch.tensor([1.94014405, 0.36770130 , 1.88317520, 0.51837117, 0.12717603, 0.65426397, 0.12717603, 0.0188729, 0.09709263])

def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(pose_mtx):
    return pose_mtx[..., :2, :].clone().reshape(*pose_mtx.size()[:-2], 6)

def compute_normalized_6d(pose_mtx):
    pose_6d = matrix_to_rotation_6d(pose_mtx[...,:3,:3])
    tr = pose_mtx[...,:3,3]
    pose_6d_norm = (torch.cat([tr,pose_6d],dim=-1) - pose_mean_6d.to(device=pose_6d.device)) / pose_std_6d.to(device=pose_6d.device)
    
    return pose_6d_norm

class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['loftr']['loss']
        self.match_type = self.config['loftr']['match_coarse']['match_type']
        self.sparse_spvs = self.config['loftr']['match_coarse']['sparse_spvs']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                
                loss =  c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))
        
    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        #print(offset_l2)
        #import pdb; pdb.set_trace()
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        #import pdb; pdb.set_trace()
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def _torch_svd_cast(self, input):
        """Helper function to make torch.svd work with other than fp32/64.

        The function torch.svd is only implemented for fp32/64 which makes
        impossible to be used by fp16 or others. What this function does, is cast
        input data type to fp32, apply torch.svd, and cast back to the input dtype.

        NOTE: in torch 1.8.1 this function is recommended to use as torch.linalg.svd
        """
        # if not isinstance(input, torch.Tensor):
        #    raise AssertionError(f"Input must be torch.Tensor. Got: {type(input)}.")
        dtype = input.dtype
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float32

        out1, out2, out3H = torch.linalg.svd(input.to(dtype))
        #if torch_version_ge(1, 11):
        #    out3 = out3H.mH
        #else:
        out3 = out3H.transpose(-1, -2)
        return (out1.to(input.dtype), out2.to(input.dtype), out3.to(input.dtype))

    def essential_to_essential_rotmtx(self, E_mat):
        r"""Decompose an essential matrix to possible rotations and translation.

        Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

        Returns:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`. such that is rotation vector
        """
        if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:]):
            raise AssertionError(E_mat.shape)

        # decompose matrix by its singular values
        U, _, V = self._torch_svd_cast(E_mat)
        Vt = V.transpose(-2, -1)

        mask = torch.ones_like(E_mat)
        mask[..., -1:] *= -1.0  # fill last column with negative values

        maskt = mask.transpose(-2, -1)

        # avoid singularities
        U = torch.where((torch.det(U) < 0.0)[..., None, None], U * mask, U)
        Vt = torch.where((torch.det(Vt) < 0.0)[..., None, None], Vt * maskt, Vt)

        return U @ Vt

    def compute_rt_loss(self, expec_rt, expec_rt_gt):
        '''
        expec_rt and expec_rt_gt are 3x4 rotation-translation matrices
        we convert them to normalized 6D representation and compute l2 loss
        normalization stats computed on val set

        at the moment, expec_rt_gt is shape 1,3,4 and expec_rt is shape 3,4.
        so we take only expec_rt_gt[0]
        '''
        
        if self.config['loftr']['regress_rt']:
            expec_rt_6d = expec_rt
            expec_rt_gt_6d = compute_normalized_6d(expec_rt_gt[0])
        else:
            expec_rt_6d = compute_normalized_6d(expec_rt)
            expec_rt_gt_6d = compute_normalized_6d(expec_rt_gt[0])

        if self.loss_config['use_l1_rt_loss']:
            power = 1
        else:
            power = 2

        optional_rt_tr_rot = None
        loss_tr = torch.pow(torch.abs(expec_rt_6d[:3]-expec_rt_gt_6d[:3]),power).mean()

        loss_rot = torch.pow(torch.abs(expec_rt_6d[3:]-expec_rt_gt_6d[3:]),power).mean()

        loss_tr = torch.clamp(loss_tr, 1e-8, 1e5)
        loss_rot = torch.clamp(loss_rot, 1e-8, 1e5)
        
        return loss_tr, loss_rot, optional_rt_tr_rot

    def compute_scale_loss(self, translation_scale, expec_rt_gt):
        scale_gt = torch.linalg.norm(expec_rt_gt[0][...,:3,3])

        scale_threshold = self.loss_config['max_scale_loss']

        power = 2
        loss_scale = torch.pow(translation_scale-scale_gt,power).mean()
        loss_scale_reported = min(loss_scale, torch.tensor(scale_threshold).to(device=translation_scale.device))
        if loss_scale_reported == torch.tensor(scale_threshold).to(device=translation_scale.device):
            # making backprop legit
            loss_scale *= 0

        return loss_scale, loss_scale_reported

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars, loss = {}, torch.tensor([0.0]).cuda()

        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)
        if self.config['loftr']['from_saved_preds'] is None and not self.config['use_correspondence_transformer']:
            # 1. coarse-level loss 
            loss_c = self.compute_coarse_loss(
                data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                    else data['conf_matrix'],
                data['conf_matrix_gt'],
                weight=c_weight)
            loss += loss_c * self.loss_config['coarse_weight']
            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
                
            # 2. fine-level loss
            loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
            if loss_f is not None:
                loss += loss_f * self.loss_config['fine_weight']
                loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
            else:
                assert self.training is False
                loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        # 3. RT loss (if applicable)
        if (self.loss_config['rt_weight_tr']+self.loss_config['rt_weight_rot']) > 0 and data['expec_rt'] is not None:
            loss_rt_tr, loss_rt_rot, optional_rt_tr_rot = self.compute_rt_loss(data['expec_rt'], data['T_0to1'][:,:3])
            loss += loss_rt_tr * self.loss_config['rt_weight_tr'] + loss_rt_rot * self.loss_config['rt_weight_rot']
            loss_scalars.update({"loss_rot": loss_rt_rot.clone().detach().cpu()})
            loss_scalars.update({"loss_tr": loss_rt_tr.clone().detach().cpu()})
            if optional_rt_tr_rot is not None:
                loss_scalars.update({"loss_tr_rot": optional_rt_tr_rot.clone().detach().cpu()})
        else:
            loss_scalars.update({'loss_rot': torch.tensor(100.)})
            loss_scalars.update({'loss_tr': torch.tensor(4.)})

        # 6 Scale Loss (if applicable)
        if self.config['loftr']['predict_translation_scale']:
            loss_scale, loss_scale_reported = self.compute_scale_loss(data['translation_scale'], data['T_0to1'][:,:3])
            loss += loss_scale * self.loss_config['scale_weight']
            loss_scalars.update({"loss_scale": loss_scale_reported.clone().detach().cpu()})

        if not torch.is_tensor(data['num_correspondences_after_ransac']):
            nc = torch.tensor(data['num_correspondences_after_ransac'], dtype=torch.float32).clone().detach().cpu()
        else:
            nc = data['num_correspondences_after_ransac'].clone().detach().cpu()

        if not torch.is_tensor(data['num_correspondences_before_ransac']):
            ncb = torch.tensor(data['num_correspondences_before_ransac'], dtype=torch.float32).clone().detach().cpu()
        else:
            ncb = data['num_correspondences_before_ransac'].clone().detach().cpu()

        loss_scalars.update({'num_correspondences_before_ransac': ncb, 'num_correspondences_after_ransac': nc})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
