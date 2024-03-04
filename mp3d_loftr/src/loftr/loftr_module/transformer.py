import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
from functools import partial

from .vit_layers import Mlp, DropPath, trunc_normal_, lecun_normal_
import collections.abc
from itertools import repeat
from src.losses.loftr_loss import pose_mean_6d, pose_std_6d

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear',
                 use_num_corres=False):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        if attention == 'linear':
            self.attention = LinearAttention(use_num_corres=use_num_corres) 
        else: 
            self.attention = FullAttention(use_num_corres=use_num_corres)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None, loftr_preds=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask, loftr_preds=loftr_preds)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        if 'regress_use_num_corres' not in config:
            config['regress_use_num_corres'] = False
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'], config['regress_use_num_corres'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, loftr_preds=None, inv_loftr_preds=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                # doesn't make a ton of sense to care about camera for SA, but can't hurt?
                feat0 = layer(feat0, feat0, mask0, mask0, inv_loftr_preds)
                feat1 = layer(feat1, feat1, mask1, mask1, loftr_preds)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1, inv_loftr_preds)
                feat1 = layer(feat1, feat0, mask1, mask0, loftr_preds)
            else:
                raise KeyError

        return feat0, feat1

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def get_positional_encodings(B, N, intrinsics=None):
    '''
    # we now append a positional encoding onto v
    # of dim 6 (x^2, y^2, xy, x, y, 1)
    # this way, we can model linear & non-linear
    # relations between height & width. 
    # we multiply attention by this encoding on both sides
    # the results correspond to the variables in UTU
    # from the fundamental matrix
    # so, v is of shape B, N, C + 6
    '''
    h,w = 60,80
    intrinsics = torch.tensor([[[517/9,517/8,40,30],
                              [517/9,517/8,40,30]]]).cuda()
    positional = torch.ones([B, N, 6])

    ys = torch.linspace(-1,1,steps=h)
    xs = torch.linspace(-1,1,steps=w)
    p3 = ys.unsqueeze(0).repeat(B,w)
    p4 = xs.repeat_interleave(h).unsqueeze(0).repeat(B,1)

    if intrinsics is not None:
        # make sure not changing over frames
        assert(torch.all(intrinsics[:,0]==intrinsics[:,1]).cpu().numpy().item())

        '''
        use [x'/w', y'/w'] instead of x,y for coords. Where [x',y',w'] = K^{-1} [x,y,1]
        '''
        fx, fy, cx, cy = intrinsics[:,0].unbind(dim=-1)

        if cx[0] * cy[0] == 0:
            print('principal point is in upper left, not setup for this right now.')
            import pdb; pdb.set_trace()

        hpix = cy * 2
        wpix = cx * 2
        # map to between -1 and 1
        fx_normalized = (fx / wpix) * 2
        cx_normalized = (cx / wpix) * 2 - 1 
        fy_normalized = (fy / hpix) * 2
        cy_normalized = (cy / hpix) * 2 - 1
        # in fixed case, if we are mapping rectangular img with width > height,
        # then fy will be > fx and therefore p3 will be both greater than -1 and less than 1. ("y is zoomed out")
        # p4 will be -1 to 1.

        K = torch.zeros([B,3,3])
        K[:,0,0] = fx_normalized
        K[:,1,1] = fy_normalized
        K[:,0,2] = cx_normalized
        K[:,1,2] = cy_normalized
        K[:,2,2] = 1
    
        Kinv = torch.inverse(K)
        for j in range(h):
            for k in range(w):
                w1, w2, w3 = torch.split(Kinv @ torch.tensor([xs[k], ys[j], 1]), 1, dim=1)
                p3[:, int(j * w + k)] = w2.squeeze() / w3.squeeze() 
                p4[:, int(j * w + k)] = w1.squeeze() / w3.squeeze() 
    

    p2 = p3 * p4
    p1 = p4 * p4
    p0 = p3 * p3
    positional[:,:,:5] = torch.stack([p0,p1,p2,p3,p4],dim=2)

    return positional

class CrossAttention(nn.Module):
    """
    Our custom Cross-Attention Block. Have options to use dual softmax, 
    add positional encoding and use bilinear attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_fundamental = nn.Linear(dim+int(6*self.num_heads), dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2, intrinsics=None, loftr_preds=None, inv_loftr_preds=None):
        B, N, C = x1.shape

        qkv1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]   # make torchscript happy (cannot use tensor as tuple)

        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]   # make torchscript happy (cannot use tensor as tuple)

        attn_1 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn_2 = (q1 @ k2.transpose(-2, -1)) * self.scale

        attn_1_out = attn_1
        attn_2_out = attn_2

        attn_fundamental_1 = attn_1_out.softmax(dim=-1) * attn_1_out.softmax(dim=-2)
        attn_fundamental_2 = attn_2_out.softmax(dim=-1) * attn_2_out.softmax(dim=-2)

        positional = get_positional_encodings(B, N, intrinsics=intrinsics).cuda() # shape B,N,6
        v1 = torch.cat([v1,positional.unsqueeze(1).repeat(1,self.num_heads,1,1)],dim=3)
        v2 = torch.cat([v2,positional.unsqueeze(1).repeat(1,self.num_heads,1,1)],dim=3)

        v1_out = v1
        v2_out = v2
        
        fundamental_1 = (v1_out.transpose(-2, -1) @ attn_fundamental_1) @ v1_out
        fundamental_2 = (v2_out.transpose(-2, -1) @ attn_fundamental_2) @ v2_out

        fundamental_1 = fundamental_1.reshape(B, int(C+6*self.num_heads), int((C+6*self.num_heads)/self.num_heads)).transpose(-2,-1)           
        fundamental_2 = fundamental_2.reshape(B, int(C+6*self.num_heads), int((C+6*self.num_heads)/self.num_heads)).transpose(-2,-1)
        # fundamental is C/3+6,C/3+6 (for each head)

        fundamental_2 = self.proj_fundamental(fundamental_2)
        fundamental_1 = self.proj_fundamental(fundamental_1)

        # we flip these: we want x1 to be (q1 @ k2) @ v2
        # impl is similar to ViLBERT
        return fundamental_2, fundamental_1
        
class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 use_pos_embedding=False, distilled=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.patch_embed = nn.Identity()
        self.h, self.w = 60, 80
        num_patches = self.h * self.w
        embed_dim = 256
        self.num_tokens = 0 

        self.pos_embed = 0
        if use_pos_embedding:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.init_weights(use_pos_embedding=use_pos_embedding)

    def init_weights(self, use_pos_embedding=False):
        if use_pos_embedding:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(_init_vit_weights)

    def forward(self, x, intrinsics=None, loftr_preds=None, inv_loftr_preds=None):
        b_s, h_w, nf = x.shape
        x = x + self.pos_embed

        x = x.reshape([-1, 2, h_w, nf])
        x1_in = x[:,0]
        x2_in = x[:,1]

        fundamental1, fundamental2 = self.cross_attn(self.norm1(x1_in), self.norm1(x2_in), intrinsics=intrinsics, 
                                                     loftr_preds=loftr_preds, inv_loftr_preds=inv_loftr_preds)
        fundamental_inter = torch.cat([fundamental1.unsqueeze(1), fundamental2.unsqueeze(1)], dim=1)
        fundamental = fundamental_inter.reshape(b_s, -1, nf)
        fundamental = fundamental + self.drop_path(self.mlp(self.norm2(fundamental)))
        return fundamental
        
class LocalFeatureTransformerRegressor(nn.Module):
    """LoFTR network + EMM head."""

    def __init__(self, config):
        super(LocalFeatureTransformerRegressor, self).__init__()
        self.config = config

        num_heads = 4
        feat_size = 256
        pos_enc = 6

        pose_size_in = 9
        pose_size = 9
            
        self.pose_size = pose_size

        if self.config['regress']['regress_use_num_corres']:
            pose_size_in += 1
        if self.config['use_many_ransac_thr']:
            pose_size_in += 3

        self.pose_size_in = pose_size_in

        print("using num correspondence?", str(pose_size_in-9))

        self.H = int(num_heads*2*(feat_size//num_heads + pos_enc) * (feat_size//num_heads))
        self.H2 = 512
        if config['regress']['use_simple_moe']:
            self.encoder = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
            )

            if config['regress']['use_1wt']:
                local_pose_size = 1
            elif config['regress']['use_2wt']:
                local_pose_size = 2
            else:
                local_pose_size = pose_size

            self.moe_predictor = nn.Sequential(
                nn.Linear(self.H+pose_size+pose_size_in, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(),
                nn.Linear(self.H2, local_pose_size),
                nn.Sigmoid(),
            )
            self.pose_regressor_simple_moe = nn.Sequential(
                nn.Linear(self.H2, self.H2),
                nn.ReLU(),
                nn.Linear(self.H2, pose_size),
            )

        else:        
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, pose_size)
            )
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(feat_size)

        self.emm = CrossBlock(dim=feat_size, num_heads=num_heads, qkv_bias=True,
                            use_pos_embedding=config['regress']['use_pos_embedding'])

        if config['regress_loftr_layers'] > 0:
            self.loftr = LocalFeatureTransformer(config['regress'])

    def forward_emm(self, feat0, feat1, loftr_preds=None, inv_loftr_preds=None):
        B, _, _ = feat0.shape
        x = torch.cat([feat0, feat1], dim=0)
        x = self.emm(x, loftr_preds=loftr_preds, inv_loftr_preds=inv_loftr_preds)
        features = self.norm(x).reshape([B, -1])

        if self.config['regress']['use_simple_moe']:
            feats = self.encoder(features)
            pred_reg_6d = self.pose_regressor_simple_moe(feats)

            pred_reg_t_in = pred_reg_6d[...,:3]
            loftr_pred_t_in = loftr_preds[...,:3]

            if self.config['regress']['scale_8pt']:
                pred_reg_t = pred_reg_t_in

                # unnormalize loftr_t by mean and stdev
                loftr_preds_unnorm = loftr_pred_t_in * pose_std_6d[:3].cuda() + pose_mean_6d[:3].cuda()
                pred_reg_t_unnorm = pred_reg_t_in * pose_std_6d[:3].cuda() + pose_mean_6d[:3].cuda()

                loftr_pred_t_unnorm = loftr_preds_unnorm[...,:3] * torch.linalg.norm(pred_reg_t_unnorm, dim=-1) / torch.clamp(torch.linalg.norm(loftr_preds_unnorm[...,:3], dim=-1), 1e-3, 100)
                
                # normalize loftr_t by mean and stdev
                loftr_pred_t = (loftr_pred_t_unnorm - pose_mean_6d[:3].cuda()) / pose_std_6d[:3].cuda()

            else:
                loftr_pred_t = loftr_pred_t_in
                pred_reg_t = pred_reg_t_in

            if self.pose_size_in > 0:
                loftr_pred_R = loftr_preds[...,3:-(self.pose_size_in-self.pose_size)]
            else:
                loftr_pred_R = loftr_preds[...,3:]

            feats_preds = torch.cat([features, pred_reg_6d, loftr_preds],dim=-1)
            pred_RT_wt = self.moe_predictor(feats_preds)
            if self.config['regress']['use_2wt']:
                if self.config['regress']['use_5050_weight']:
                    print('using 5050weight')
                    pred_T = (pred_reg_t + loftr_pred_t) * 0.5
                    pred_R = (pred_reg_6d[...,3:] + loftr_pred_R) * 0.5
                    import pdb; pdb.set_trace()
                else:
                    pred_T = pred_RT_wt[...,0] * pred_reg_t + (1-pred_RT_wt[...,0]) * loftr_pred_t
                    pred_R = pred_RT_wt[...,1] * pred_reg_6d[...,3:] + (1-pred_RT_wt[...,1]) * loftr_pred_R
                
                pose_preds = torch.cat([pred_T, pred_R], dim=-1)
            else:
                pred_T = pred_RT_wt[...,0] * pred_reg_t + (1-pred_RT_wt[...,0]) * loftr_pred_t
                pred_R = pred_RT_wt[...,0] * pred_reg_6d[...,3:] + (1-pred_RT_wt[...,0]) * loftr_pred_R
                pose_preds = torch.cat([pred_T, pred_R], dim=-1)
        else:
            pose_preds = self.pose_regressor(features)
            pred_RT_wt = None

        if self.config['regress']['save_mlp_feats']:
            return_features = features
        else:
            return_features = None

        return pose_preds, return_features, pred_RT_wt

    def forward(self, feat0, feat1, loftr_preds=None, inv_loftr_preds=None, mask0=None, mask1=None, F=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        if self.config['regress_loftr_layers'] > 0:
            feat0, feat1 = self.loftr(feat0, feat1, loftr_preds=loftr_preds, inv_loftr_preds=inv_loftr_preds)

        x, mlp_features, pred_RT_wt = self.forward_emm(feat0, feat1, loftr_preds, inv_loftr_preds)

        return x, mlp_features, pred_RT_wt