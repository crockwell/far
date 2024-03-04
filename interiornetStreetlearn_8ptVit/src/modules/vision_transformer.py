""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2021 Ross Wightman
"""
import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_helpers import build_model_with_cfg, named_apply
from .vit_layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


def get_l1_positional_encodings(B, N, intrinsics=None):

    h,w = 48,64
    if N == 24*24:
        h,w = 24,24
    elif N != 48*64:
        print('unexpected resolution for positional encoding')
        assert(False)

    positional = torch.ones([B, N, 6])

    ys = torch.linspace(-1,1,steps=h)
    xs = torch.linspace(-1,1,steps=w)
    p3 = ys.unsqueeze(0).repeat(B,w)
    p4 = xs.repeat_interleave(h).unsqueeze(0).repeat(B,1)

    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics[:,0].unbind(dim=-1)

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
                p3[:, int(k * w + j)] = w2.squeeze() / w3.squeeze() 
                p4[:, int(k * w + j)] = w1.squeeze() / w3.squeeze() 
        
        
    #p2 = p3 * p4
    #p1 = p4 * p4
    #p0 = p3 * p3
    positional[:,:,3:5] = torch.stack([p3,p4],dim=2)

    return positional


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
    h,w = 48,64
    if N == 24*24:
        h,w = 24,24
    elif N != 48*64:
        print('unexpected resolution for positional encoding')
        assert(False)

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
                p3[:, int(k * w + j)] = w2.squeeze() / w3.squeeze() 
                p4[:, int(k * w + j)] = w1.squeeze() / w3.squeeze() 

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
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., 
                proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_fundamental = nn.Linear(dim+int(6*self.num_heads), dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2, camera=None, intrinsics=None):
        B, N, C = x1.shape

        qkv1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]   # make torchscript happy (cannot use tensor as tuple)

        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]   # make torchscript happy (cannot use tensor as tuple)

        attn_1 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn_2 = (q1 @ k2.transpose(-2, -1)) * self.scale

        attn_fundamental_1 = attn_1.softmax(dim=-1) * attn_1.softmax(dim=-2)
        attn_fundamental_2 = attn_2.softmax(dim=-1) * attn_2.softmax(dim=-2)

        positional = get_positional_encodings(B, N, intrinsics=intrinsics).cuda() # shape B,N,6
        v1 = torch.cat([v1,positional.unsqueeze(1).repeat(1,self.num_heads,1,1)],dim=3)
        v2 = torch.cat([v2,positional.unsqueeze(1).repeat(1,self.num_heads,1,1)],dim=3)
        
        fundamental_1 = (v1.transpose(-2, -1) @ attn_fundamental_1) @ v1
        fundamental_2 = (v2.transpose(-2, -1) @ attn_fundamental_2) @ v2

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
                 drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, camera=None, intrinsics=None):
        b_s, h_w, nf = x.shape
        x = x.reshape([-1, 2, h_w, nf])
        x1_in = x[:,0]
        x2_in = x[:,1]

        fundamental1, fundamental2 = self.cross_attn(self.norm1(x1_in), self.norm1(x2_in), camera, intrinsics=intrinsics)
        fundamental_inter = torch.cat([fundamental1.unsqueeze(1), fundamental2.unsqueeze(1)], dim=1)
        fundamental = fundamental_inter.reshape(b_s, -1, nf)
        fundamental = fundamental + self.drop_path(self.mlp(self.norm2(fundamental)))
        return fundamental

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim_in = dim

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_in, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, camera=None, intrinsics=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)           

    def forward(self, x, camera=None, intrinsics=None):
        identity = x 
        out = x
        x = identity + self.drop_path(self.attn(self.norm1(out), camera, intrinsics=intrinsics))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_list = []
        for i in range(depth):
            if i == depth - 1:
                this_block = CrossBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                            drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            else:
                this_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                            drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            block_list.append(this_block)
        self.blocks = nn.Sequential(*block_list)
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

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

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
}

def _create_vision_transformer(variant, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)

    model = build_model_with_cfg(
        VisionTransformer, variant,
        default_cfg=default_cfg,
        **kwargs)
    return model