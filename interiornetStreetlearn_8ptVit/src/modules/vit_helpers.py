""" Model creation / weight loading / state_dict helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import collections.abc
from itertools import repeat
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def default_cfg_for_features(default_cfg):
    default_cfg = deepcopy(default_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'crop_pct', 'classifier', 'global_pool')  # add default final pool size?
    for tr in to_remove:
        default_cfg.pop(tr, None)
    return default_cfg


def overlay_external_default_cfg(default_cfg, kwargs):
    """ Overlay 'external_default_cfg' in kwargs on top of default_cfg arg.
    """
    external_default_cfg = kwargs.pop('external_default_cfg', None)
    if external_default_cfg:
        default_cfg.pop('url', None)  # url should come from external cfg
        default_cfg.pop('hf_hub', None)  # hf hub id should come from external cfg
        default_cfg.update(external_default_cfg)


def set_default_kwargs(kwargs, names, default_cfg):
    for n in names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # default_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = default_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = default_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, default_cfg[n])


def filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model

    FIXME this sequence of overlay default_cfg, set default kwargs, filter kwargs
    could/should be replaced by an improved configuration mechanism

    Args:
        default_cfg: input default_cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Overlay default cfg values from `external_default_cfg` if it exists in kwargs
    overlay_external_default_cfg(default_cfg, kwargs)
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if default_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)
    set_default_kwargs(kwargs, names=default_kwarg_names, default_cfg=default_cfg)
    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    filter_kwargs(kwargs, names=kwargs_filter)

def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        default_cfg: dict,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[dict] = None,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        default_cfg (dict): model's default pretrained/task config
        model_cfg (Optional[Dict]): model's architecture config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    features = False
    feature_cfg = feature_cfg or {}
    default_cfg = deepcopy(default_cfg) if default_cfg else {}
    update_default_cfg_and_kwargs(default_cfg, kwargs, kwargs_filter)
    default_cfg.setdefault('architecture', variant)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    # Build the model
    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = default_cfg

    return model