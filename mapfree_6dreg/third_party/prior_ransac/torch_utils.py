import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy
import torch
from typing import Tuple, Optional
from linalg import transform_points
from torch_version import torch_version_ge
Tensor = torch.Tensor

def _torch_svd_cast(input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
    if torch_version_ge(1, 11):
        out3 = out3H.mH
    else:
        out3 = out3H.transpose(-1, -2)
    return (out1.to(input.dtype), out2.to(input.dtype), out3.to(input.dtype))


def cross_product_matrix(x: torch.Tensor) -> torch.Tensor:
    r"""Return the cross_product_matrix symmetric matrix of a vector.

    Args:
        x: The input vector to construct the matrix in the shape :math:`(*, 3)`.

    Returns:
        The constructed cross_product_matrix symmetric matrix with shape :math:`(*, 3, 3)`.
    """
    if not x.shape[-1] == 3:
        raise AssertionError(x.shape)
    # get vector compononens
    x0 = x[..., 0]
    x1 = x[..., 1]
    x2 = x[..., 2]

    # construct the matrix, reshape to 3x3 and return
    zeros = torch.zeros_like(x0)
    cross_product_matrix_flat = torch.stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros], dim=-1)
    shape_ = x.shape[:-1] + (3, 3)
    return cross_product_matrix_flat.view(*shape_)
