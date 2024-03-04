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
from torch_utils import _torch_svd_cast
from torch_utils import cross_product_matrix
#import torch_utils

def essential_from_fundamental(F_mat: torch.Tensor, K1: torch.Tensor, K2: torch.Tensor) -> torch.Tensor:
    r"""Get Essential matrix from Fundamental and Camera matrices.

    Uses the method from Hartley/Zisserman 9.6 pag 257 (formula 9.12).

    Args:
        F_mat: The fundamental matrix with shape of :math:`(*, 3, 3)`.
        K1: The camera matrix from first camera with shape :math:`(*, 3, 3)`.
        K2: The camera matrix from second camera with shape :math:`(*, 3, 3)`.

    Returns:
        The essential matrix with shape :math:`(*, 3, 3)`.
    """
    if not (len(F_mat.shape) >= 2 and F_mat.shape[-2:] == (3, 3)):
        raise AssertionError(F_mat.shape)
    if not (len(K1.shape) >= 2 and K1.shape[-2:] == (3, 3)):
        raise AssertionError(K1.shape)
    if not (len(K2.shape) >= 2 and K2.shape[-2:] == (3, 3)):
        raise AssertionError(K2.shape)
    if not len(F_mat.shape[:-2]) == len(K1.shape[:-2]) == len(K2.shape[:-2]):
        raise AssertionError

    return K2.transpose(-2, -1) @ F_mat @ K1


def motion_from_essential(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Get Motion (R's and t's ) from Essential matrix.

    Computes and return four possible poses exist for the decomposition of the Essential
    matrix. The possible solutions are :math:`[R1,t], [R1,-t], [R2,t], [R2,-t]`.

    Args:
        E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
        The rotation and translation containing the four possible combination for the retrieved motion.
        The tuple is as following :math:`[(*, 4, 3, 3), (*, 4, 3, 1)]`.
    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:] == (3, 3)):
        raise AssertionError(E_mat.shape)

    # decompose the essential matrix by its possible poses
    R1, R2, t = decompose_essential_matrix(E_mat)

    # compbine and returns the four possible solutions
    Rs = torch.stack([R1, R1, R2, R2], dim=-3)
    Ts = torch.stack([t, -t, t, -t], dim=-3)

    return (Rs, Ts)


def essential_from_Rt(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    r"""Get the Essential matrix from Camera motion (Rs and ts).

    Reference: Hartley/Zisserman 9.6 pag 257 (formula 9.12)

    Args:
        R1: The first camera rotation matrix with shape :math:`(*, 3, 3)`.
        t1: The first camera translation vector with shape :math:`(*, 3, 1)`.
        R2: The second camera rotation matrix with shape :math:`(*, 3, 3)`.
        t2: The second camera translation vector with shape :math:`(*, 3, 1)`.

    Returns:
        The Essential matrix with the shape :math:`(*, 3, 3)`.
    """
    if not (len(R1.shape) >= 2 and R1.shape[-2:] == (3, 3)):
        raise AssertionError(R1.shape)
    if not (len(t1.shape) >= 2 and t1.shape[-2:] == (3, 1)):
        raise AssertionError(t1.shape)
    if not (len(R2.shape) >= 2 and R2.shape[-2:] == (3, 3)):
        raise AssertionError(R2.shape)
    if not (len(t2.shape) >= 2 and t2.shape[-2:] == (3, 1)):
        raise AssertionError(t2.shape)

    # first compute the camera relative motion
    R, t = relative_camera_motion(R1, t1, R2, t2)

    # get the cross product from relative translation vector
    Tx = cross_product_matrix(t[..., 0])

    return Tx @ R


def decompose_essential_matrix(E_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Decompose an essential matrix to possible rotations and translation.

    This function decomposes the essential matrix E using svd decomposition [96]
    and give the possible solutions: :math:`R1, R2, t`.

    Args:
       E_mat: The essential matrix in the form of :math:`(*, 3, 3)`.

    Returns:
       A tuple containing the first and second possible rotation matrices and the translation vector.
       The shape of the tensors with be same input :math:`[(*, 3, 3), (*, 3, 3), (*, 3, 1)]`.
    """
    if not (len(E_mat.shape) >= 2 and E_mat.shape[-2:]):
        raise AssertionError(E_mat.shape)

    # decompose matrix by its singular values
    U, _, V = _torch_svd_cast(E_mat)
    Vt = V.transpose(-2, -1)

    mask = torch.ones_like(E_mat)
    mask[..., -1:] *= -1.0  # fill last column with negative values

    maskt = mask.transpose(-2, -1)

    # avoid singularities
    U = torch.where((torch.det(U) < 0.0)[..., None, None], U * mask, U)
    Vt = torch.where((torch.det(Vt) < 0.0)[..., None, None], Vt * maskt, Vt)

    W = cross_product_matrix(torch.tensor([[0.0, 0.0, 1.0]]).type_as(E_mat))
    W[..., 2, 2] += 1.0

    # reconstruct rotations and retrieve translation vector
    U_W_Vt = U @ W @ Vt
    U_Wt_Vt = U @ W.transpose(-2, -1) @ Vt

    # return values
    R1 = U_W_Vt
    R2 = U_Wt_Vt
    T = U[..., -1:]
    return (R1, R2, T)