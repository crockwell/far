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
from kornia.core.check import KORNIA_CHECK_SHAPE, KORNIA_CHECK
from kornia.core.check import KORNIA_CHECK_SAME_SHAPE
from kornia.geometry import solvers
import cv2
import numpy as np

from torch_utils import _torch_svd_cast

# Compute degree 10 poly representing determinant (equation 14 in the paper)
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L368C5-L368C82
def determinant_to_polynomial(A: Tensor) -> Tensor:
    r"""Represent the determinant by the 10th polynomial, used for 5PC solver [@nister2004efficient].

    Args:
        A: Tensor :math:`(*, 3, 13)`.

    Returns:
        a degree 10 poly, representing determinant (Eqn. 14 in the paper).
    """

    cs = torch.zeros(A.shape[0], 11, device=A.device, dtype=A.dtype)
    cs[:, 0] = (
        A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 3]
        - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 12]
    )

    cs[:, 1] = (
        A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 2]
        - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 11]
    )

    cs[:, 2] = (
        A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 7]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 1]
        - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 7] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 10]
    )

    cs[:, 3] = (
        A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 7]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 3]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 7]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 6]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 12] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 12] * A[:, 1, 7] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 7] * A[:, 1, 10]
        - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 6] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 10]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 10]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 10]
    )

    cs[:, 4] = (
        A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 7]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 3] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 6]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 3]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 7] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 2]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 7]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 3]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 7]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 6]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 11] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 11] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 5]
        + A[:, 0, 12] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 12] * A[:, 1, 6] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 12]
        - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 7] * A[:, 1, 10]
        - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 6] * A[:, 1, 10]
        - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 3] * A[:, 2, 5] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 12]
        + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 7] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 12]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 10]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 10]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 12]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 10]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 10]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 10]
    )

    cs[:, 5] = (
        A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 7]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 2] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 6]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 3] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 5]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 3]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 6] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 2]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 7] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 1]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 7]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 6]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 3]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 2]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 7]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 10] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 10] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 6]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 5]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 11] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 11] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 1, 1] * A[:, 0, 12] * A[:, 2, 4]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 12] * A[:, 1, 5] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 12]
        - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 11]
        - A[:, 0, 0] * A[:, 2, 7] * A[:, 1, 10]
        - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 6] * A[:, 1, 10]
        - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 2] * A[:, 2, 5] * A[:, 1, 10]
        - A[:, 0, 3] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 12]
        + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 11]
        + A[:, 0, 4] * A[:, 2, 3] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 6] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 7] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 12]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 11]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 10]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 10]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 10]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 12]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 11]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 10]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 10]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 6] = (
        A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 9]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 7]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 1] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 6]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 2] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 5]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 3] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 3] * A[:, 1, 9] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 9]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 3]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 5] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 2]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 6] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 1]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 7] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 7] * A[:, 1, 9] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 7]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 6]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 5]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 3]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 2]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 1]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 7]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 9] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 9] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 6]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 10] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 10] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 5]
        + A[:, 1, 0] * A[:, 0, 12] * A[:, 2, 4]
        + A[:, 0, 11] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 11] * A[:, 1, 5] * A[:, 2, 0]
        - A[:, 0, 12] * A[:, 1, 4] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 12]
        - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 11]
        - A[:, 0, 0] * A[:, 2, 6] * A[:, 1, 10]
        - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 1] * A[:, 2, 5] * A[:, 1, 10]
        - A[:, 0, 2] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 12]
        + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 11]
        + A[:, 0, 4] * A[:, 2, 2] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 5] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 6] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 12]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 11]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 10]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 10]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 12]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 11]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 10]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 7] = (
        A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 9]
        + A[:, 0, 0] * A[:, 1, 7] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 7]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 6]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 1] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 5]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 2] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 2] * A[:, 1, 9] * A[:, 2, 4]
        + A[:, 0, 3] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 3] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 9]
        - A[:, 0, 4] * A[:, 1, 3] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 3]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 2]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 5] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 1]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 6] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 6] * A[:, 1, 9] * A[:, 2, 0]
        - A[:, 0, 7] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 7] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 7]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 6]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 5]
        + A[:, 0, 8] * A[:, 1, 3] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 3]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 2]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 1]
        - A[:, 0, 8] * A[:, 1, 7] * A[:, 2, 0]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 6]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 9] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 9] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 5]
        + A[:, 0, 10] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 10] * A[:, 1, 5] * A[:, 2, 0]
        + A[:, 1, 0] * A[:, 0, 11] * A[:, 2, 4]
        - A[:, 0, 11] * A[:, 1, 4] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 11]
        - A[:, 0, 0] * A[:, 2, 5] * A[:, 1, 10]
        - A[:, 0, 1] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 11]
        + A[:, 0, 4] * A[:, 2, 1] * A[:, 1, 10]
        + A[:, 0, 5] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 11]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 10]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 11]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 10]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 8] = (
        A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 9]
        + A[:, 0, 0] * A[:, 1, 6] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 6]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 5]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 1] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 1] * A[:, 1, 9] * A[:, 2, 4]
        + A[:, 0, 2] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 2] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 9]
        - A[:, 0, 4] * A[:, 1, 2] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 2]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 1]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 5] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 5] * A[:, 1, 9] * A[:, 2, 0]
        - A[:, 0, 6] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 6] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 6]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 5]
        + A[:, 0, 8] * A[:, 1, 2] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 2]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 1]
        - A[:, 0, 8] * A[:, 1, 6] * A[:, 2, 0]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 5]
        + A[:, 0, 9] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 9] * A[:, 1, 5] * A[:, 2, 0]
        + A[:, 0, 10] * A[:, 1, 0] * A[:, 2, 4]
        - A[:, 0, 10] * A[:, 1, 4] * A[:, 2, 0]
        - A[:, 0, 0] * A[:, 2, 4] * A[:, 1, 10]
        + A[:, 0, 4] * A[:, 2, 0] * A[:, 1, 10]
        + A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 10]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 10]
    )

    cs[:, 9] = (
        A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 9]
        + A[:, 0, 0] * A[:, 1, 5] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 5]
        - A[:, 0, 0] * A[:, 1, 9] * A[:, 2, 4]
        + A[:, 0, 1] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 1] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 9]
        - A[:, 0, 4] * A[:, 1, 1] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 1]
        + A[:, 0, 4] * A[:, 1, 9] * A[:, 2, 0]
        - A[:, 0, 5] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 5] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 5]
        + A[:, 0, 8] * A[:, 1, 1] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 1]
        - A[:, 0, 8] * A[:, 1, 5] * A[:, 2, 0]
        + A[:, 0, 9] * A[:, 1, 0] * A[:, 2, 4]
        - A[:, 0, 9] * A[:, 1, 4] * A[:, 2, 0]
    )

    cs[:, 10] = (
        A[:, 0, 0] * A[:, 1, 4] * A[:, 2, 8]
        - A[:, 0, 0] * A[:, 1, 8] * A[:, 2, 4]
        - A[:, 0, 4] * A[:, 1, 0] * A[:, 2, 8]
        + A[:, 0, 4] * A[:, 1, 8] * A[:, 2, 0]
        + A[:, 0, 8] * A[:, 1, 0] * A[:, 2, 4]
        - A[:, 0, 8] * A[:, 1, 4] * A[:, 2, 0]
    )

    for i in range(len(cs)):
        if cs[i][-1] == 0:
            import pdb; pdb.set_trace()

    return cs

def compute_fundamental(x1, x2):
  '''Computes the fundamental matrix from corresponding points x1, x2 using
  the 8 point algorithm.'''
  n = x1.shape[1]
  if x2.shape[1] != n:
    raise ValueError('Number of points do not match.')

  # Normalization is done in compute_fundamental_normalized().
  A = numpy.zeros((n, 9))
  for i in range(n):
    A[i] = [x1[0, i] * x2[0, i],  x1[0, i] * x2[1, i],  x1[0, i] * x2[2, i],
            x1[1, i] * x2[0, i],  x1[1, i] * x2[1, i],  x1[1, i] * x2[2, i],
            x1[2, i] * x2[0, i],  x1[2, i] * x2[1, i],  x1[2, i] * x2[2, i],
           ]

  # Solve A*f = 0 using least squares.
  U, S, V = numpy.linalg.svd(A)
  F = V[-1].reshape(3, 3)

  # Constrain F to rank 2 by zeroing out last singular value.
  U, S, V = numpy.linalg.svd(F)
  S[2] = 0
  F = numpy.dot(U, numpy.dot(numpy.diag(S), V))
  return F / F[2, 2]


def compute_fundamental_normalized(x1, x2):
  '''Computes the fundamental matrix from corresponding points x1, x2 using
  the normalized 8 point algorithm.'''
  n = x1.shape[1]
  if x2.shape[1] != n:
    raise ValueError('Number of points do not match.')

  # normalize.
  x1 = x1 / x1[2]
  mean_1 = numpy.mean(x1[:2], axis=1)
  S1 = numpy.sqrt(2) / numpy.std(x1[:2])
  T1 = numpy.array([[S1, 0, -S1 * mean_1[0]],
                    [0, S1, -S1 * mean_1[1]],
                    [0, 0, 1]])
  x1 = numpy.dot(T1, x1)

  x2 = x2 / x2[2]
  mean_2 = numpy.mean(x2[:2], axis=1)
  S2 = numpy.sqrt(2) / numpy.std(x2[:2])
  T2 = numpy.array([[S2, 0, -S2 * mean_2[0]],
                    [0, S2, -S2 * mean_2[1]],
                    [0, 0, 1]])
  x2 = numpy.dot(T2, x2)

  F = compute_fundamental(x1, x2)

  # denormalize.
  F = numpy.dot(T1.T, numpy.dot(F, T2))
  return F / F[2, 2]


def compute_right_epipole(F):
  '''Returns e with F * e = 0 (call with F.T for left epipole).'''
  U, S, V = numpy.linalg.svd(F)
  e = V[-1]  # S is diag([l1, l2, 0]). e's scale is arbitrary.
  return e / e[2]


def plot_epipolar_line(im, F, x, epipole=None, show_epipole=True):
  '''Plot the epipole and epipolar line F*x = 0.'''
  import pylab

  m, n = im.shape[:2]
  line = numpy.dot(F, x)

  t = numpy.linspace(0, n, 100)
  lt = numpy.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

  ndx = (lt >= 0) & (lt < m)
  pylab.plot(t[ndx], lt[ndx], linewidth=2)

  if show_epipole:
    if epipole is None:
      epipole = compute_right_epipole(F)
    pylab.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


def triangulate_point(x1, x2, P1, P2):
  '''Given two image coordinates x1, x2 of the same point X under different
  projections P1, P2, recovers X.'''
  M = numpy.zeros((6, 6))
  M[:3, :4] = P1
  M[:3, 4] = -x1

  M[3:, :4] = P2
  M[3:, 5] = -x2  # Intentionally 5, not 4.

  U, S, V = numpy.linalg.svd(M)
  X = V[-1, :4]
  return X / X[3]


def triangulate(x1, x2, P1, P2):
  '''Given n pairs of points, returns their 3d coordinates.'''
  n = x1.shape[1]
  if x2.shape[1] != n:
    raise ValueError('Number of points do not match.')

  X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]
  return numpy.array(X).T


def compute_P(x, X):
  '''Computes camera matrix from corresponding (homogeneous)
  2D and 3D points.'''
  n = x.shape[1]
  if X.shape[1] != n:
    raise ValueError('Number of points do not match.')

  M = numpy.zeros((3 * n, 12 + n))
  for i in range(n):
    M[3 * i          , 0:4] = X[:, i]
    M[3 * i + 1      , 4:8] = X[:, i]
    M[3 * i + 2      , 8:12] = X[:, i]
    M[3 * i:3 * i + 3, i + 12] = -x[:, i]

  U, S, V = numpy.linalg.svd(M)
  return V[-1, :12].reshape((3, 4))


def skew(a):
  '''Skew matrix A such that a x v = A*v for any v.'''
  return numpy.array([[0, -a[2], a[1]],
                      [a[2], 0, -a[0]],
                      [-a[1], a[0], 0]])


def compute_P_from_fundamental(F):
  '''Computes second camera matrix, assuming P1 = [I 0].
  Only up to a homography, since no calibration is given.'''
  e = compute_right_epipole(F.T)  # left epipole
  Te = skew(e)
  return numpy.vstack((numpy.dot(Te, F.T).T, e)).T


def compute_P_from_essential(E):
  # make sure E is rank 2
  U, S, V = numpy.linalg.svd(E)
  if numpy.linalg.det(numpy.dot(U, V)) < 0:
    V = -V
  E = numpy.dot(U, numpy.dot(numpy.diag([1, 1, 0]), V))

  # create matrices ("Hartley p 258" XXX)
  Z = skew([0, 0, -1])  # FIXME: Unused?
  W = numpy.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

  P2 = [numpy.vstack((numpy.dot(U, numpy.dot(W, V)).T,  U[:,2])).T,
        numpy.vstack((numpy.dot(U, numpy.dot(W, V)).T, -U[:,2])).T,
        numpy.vstack((numpy.dot(U, numpy.dot(W.T, V)).T,  U[:,2])).T,
        numpy.vstack((numpy.dot(U, numpy.dot(W.T, V)).T, -U[:,2])).T]
  return P2



def normalize_points(points: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    r"""Normalizes points (isotropic).

    Computes the transformation matrix such that the two principal moments of the set of points
    are equal to unity, forming an approximately symmetric circular cloud of points of radius 1
    about the origin. Reference: Hartley/Zisserman 4.4.4 pag.107

    This operation is an essential step before applying the DLT algorithm in order to consider
    the result as optimal.

    Args:
       points: Tensor containing the points to be normalized with shape :math:`(B, N, 2)`.
       eps: epsilon value to avoid numerical instabilities.

    Returns:
       tuple containing the normalized points in the shape :math:`(B, N, 2)` and the transformation matrix
       in the shape :math:`(B, 3, 3)`.
    """
    if len(points.shape) != 3:
        raise AssertionError(points.shape)
    if points.shape[-1] != 2:
        raise AssertionError(points.shape)

    x_mean = torch.mean(points, dim=1, keepdim=True)  # Bx1x2

    scale = (points - x_mean).norm(dim=-1, p=2).mean(dim=-1)  # B
    scale = torch.sqrt(torch.tensor(2.0)) / (scale + eps)  # B

    ones, zeros = torch.ones_like(scale), torch.zeros_like(scale)

    transform = torch.stack(
        [scale, zeros, -scale * x_mean[..., 0, 0], zeros, scale, -scale * x_mean[..., 0, 1], zeros, zeros, ones], dim=-1
    )  # Bx9

    transform = transform.view(-1, 3, 3)  # Bx3x3
    points_norm = transform_points(transform, points)  # BxNx2

    return (points_norm, transform)


def normalize_transformation(M: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Normalize a given transformation matrix.

    The function trakes the transformation matrix and normalize so that the value in
    the last row and column is one.

    Args:
        M: The transformation to be normalized of any shape with a minimum size of 2x2.
        eps: small value to avoid unstabilities during the backpropagation.

    Returns:
        the normalized transformation matrix with same shape as the input.
    """
    if len(M.shape) < 2:
        raise AssertionError(M.shape)
    norm_val: Tensor = M[..., -1:, -1:]
    return torch.where(norm_val.abs() > eps, M / (norm_val + eps), M)


def run_8point(points1: Tensor, points2: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    r"""Compute the fundamental matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 8 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed fundamental matrix with shape :math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape, points2.shape)
    if points1.shape[1] < 8:
        raise AssertionError(points1.shape)
    if weights is not None:
        if not (len(weights.shape) == 2 and weights.shape[1] == points1.shape[1]):
            raise AssertionError(weights.shape)

    points1_norm, transform1 = normalize_points(points1)
    points2_norm, transform2 = normalize_points(points2)

    # points1_norm = points1
    # points2_norm = points2
    # transform1 = torch.eye(3)[None].to(points1)
    # transform2 = torch.eye(3)[None].to(points2)
    
    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # Bx1xN

    ones = torch.ones_like(x1)

    # build equations system and solve DLT
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]

    X = torch.cat([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, ones], dim=-1)  # BxNx9

    # apply the weights to the linear system
    if weights is None:
        X = X.transpose(-2, -1) @ X
    else:
        w_diag = torch.diag_embed(weights)
        X = X.transpose(-2, -1) @ w_diag @ X
    # compute eigevectors and retrieve the one with the smallest eigenvalue

    _, _, V = _torch_svd_cast(X)
    F_mat = V[..., -1].view(-1, 3, 3)

    # reconstruct and force the matrix to have rank2
    U, S, V = _torch_svd_cast(F_mat)
    rank_mask = torch.tensor([1.0, 1.0, 0.0], device=F_mat.device, dtype=F_mat.dtype)

    F_projected = U @ (torch.diag_embed(S * rank_mask) @ V.transpose(-2, -1))
    F_est = transform2.transpose(-2, -1) @ (F_projected @ transform1)
    # hm_points1 = torch.pad(points1_norm, )
    hm_points1 = torch.nn.functional.pad(points1_norm, [0, 1], "constant", 1.0)
    hm_points2 = torch.nn.functional.pad(points2_norm, [0, 1], "constant", 1.0)
    err = torch.bmm(torch.bmm(hm_points2, F_est), hm_points1.permute(0, 2, 1))
    return normalize_transformation(F_est)


def run_5point_cv2(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    output = []
    #import time
    #start = time.time()

    points1_numpy = points1.cpu().numpy()
    points2_numpy = points2.cpu().numpy()

    for i in range(len(points1)):
        #import pdb; pdb.set_trace()
        estimate = cv2.findEssentialMat(points1_numpy[i], points2_numpy[i], np.eye(3), method=cv2.LMEDS, threshold=np.inf, maxIters=1)[0]
        if estimate is None:
            #import pdb; pdb.set_trace()
            estimate = np.eye(3)
        output.append(estimate)
    #end = time.time()
    #print(f"Time taken for 5pt: {round(end - start,2)}")

    try:
        out = torch.from_numpy(np.array(output)).to(points1.device).float()
    except:
        import pdb; pdb.set_trace()

    return out

def run_5point_our_kornia(points1: torch.Tensor, points2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Compute the essential matrix using the 5-point algorithm from Nister.

    The linear system is solved by Nister's 5-point algorithm [@nister2004efficient],
    and the solver implemented referred to [@barath2020magsac++][@wei2023generalized].

    Args:
        points1: A set of carlibrated points in the first image with a tensor shape :math:`(B, N, 2), N>=8`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2), N>=8`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed essential matrix with shape :math:`(B, 3, 3)`.
    """
    KORNIA_CHECK_SHAPE(points1, ['B', 'N', '2'])
    KORNIA_CHECK_SAME_SHAPE(points1, points2)
    KORNIA_CHECK(points1.shape[1] >= 5, "Number of points should be >=5")
    if weights is not None:
        KORNIA_CHECK_SAME_SHAPE(points1[:, :, 0], weights)

    batch_size, _, _ = points1.shape
    x1, y1 = torch.chunk(points1, dim=-1, chunks=2)  # Bx1xN
    x2, y2 = torch.chunk(points2, dim=-1, chunks=2)  # Bx1xN
    ones = torch.ones_like(x1)

    # build equations system and find null space.
    # https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf
    # [x * x', x * y', x, y * x', y * y', y, x', y', 1]
    # BxNx9
    X = torch.cat([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones], dim=-1)

    # apply the weights to the linear system
    if weights is None:
        X = X.transpose(-2, -1) @ X
    else:
        w_diag = torch.diag_embed(weights)
        X = X.transpose(-2, -1) @ w_diag @ X
    # compute eigevectors and retrieve the one with the smallest eigenvalue, using SVD
    # turn off the grad check due to the unstable gradients from SVD.
    # several close to zero values of eigenvalues.
    _, _, V = _torch_svd_cast(X)  # torch.svd
    null_ = V[:, :, -4:]  # the last four rows
    nullSpace = V.transpose(-1, -2)[:, -4:, :]

    coeffs = torch.zeros(batch_size, 10, 20, device=null_.device, dtype=null_.dtype)
    d = torch.zeros(batch_size, 60, device=null_.device, dtype=null_.dtype)

    def fun(i: int, j: int) -> torch.Tensor:
        return null_[:, 3 * j + i]

    # Determinant constraint
    coeffs[:, 9] = (
        solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(fun(0, 1), fun(1, 2)) - solvers.multiply_deg_one_poly(fun(0, 2), fun(1, 1)),
            fun(2, 0),
        )
        + solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(fun(0, 2), fun(1, 0)) - solvers.multiply_deg_one_poly(fun(0, 0), fun(1, 2)),
            fun(2, 1),
        )
        + solvers.multiply_deg_two_one_poly(
            solvers.multiply_deg_one_poly(fun(0, 0), fun(1, 1)) - solvers.multiply_deg_one_poly(fun(0, 1), fun(1, 0)),
            fun(2, 2),
        )
    )

    indices = torch.tensor([[0, 10, 20], [10, 40, 30], [20, 30, 50]])

    # Compute EE^T (Eqn. 20 in the paper)
    for i in range(3):
        for j in range(3):
            d[:, indices[i, j] : indices[i, j] + 10] = (
                solvers.multiply_deg_one_poly(fun(i, 0), fun(j, 0))
                + solvers.multiply_deg_one_poly(fun(i, 1), fun(j, 1))
                + solvers.multiply_deg_one_poly(fun(i, 2), fun(j, 2))
            )

    for i in range(10):
        t = 0.5 * (d[:, indices[0, 0] + i] + d[:, indices[1, 1] + i] + d[:, indices[2, 2] + i])
        d[:, indices[0, 0] + i] -= t
        d[:, indices[1, 1] + i] -= t
        d[:, indices[2, 2] + i] -= t

    cnt = 0
    for i in range(3):
        for j in range(3):
            row = (
                solvers.multiply_deg_two_one_poly(d[:, indices[i, 0] : indices[i, 0] + 10], fun(0, j))
                + solvers.multiply_deg_two_one_poly(d[:, indices[i, 1] : indices[i, 1] + 10], fun(1, j))
                + solvers.multiply_deg_two_one_poly(d[:, indices[i, 2] : indices[i, 2] + 10], fun(2, j))
            )
            coeffs[:, cnt] = row
            cnt += 1

    b = coeffs[:, :, 10:]
    singular_filter = torch.linalg.matrix_rank(coeffs[:, :, :10]) >= torch.max(
        torch.linalg.matrix_rank(coeffs), torch.ones_like(torch.linalg.matrix_rank(coeffs[:, :, :10])) * 10
    )

    eliminated_mat = torch.linalg.solve(coeffs[singular_filter, :, :10], b[singular_filter])

    coeffs_ = torch.cat((coeffs[singular_filter, :, :10], eliminated_mat), dim=-1)

    A = torch.zeros(coeffs_.shape[0], 3, 13, device=coeffs_.device, dtype=coeffs_.dtype)

    for i in range(3):
        A[:, i, 0] = 0.0
        A[:, i : i + 1, 1:4] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 10:13]
        A[:, i : i + 1, 0:3] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 10:13]
        A[:, i, 4] = 0.0
        A[:, i : i + 1, 5:8] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 13:16]
        A[:, i : i + 1, 4:7] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 13:16]
        A[:, i, 8] = 0.0
        A[:, i : i + 1, 9:13] = coeffs_[:, 4 + 2 * i : 5 + 2 * i, 16:20]
        A[:, i : i + 1, 8:12] -= coeffs_[:, 5 + 2 * i : 6 + 2 * i, 16:20]

    cs = solvers.determinant_to_polynomial(A)
    #cs = determinant_to_polynomial(A)
    E_models = []

    # for loop because of different numbers of solutions
    for bi in range(A.shape[0]):
        A_i = A[bi]
        null_i = nullSpace[bi]

        # companion matrix solver for polynomial
        C = torch.zeros((10, 10), device=cs.device, dtype=cs.dtype)
        C[0:-1, 1:] = torch.eye(C[0:-1, 0:-1].shape[0], device=cs.device, dtype=cs.dtype)
        C[-1, :] = -cs[bi][:-1] / cs[bi][-1]

        if torch.isinf(C).sum() > 0:
            continue
            #import pdb; pdb.set_trace()

        roots = torch.real(torch.linalg.eigvals(C))

        if roots is None:
            continue
        n_sols = roots.size()
        if n_sols == 0:
            continue
        Bs = torch.stack(
            (
                A_i[:3, :1] * (roots**3) + A_i[:3, 1:2] * roots.square() + A_i[0:3, 2:3] * (roots) + A_i[0:3, 3:4],
                A_i[0:3, 4:5] * (roots**3) + A_i[0:3, 5:6] * roots.square() + A_i[0:3, 6:7] * (roots) + A_i[0:3, 7:8],
            ),
            dim=0,
        ).transpose(0, -1)

        bs = (
            A_i[0:3, 8:9] * (roots**4)
            + A_i[0:3, 9:10] * (roots**3)
            + A_i[0:3, 10:11] * roots.square()
            + A_i[0:3, 11:12] * roots
            + A_i[0:3, 12:13]
        ).T.unsqueeze(-1)

        # We try to solve using top two rows,
        xzs = Bs[:, 0:2, 0:2].inverse() @ (bs[:, 0:2])

        mask = (abs(Bs[:, 2].unsqueeze(1) @ xzs - bs[:, 2].unsqueeze(1)) > 1e-3).flatten()
        if torch.sum(mask) != 0:
            q, r = torch.linalg.qr(Bs[mask].clone())  #
            xzs[mask] = torch.linalg.solve(r, q.transpose(-1, -2) @ bs[mask])  # [mask]

        # models
        Es = null_i[0] * (-xzs[:, 0]) + null_i[1] * (-xzs[:, 1]) + null_i[2] * roots.unsqueeze(-1) + null_i[3]

        # Since the rows of N are orthogonal unit vectors, we can normalize the coefficients instead
        inv = 1.0 / torch.sqrt((-xzs[:, 0]) ** 2 + (-xzs[:, 1]) ** 2 + roots.unsqueeze(-1) ** 2 + 1.0)
        Es *= inv
        if Es.shape[0] < 10:
            Es = torch.cat(
                (Es.clone(), torch.eye(3, device=Es.device, dtype=Es.dtype).repeat(10 - Es.shape[0], 1).reshape(-1, 9))
            )
        E_models.append(Es)

    # if not E_models:
    #     return torch.eye(3, device=cs.device, dtype=cs.dtype).unsqueeze(0)
    # else:
    return torch.cat(E_models).view(-1, 3, 3).transpose(-1, -2)

class RansacModel(object):
  def fit(self, data):
    data = data.T
    x1 = data[:3, :8]
    x2 = data[3:, :8]
    return compute_fundamental_normalized(x1, x2)

  def get_error(self, data, F):
    data = data.T
    x1 = data[:3]
    x2 = data[3:]

    # Sampson distance as error.
    Fx1 = numpy.dot(F, x1)
    Fx2 = numpy.dot(F, x2)
    denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    err = (numpy.diag(numpy.dot(x1.T, numpy.dot(F, x2))))**2 / denom
    return err


def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
  import ransac
  data = numpy.vstack((x1, x2))
  F, ransac_data = ransac.ransac(data.T, model, 8, maxiter, match_threshold, 20,
                                 return_all=True)
  return F, ransac_data['inliers']