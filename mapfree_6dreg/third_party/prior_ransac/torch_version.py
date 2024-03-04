from typing import TYPE_CHECKING, Callable, ContextManager, List, Optional, Tuple, TypeVar

import torch
from packaging import version
from torch import Tensor


def torch_version() -> str:
    """Parse the `torch.__version__` variable and removes +cu*/cpu."""
    return torch.__version__.split('+')[0]


def torch_version_lt(major: int, minor: int, patch: int) -> bool:
    _version = version.parse(torch_version())
    return _version < version.parse(f"{major}.{minor}.{patch}")


def torch_version_le(major: int, minor: int, patch: int) -> bool:
    _version = version.parse(torch_version())
    return _version <= version.parse(f"{major}.{minor}.{patch}")


def torch_version_ge(major: int, minor: int, patch: Optional[int] = None) -> bool:
    _version = version.parse(torch_version())
    if patch is None:
        return _version >= version.parse(f"{major}.{minor}")
    else:
        return _version >= version.parse(f"{major}.{minor}.{patch}")