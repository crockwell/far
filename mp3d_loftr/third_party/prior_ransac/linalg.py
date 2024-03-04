import torch
Tensor = torch.Tensor

pad = torch.nn.functional.pad
where = torch.where

def convert_points_from_homogeneous(points: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.

    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.

    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.

    Examples:
        >>> input = tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")

    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    # we check for points at max_val
    z_vec: Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: Tensor = torch.abs(z_vec) > eps
    scale = where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: Tensor) -> Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.

    Args:
        points: the points to be transformed with shape :math:`(*, N, D)`.

    Returns:
        the points in homogeneous coordinates :math:`(*, N, D+1)`.

    Examples:
        >>> input = tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    if not isinstance(points, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")


    return pad(points, [0, 1], "constant", 1.0)



def transform_points(trans_01: Tensor, points_1: Tensor) -> Tensor:
    r"""Function that applies transformations to a set of points.

    Args:
        trans_01: tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1: tensor of points of shape :math:`(B, N, D)`.
    Returns:
        a tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`

    Examples:

        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
    """
    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError(
            f"Input batch size must be the same for both tensors or 1. Got {trans_01.shape} and {points_1.shape}"
        )
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError(f"Last input dimensions must differ by one unit Got{trans_01} and {points_1}")

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(points_1.shape)
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = torch.repeat_interleave(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0)
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)
    return points_0