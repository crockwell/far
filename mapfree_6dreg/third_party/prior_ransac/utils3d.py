
import numpy as np
import torch

def prespective_transform(points, K):
    # points : 3 x N
    img_points = np.matmul(K, points)
    z = points[:, 2:3, :]
    z_sign = np.sign(z)
    eps = 1e-4
    z_sign[z == 0] = 1
    zabs = np.abs(z)
    np.clip(zabs, a_min=eps, a_max=None, out=zabs)
    z = z_sign * zabs
    xy = img_points[:, :2, :] / z
    xyz = np.concatenate([xy, z], axis=1)
    return xyz
    
def convert_world_points_to_pixel(coords, RT, Kndc, use_cuda=False):
    batch = True
    if len(coords.shape) == 2:
        batch = False
        coords = coords[None]
        RT = RT[None]
        Kndc = Kndc[None]
    np_array = False
    if type(coords) == np.ndarray:
        np_array = True

    if np_array and (not use_cuda):
        points_cam = transform_points(coords, RT)
        xyz = prespective_transform(points_cam, Kndc)
    else:
        coords = tensor_utils.tensor_to_cuda(coords, use_cuda)
        RT = tensor_utils.tensor_to_cuda(RT, use_cuda)
        Kndc = tensor_utils.tensor_to_cuda(Kndc, use_cuda)

        points_cam = transform_points(coords, RT)
        xyz = prespective_transform(points_cam, Kndc)

    if not batch:
        xyz = xyz[0]

    return xyz


def transform_points_torch(points, RT):
    pointsCam = torch.bmm(RT, points)
    return pointsCam

def transform_points(points, RT):
    batched = True
    if len(points.shape) == 2:
        points = points[None]
        RT = RT[None]
        batched = False

    if type(points) == np.ndarray:
        points = np.concatenate([points, points[:, 0:1, :] * 0 + 1], axis=1)
        pointsCam = np.matmul(RT, points)
    else:
        points = torch.cat([points, points[:, 0:1, :] * 0 + 1], dim=1)
        
        pointsCam = transform_points_torch(points, RT)
    pointsCam = pointsCam[:, 0:3, :]
    if not batched:
        pointsCam = pointsCam[0]

    return pointsCam


def convert_pixel_to_world_points(coords, RT, Kndc):

    batch = True

    if len(coords.shape) == 2:
        batch = False
        coords = coords[None]
        RT = RT[None]
        Kndc = Kndc[None]
    np_array = False
    if type(coords) == np.ndarray:
        np_array = True

    if np_array:
        invK = np.linalg.inv(Kndc)
        invRT = np.linalg.inv(RT)
        coords = np.matmul(invK, coords)
        coords = transform_points(coords, invRT)
    else:
        assert False, 'need numpy array'
    if not batch:
        coords = coords[0]

    return coords

def create_depth2img_points(depth_map, RT, kNDC, use_cuda=True, step_size=1):
    img_h, img_w = depth_map.shape[0], depth_map.shape[1]
    x = np.linspace(0, img_w - 1, num=img_w)
    y = np.linspace(0, img_h - 1, num=img_h)
    xs, ys = np.meshgrid(x, y, indexing="xy")
    coordinates = np.stack([xs / (img_h - 1), ys / (img_h - 1)], axis=0)
    s = img_w / img_h
    coordinates[0, ...] = coordinates[0, ...] * 2 - s

    coordinates[1, ...] = coordinates[1, ...] * 2 - 1

    ndc_pts = np.concatenate(
        [
            coordinates,
            1
            + 0
            * depth_map[
                None,
            ],
        ],
        axis=0,
    )
    points = (
        ndc_pts
        * depth_map[
            None,
        ]
    )
    
    
    valid_points = depth_map > 0.01
    if step_size > 1:
        points = points[:, ::step_size, ::step_size]
        valid_points = valid_points[::step_size, ::step_size]

    points = points.reshape(3, -1)
    valid_points = valid_points.reshape(-1)
    points_world = convert_pixel_to_world_points(points, RT, kNDC)
    points_world = points_world.transpose(1, 0)
    points_world = points_world[valid_points == True, :]
    return points_world
