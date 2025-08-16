#!/usr/bin/env python3

from typing import Optional
import torch as th
import numpy as np
import open3d as o3d
import numpy as np

from ham.util.torch_util import dcn

def np2o3d(x: np.ndarray):
    if x is None:
        return None
    return o3d.core.Tensor.from_numpy(x)


def np2o3d_img2pcd(
        color: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_cb: Optional[np.ndarray] = None,
        normal: bool = True,
        depth_max: float = 3.0
):
    """
    Args:
        color: color image
        depth: depth image
        K: intrinsics
        T_cb: extrinsics (camera_from_base)
    """
    T_cb = dcn(T_cb)
    T_cb = o3d.core.Tensor.from_numpy(T_cb)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(color.astype(np.uint8)),
            o3d.t.geometry.Image(depth.astype(np.float32))
        ),
        K, #np2o3d(K),
        depth_scale=1.0,
        with_normals=normal,
        extrinsics=T_cb,
        depth_max=depth_max
    )
    return pcd


def th2o3d_dev(device):
    device = str(device)
    if 'cpu' in device:
        return o3d.core.Device('CPU:0')
    else:
        if ':' not in device:
            dev_id = th.cuda.current_device()
            device = F'cuda:{dev_id}'
        return o3d.core.Device(device.upper())


def o3d2th(x):
    return th.utils.dlpack.from_dlpack(x.to_dlpack())

def np2o3d(x: np.ndarray):
    return o3d.core.Tensor.from_dlpack(
        x.__dlpack__()
    )

def th2o3d(x: th.Tensor):
    return o3d.core.Tensor.from_dlpack(
        th.utils.dlpack.to_dlpack(x)
    )


def o3d2th_pcd(x: o3d.t.geometry.PointCloud):
    return o3d2th(x.point.positions)


def th2o3d_pcd(x: th.Tensor, c: Optional[th.Tensor] = None):
    x = th2o3d(x)
    pcd = o3d.t.geometry.PointCloud(x.device)
    pcd.point.positions = x
    if c is not None:
        c = th2o3d(c)
        pcd.point.colors = c
    return pcd



def main():
    for dev in ['cpu', 'cuda', 'cuda:0']:
        for wrap in (True, False):
            if wrap:
                device = th.device(dev)
            else:
                device = dev
            o3d_dev = th2o3d_dev(dev)

            x = th.randn((512, 3), dtype=th.float32)
            o3d.visualization.draw(th2o3d_pcd(x))


if __name__ == '__main__':
    main()
