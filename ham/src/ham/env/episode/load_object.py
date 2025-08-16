#!/usr/bin/env python3

from typing import Dict, Tuple, List

import itertools
from functools import partial
import json
import torch as th
import numpy as np
import trimesh
from pathlib import Path
from cho_util.math import transform as tx

from ham.env.episode.spec import DefaultSpec
from ham.util.torch_util import dcn
from ham.util.math_util import quat_rotate
from ham.env.scene.util import (_foot_radius,
                                _is_stable)


def _pad(x: np.ndarray, max_len: int):
    if len(x) < max_len:
        extra = max_len - len(x)
        x = np.concatenate(
            [x, x[np.random.choice(len(x), size=extra, replace=True)]],
            axis=0)
    else:
        x = x[np.random.choice(
            len(x), size=max_len, replace=False)]
    return x


def _pad_hulls(hulls: Dict[str,
                           trimesh.Trimesh]) -> Dict[str, np.ndarray]:
    n: int = max([len(h.vertices) for h in hulls.values()])
    out = {}
    for k, h in hulls.items():
        p = np.empty((n, 3), dtype=np.float32)
        v = h.vertices
        c = np.mean(v, axis=0, keepdims=True)
        p[:len(v)] = v
        p[len(v):] = c
        out[k] = p
    return out


def _array_from_map(
        keys: List[str],
        maps: Dict[str, th.Tensor],
        **kwds):
    if not isinstance(next(iter(maps.values())), th.Tensor):
        arr = np.stack([maps[k] for k in keys])
        return th.as_tensor(arr, **kwds)
    return th.stack([maps[k] for k in keys], dim=0).to(**kwds)


def is_thin(extent, threshold: float = 2.5):
    size = np.sort(extent)
    return (size[1] >= threshold * size[0])


class LoadObject(DefaultSpec):
    """
    Load metadata
    """

    @property
    def setup_keys(self): return ('obj_ctx',)

    @property
    def setup_deps(self): return (
        'object_files',
    )

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        obj_set = ctx['obj_set']
        num_env = ctx['num_env']
        num_obj_per_env = ctx['num_obj_per_env']
        device = ctx['device']

        obj_ctx = {}

        keys = [Path(f).stem for f in data['object_files']]
        object_keys = list(itertools.islice(
            itertools.cycle(keys),
            num_env * num_obj_per_env
        ))
        object_keys = np.asarray(object_keys)

        # map from dictionary to array, basically.
        convert = partial(_array_from_map,
                          object_keys,
                          dtype=th.float,
                          device=device)

        # NOTE(ycho): this is the only _numpy array_
        obj_ctx['obj_name'] = np.asarray(object_keys)

        obj_ctx['obj_radius'] = convert(
            {k: obj_set.radius(k) for k in object_keys})
        obj_ctx['obj_hull'] = convert(_pad_hulls(
            {k: obj_set.hull(k) for k in object_keys}
        ))
        obj_ctx['obj_bbox'] = convert({k: obj_set.bbox(k)
                                       for k in object_keys})
        obj_ctx['obj_cloud'] = convert({k: obj_set.cloud(k)
                                        for k in object_keys})

        # store __padded__ stable poses
        stable_poses = {k: obj_set.pose(k) for k in object_keys}
        min_len = min([len(v) for v in stable_poses.values()])
        max_len = min(
            max([len(v) for v in stable_poses.values()]),
            ctx['max_pose_count']
        )
        print(F'\tmin_len = {min_len}, max_len = {max_len}')

        stable_poses = {k: _pad(v[..., : 7], max_len)
                        for k, v in stable_poses.items()}
        obj_ctx['obj_stable_pose'] = convert(stable_poses)

        # == yaw-only logic ==
        is_yaw_only = None

        if ctx.get('use_yaw_only_logic', False):
            is_yaw_only = {
                k: is_thin(obj_set.obb(k)[1],
                           threshold=ctx['thin_threshold'])
                for k in object_keys
            }

        if ctx.get('yaw_only_key', None) is not None:
            is_yaw_only = {
                k: 1. if k in ctx['yaw_only_key'] else 0.
                for k in object_keys
            }

        if ctx.get('yaw_only_file', None) is not None:
            with open(ctx['yaw_only_file'], 'r') as fp:
                yawonly_list = [str(s) for s in json.load(fp)]
            if is_yaw_only is not None:
                for k in yawonly_list:
                    is_yaw_only[k] = 1.
            else:
                is_yaw_only = {
                    k: 1. if k in yawonly_list else 0.
                    for k in object_keys
                }

        if is_yaw_only is not None:
            is_yaw_only = convert(is_yaw_only).bool()
        else:
            is_yaw_only = th.zeros(len(object_keys), dtype=th.bool,
                                   device=ctx['device'])
        obj_ctx['obj_yaw_only'] = is_yaw_only

        # _Online_ load for stable_masks
        cloud_np = dcn(obj_ctx['obj_cloud'])
        stable_poses_np = dcn(obj_ctx['obj_stable_pose'])
        stable_masks_np = np.zeros(stable_poses_np.shape[:-1],
                                   dtype=bool)
        for i in range(obj_ctx['obj_stable_pose'].shape[0]):
            for j in range(obj_ctx['obj_stable_pose'].shape[1]):
                pose = stable_poses_np[i, j]
                cloud_at_pose = tx.rotation.quaternion.rotate(
                    pose[None, 3:7],
                    cloud_np[i, ..., :3]) + pose[None, 0:3]
                stable_masks_np[i, j] = _is_stable(cloud_at_pose)
        obj_ctx['obj_stable_mask'] = th.as_tensor(stable_masks_np,
                                                  dtype=bool,
                                                  device=ctx['device'])

        # Add (dynamically...) foot_radius
        # TODO(ycho): cache
        if ctx['load_foot_radius']:
            cloud_np = dcn(obj_ctx['obj_cloud'])
            poses_np = dcn(obj_ctx['obj_stable_pose'])
            foot_radius_np = np.zeros(poses_np.shape[:-1],
                                      dtype=float)

            for i in range(cloud_np.shape[0]):
                cloud_i = cloud_np[i]
                for j in range(poses_np.shape[1]):
                    pose_ij = poses_np[i, j]
                    cloud_at_orn = tx.rotation.quaternion.rotate(
                        pose_ij[None, 3:7],
                        cloud_i)
                    foot_radius_np[i, j] = _foot_radius(
                        cloud_at_orn,
                        ctx['max_vertical_obstacle_height']
                    )
            obj_ctx['obj_foot_radius'] = th.as_tensor(foot_radius_np,
                                                      dtype=th.float,
                                                      device=ctx['device'])

        cloud_at_orn = quat_rotate(obj_ctx['obj_stable_pose'][..., None, :],
                                   obj_ctx['obj_cloud'][..., None, :, :])
        obj_ctx['obj_height'] = (cloud_at_orn[..., 2].amax(-1)
                                 - cloud_at_orn[..., 2].amin(-1))
        data['obj_ctx'] = obj_ctx
        return data
