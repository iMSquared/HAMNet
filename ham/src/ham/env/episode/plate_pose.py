#!/usr/bin/env python3

from isaacgym import gymapi
from typing import Tuple, Dict
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.models.common import map_tensor


class PlatePose(DefaultSpec):

    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        'pose_meta',
        'curr_pose_meta',
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return ('reset_ids',
                                                     'obj_height',
                                                     'table_pos',
                                                     'table_dim',
                                                     'which_pose'
                                                     )

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        # Parse context
        z_eps = ctx['z_eps']
        min_height = ctx['min_height']
        gen = ctx['gen']

        reset_ids = data['reset_ids']

        if len(reset_ids) <= 0:
            return data
        obj_height = data['obj_height'][reset_ids]
        table_pos = data['table_pos'][reset_ids]
        table_geom = data['table_geom'][reset_ids]
        table_dim = data['table_dim'][reset_ids]
        which_pose = data['which_pose'][reset_ids]

        # base_height = location of the "tabletop" not "table"
        base_height = (
            table_pos[..., 2]
            + 0.5 * table_dim[..., 2]
        )

        obj_height_at_orn = th.take_along_dim(
            obj_height,
            which_pose,
            -1)
        object_height_in = (obj_height_at_orn + z_eps).clamp_min_(
            min_height
        )

        curr_pose_meta = gen.pose(
            table_geom,
            table_dim[..., :3],
            base_height,
            level=data.get('env_level', None),
            object_height=object_height_in
        )

        # Update the global `pose_meta` as well.
        # TODO(ycho): clone? or in-place?
        def _update_pose_meta(src, dst):
            dst[reset_ids] = src
            return dst
        if 'pose_meta' in data:
            data['pose_meta'] = map_tensor(src=curr_pose_meta,
                                           dst=data['pose_meta'],
                                           op=_update_pose_meta)
        else:
            # FIXME(ycho): what if the initial
            # set of reset indices are _not_ dense ?
            data['pose_meta'] = curr_pose_meta
        # cache curr_pose_meta
        data['curr_pose_meta'] = curr_pose_meta
        return data

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        root_tensor = ctx['root_tensor']
        reset_ids = data['reset_ids'].long()
        if len(reset_ids) <= 0:
            return data
        barrier_ids = ctx['barrier_id'][reset_ids]

        # Apply plate pose offsets w.r.t the table pos
        curr_pose_meta = data['curr_pose_meta']
        # zero-out velocity
        root_tensor[barrier_ids, ..., 7:] = 0
        # set pose
        root_tensor[barrier_ids, ..., :7] = (
            curr_pose_meta['pose']
        )
        # apply xy-directional pos DR
        root_tensor[barrier_ids, ..., :2] += (
            data['table_pos'][reset_ids, ..., None, :2]
        )
        return data
