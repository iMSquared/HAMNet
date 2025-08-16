#!/usr/bin/env python3

from isaacgym import gymapi
from typing import Tuple, Dict
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.models.common import map_tensor


class SceneCloud(DefaultSpec):

    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        'scene_cloud',
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return ('reset_ids',
                                                     'curr_pose_meta',
                                                     'table_pos',
                                                     'table_dim',
                                                     )

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        cloud_size: int = ctx['cloud_size']
        ws_height = ctx['max_ws_height']
        gen = ctx['gen']

        reset_ids = data['reset_ids']
        if len(reset_ids) <= 0:
            return data
        pose_meta = data['curr_pose_meta']
        table_dim = data['table_dim'][reset_ids]
        table_pos = data['table_pos'][reset_ids]

        base_height = (table_pos[..., 2] + 0.5 * table_dim[..., 2])
        # NOTE(ycho): `clear_xy` here is necessary
        # since `pose_meta` does _not_ include (and
        # currently _cannot_ include) table_pos[...,:2]
        # information (i.e. result of table pos. DR)
        clear_xy = th.as_tensor([0, 0, 1],
                                dtype=th.float32,
                                device=ctx['device'])
        ws_lo = table_pos * clear_xy[None] - 0.5 * table_dim
        # FIXME(ycho): 0.01 here is a bit of a "hack"
        ws_lo[..., 2] = base_height - 0.01
        ws_hi = table_pos * clear_xy[None] + 0.5 * table_dim
        ws_hi[..., 2] = (base_height + ws_height)
        workspace = th.stack([ws_lo, ws_hi], dim=-2)

        # Sample interior point cluod from scene.
        scene_cloud = gen.cloud(
            workspace,
            table_dim,
            pose_meta,
            num_samples=cloud_size
        )
        scene_cloud[..., :2] += table_pos[..., None, :2]

        upsert(data, reset_ids,
               'scene_cloud', scene_cloud)
        return data
