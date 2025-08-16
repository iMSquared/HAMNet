#!/usr/bin/env python3

from typing import Tuple, Dict

import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.util.torch_util import dcn


class EnvCode(DefaultSpec):
    """
    Calculate env.code (full list of stochastic params that define an episode)
    Mostly for debugging.
    """
    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        'env_code',
    )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return (
        'curr_pose_meta',
        'reset_ids',
    )

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data

        pose_meta = data['curr_pose_meta']
        table_dim = data['table_dim'][reset_ids]
        table_pos = data['table_pos'][reset_ids]

        codes = [
            # 4
            pose_meta['base']['ramp_pos'],
            # 4
            pose_meta['base']['ramp_ang'],
            # 6
            pose_meta['base']['height'],
            # 4
            pose_meta['case']['has_wall'] * pose_meta['case']['height'][..., None, None],
            # 1
            pose_meta['case']['has_ceil'] * pose_meta['case']['height'],
            # 3
            table_dim,
            # 3
            table_pos
        ]
        env_code = th.cat([x.reshape(num_reset, -1)
                           for x in codes], dim=-1)
        upsert(data, reset_ids, 'env_code', env_code)
        return data
