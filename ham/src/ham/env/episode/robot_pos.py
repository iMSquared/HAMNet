#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.util.torch_util import randu


class RobotPos(DefaultSpec):
    @property
    def reset_keys(self) -> Tuple[str, ...]: return ('robot_pos',)

    @property
    def reset_deps(self) -> Tuple[str, ...]: return ('reset_ids',
                                                     'table_pos',
                                                     'table_dim',
                                                     )

    def _sample(self, ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        # FIXME(ycho):
        # The franka base poses are still indeed
        # _sampled_, but never applied after the first step.
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data

        table_pos = data['table_pos'][reset_ids]
        table_dim = data['table_dim'][reset_ids]

        # compute (x,z) relative to the table
        x = (
            # table_pos[..., 0]
            th.zeros_like(table_pos[..., 0])
            # FIXME(ycho)
            # - 0.5 * table_dim[..., 0]
            - 0.25
            - ctx['keepout_radius']
        )
        y = 0.0 * x
        if ctx['base_height'] == 'ground':
            z = 0.0
        elif ctx['base_height'] == 'table':
            # FIXME(ycho)
            # z = table_pos[..., 2] + 0.5 * table_dim[..., 2]
            # z = 0.4 #0.5 * table_dim[..., 2]
            z = th.zeros_like(table_pos[..., 2]) + 0.4
            # root_tensor[iii, 2] = env.scene.table_dims[..., 2]
        else:
            raise KeyError(F'Unknown base_height = {ctx["base_height"]}')
        robot_pos = th.stack([x, y, z], dim=-1)

        upsert(data, reset_ids, 'robot_pos', robot_pos)
        return data

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return self._sample(ctx, data)

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        # NOTE(ycho):
        # we're going to apply_reset()
        # inside franka.py by lookup in data['robot_pos'].
        return data


def main():
    table_pos = RobotPos()


if __name__ == '__main__':
    main()
