#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.util.torch_util import randu


class RobotDof(DefaultSpec):
    @property
    def reset_keys(self) -> Tuple[str, ...]: return ('robot_dof',)

    @property
    def reset_deps(self) -> Tuple[str, ...]: return ('reset_ids',
                                                     'table_pos',
                                                     'table_dim',
                                                     'robot_pos',
                                                     # 'pose_meta',
                                                     'curr_pose_meta',
                                                     )

    def _sample(self, ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        device = ctx['device']
        box_min = ctx['box_min']
        box_max = ctx['box_max']
        cabinet_ik = ctx['cabinet_ik']
        wall_ik = ctx['wall_ik']
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data

        # FIXME(ycho): consider reducing reliance on `curr_pose_meta`
        has_ceil = data['curr_pose_meta']['case']['has_ceil']
        table_pos = data['table_pos'][reset_ids]
        table_dim = data['table_dim'][reset_ids]
        robot_pos = data['robot_pos'][reset_ids]

        rel_height = (table_pos[..., 2] + 0.5 * table_dim[..., 2]
                      - robot_pos[..., 2])
        num_buckets, num_ik = cabinet_ik.shape[:2]
        # HACK: hardcoded height assumption
        bucket_indices = (((rel_height + 0.045) / 0.6) * (num_buckets - 1)
                          ).to(th.long).clamp(0, num_buckets - 1)
        ik_indices = th.randint(high=num_ik,
                                size=(len(reset_ids),),
                                dtype=th.long,
                                device=device)

        robot_dof = th.where(
            has_ceil[..., None],
            cabinet_ik[
                bucket_indices,
                ik_indices
            ],
            wall_ik[
                bucket_indices,
                ik_indices
            ])
        # randomize q[6]
        robot_dof[..., 6] = randu(box_min[6],
                                  box_max[6],
                                  len(reset_ids),
                                  device=device)
        upsert(data, reset_ids, 'robot_dof', robot_dof)
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
    table_pos = RobotDof()


if __name__ == '__main__':
    main()
