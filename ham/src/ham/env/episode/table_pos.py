#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert
from ham.util.torch_util import randu


class TablePos(DefaultSpec):
    @property
    def reset_keys(self) -> Tuple[str, ...]: return ('table_pos',)

    @property
    def reset_deps(self) -> Tuple[str, ...]: return ('reset_ids',
                                                     'table_dim',
                                                     )

    def _sample(self, ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data

        table_pos_bound = ctx['table_pos_bound']
        if not isinstance(table_pos_bound, th.Tensor):
            table_pos_bound = th.as_tensor(table_pos_bound,
                                           dtype=th.float32,
                                           device=ctx['device'])
        pos_lo, pos_hi = th.unbind(table_pos_bound, dim=-2)

        table_pos = randu(pos_lo, pos_hi,
                          (num_reset, 3),
                          device=ctx['device'])
        table_dim = data['table_dim'][reset_ids]
        # NOTE(ycho): by default, configures the _bottom_
        # of the table at z=0.0.
        table_pos[..., 2] += 0.5 * table_dim[..., 2]
        upsert(data, reset_ids, 'table_pos', table_pos)
        return data

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return self._sample(ctx, data)

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        # NOTE(ycho): `table_pos` is applied inside `PlatePose`.
        return data


def main():
    table_pos = TablePos()


if __name__ == '__main__':
    main()
