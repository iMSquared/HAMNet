#!/usr/bin/env python3

from typing import Tuple, Dict
import torch as th

from ham.env.episode.spec import DefaultSpec
from ham.util.torch_util import randu


class TableDim(DefaultSpec):
    @property
    def setup_keys(self) -> Tuple[str, ...]: return ('table_dim',)
    @property
    def setup_deps(self) -> Tuple[str, ...]: return ()

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:

        table_dim_bound = ctx['table_dim_bound']
        if not isinstance(table_dim_bound, th.Tensor):
            table_dim_bound = th.as_tensor(table_dim_bound,
                                           dtype=th.float32,
                                           device=ctx['device'])
        dim_lo, dim_hi = th.unbind(table_dim_bound, dim=-2)
        data['table_dim'] = randu(dim_lo, dim_hi,
                                  (ctx['num_env'], 3),
                                  device=ctx['device'])
        return data

    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        return data


def main():
    table_dim = TableDim()


if __name__ == '__main__':
    main()
