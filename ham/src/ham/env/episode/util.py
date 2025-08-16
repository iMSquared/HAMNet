#!/usr/bin/env python3

from typing import (Optional, Dict)
import torch as th


def upsert(d: Dict[str, th.Tensor],
           i: Optional[th.Tensor],
           k: str,
           v: th.Tensor,
           sparse: bool = False,
           total: Optional[int] = None,
           dim: int = 0,
           fill_value: float = 0.0):
    if (k in d) and (i is not None):
        # NOTE(ycho): works for arbitrary `dim`
        # but only works for Tensor
        # d[k].index_copy_(dim, i, v)

        # NOTE(ycho): only works for dim=0
        # but it works for both ndarray+Tensor
        d[k][i] = v
    else:
        if sparse:
            if total is None:
                # NOTE(ycho): automatically infer `total`
                total = (i.max().item() + 1)
            shape = list(v.shape)
            shape[dim] = total
            out = th.full(shape, fill_value,
                          dtype=v.dtype, device=v.device)
            # d[k] = out.index_copy_(dim, i, v)
            d[k][i] = v
        else:
            # Lazy and non-robust version
            d[k] = v
    return d


def main():
    i = th.arange(32)
    v = th.randn((32, 5))
    d = upsert({}, i, 'a', v, sparse=True)
    print(d)

    d2 = upsert({}, i, 'a', v, sparse=False)
    print(d2)

    d3 = {'a': th.zeros_like(d2['a'])}
    print(d3)
    d3 = upsert(d3, i, 'a', v, sparse=False)
    print(d3)

    v = th.randn((32, 5))

    for dim in [0, 1]:
        d4 = {'a': th.zeros((7, 9))}
        d5 = {}
        i = th.randint(d4['a'].shape[dim],
                       size=(4,))
        v_shape = list(d4['a'].shape)
        v_shape[dim] = 4
        v = th.randn(v_shape)
        d4 = upsert(d4, i, 'a', v,
                    dim=dim)
        d5 = upsert(d5, i, 'a', v,
                    dim=dim,
                    total=d4['a'].shape[dim],
                    sparse=True)
        print(d4['a'].shape,
              d5['a'].shape)  # 79-49
        print(d4['a'] - d5['a'])


if __name__ == '__main__':
    main()
