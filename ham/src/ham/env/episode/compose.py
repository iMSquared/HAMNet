#!/usr/bin/env python3

from typing import Iterable, Tuple, Dict
from collections import defaultdict

import networkx as nx
import torch as th
import numpy as np
from ham.env.episode.spec import (Spec,
                                  DefaultSpec,
                                  WrapSpec)


def _print_item(item):
    k, v = item
    if isinstance(v, th.Tensor):
        print(F'{k}:T({v.shape})')
        if not th.isfinite(v).all():
            print(F'{k} seems to be non-finite.')
    elif isinstance(v, np.ndarray):
        print(F'{k}:N({v.shape})')
    else:
        print(F'{k}')


def sort_maps(maps: Iterable[Tuple[Tuple[str, ...], Tuple[str, ...]]]):
    """
    Topologically sort subnetworks in terms of
    their inputs and outputs.
    """
    G = nx.DiGraph()
    src = defaultdict(list)

    for i, (k_i, k_o) in enumerate(maps):
        for k in k_o:
            # ensure no duplicate output keys!
            if k in src:
                print('duplicate output', k)
            # assert (k not in src)
            src[k].append(i)

    for i, (k_i, k_o) in enumerate(maps):
        G.add_node(i)

        # "root"
        if len(k_i) <= 0:
            G.add_edge(-1, i)
            continue

        for k in k_i:
            if k not in src:
                # in this case, src \in `obs`
                # i.e. it's a source node (hopefully).
                continue
            for q in src[k]:
                if q == i:
                    continue
                G.add_edge(q, i)
    out = list(nx.topological_sort(G))
    if -1 in out:
        out.remove(-1)
    return out


class Compose(WrapSpec):
    def __init__(self, specs: Iterable[Spec]):
        super().__init__(specs)

        for s in specs:
            print(type(s).__qualname__)
            print(F'setup {s.setup_deps} -> {s.setup_keys}')
            print(F'reset {s.reset_deps} -> {s.reset_keys}')

        self.i_setup = list(
            sort_maps([(s.setup_deps, s.setup_keys) for s in specs]))
        # reset order
        self.i_reset = list(
            sort_maps([(s.reset_deps, s.reset_keys) for s in specs]))

        print('Setup order:')
        for ii in self.i_setup:
            print(self.specs[ii])
        print('Reset order:')
        for ii in self.i_reset:
            print(self.specs[ii])

        # declare keys/deps
        self.k_setup = tuple(set().union(*[s.setup_keys for s in specs]))
        self.k_reset = tuple(set().union(*[s.reset_keys for s in specs]))

        # We traverse through the specs in reverse order
        # to identify any dependencies that are resolved internally,
        # i.e. _produced_ by an earlier generator.
        d_setup = set()
        for i in self.i_setup[::-1]:
            d_setup.difference_update(self.specs[i].setup_keys)
            d_setup.update(self.specs[i].setup_deps)
        self.d_setup = tuple(d_setup)

        d_reset = set()
        for i in self.i_reset[::-1]:
            d_reset.difference_update(self.specs[i].reset_keys)
            d_reset.update(self.specs[i].reset_deps)
        self.d_reset = tuple(d_reset)

    @property
    def setup_keys(self) -> Tuple[str, ...]: return self.k_setup
    @property
    def setup_deps(self) -> Tuple[str, ...]: return self.d_setup
    @property
    def reset_keys(self) -> Tuple[str, ...]: return self.k_reset
    @property
    def reset_deps(self) -> Tuple[str, ...]: return self.d_reset

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        ks = set()

        for i in self.i_setup:
            data = self.specs[i].sample_setup(ctx, data)
        return data

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        ks = set()

        # NOTE(ycho): root_tensor pop/set here
        # is not really necessary... it's just a sanity-check measure
        # to ensure sample_reset() does _not_ depend on `root_tensor`.
        root_tensor = None
        if 'root_tensor' in ctx:
            root_tensor = ctx.pop('root_tensor')

        for i in self.i_reset:
            data = self.specs[i].sample_reset(ctx, data)
        if root_tensor is not None:
            ctx['root_tensor'] = root_tensor
        return data

    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        ks = set()
        for i in self.i_setup:
            data = self.specs[i].apply_setup(ctx, data)
        return data

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        ks = set()
        for i in self.i_reset:
            data = self.specs[i].apply_reset(ctx, data)
        return data
