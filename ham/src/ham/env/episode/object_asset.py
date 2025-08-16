#!/usr/bin/env python3

from isaacgym import gymapi
from typing import Dict, Tuple
from tqdm.auto import tqdm
from pathlib import Path
import itertools

import torch as th
import numpy as np

from ham.env.episode.spec import DefaultSpec


class ObjectAsset(DefaultSpec):
    @property
    def setup_keys(self) -> Tuple[str, ...]: return (
        'object_files',
        'object_asset',
    )

    @property
    def setup_deps(self) -> Tuple[str, ...]: return (
        'table_asset',
    )

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:

        max_load: int = ctx['num_obj_per_env'] * ctx['num_env']
        obj_set = ctx['obj_set']
        urdfs = [obj_set.urdf(k) for k in obj_set.keys()
                 if (obj_set.pose(k) is not None)
                 ]
        num_obj = min(max_load, len(urdfs), ctx['num_object_types'])

        # FIXME(ycho): organize configs ...
        mode = ctx['obj_load_mode']
        if mode == 'train':
            object_files = np.random.choice(
                urdfs,
                size=num_obj,
                replace=False
            )
        elif mode == 'valid':
            # Deterministic and ordered list of object_files
            object_files = list(itertools.islice(
                itertools.cycle(urdfs),
                num_obj))
        else:
            raise KeyError(F'Unknown mode = {mode}')

        data['object_files'] = object_files
        return data

    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        data = dict(data)

        gym = ctx.get('gym', None)
        sim = ctx.get('sim', None)
        object_files = data['object_files']

        obj_assets = {}
        for index, filename in enumerate(
            tqdm(object_files, desc='create_object_assets')
        ):

            # FIXME(ycho): relies on string parsing
            # to identify the key for the specific
            # URDF file
            key = Path(filename).stem

            if key in obj_assets:
                continue

            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = False
            asset_options.fix_base_link = False
            # FIXME(ycho): hardcoded `thickness`
            asset_options.thickness = 0.001

            if ctx['override_inertia']:
                asset_options.override_com = True
                asset_options.override_inertia = True
                asset_options.density = ctx['density']
            else:
                asset_options.override_com = False
                asset_options.override_inertia = False

            # NOTE(ycho): set to `True` since we're directly using
            # the convex decomposition result from CoACD.
            asset_options.vhacd_enabled = False
            if ctx['load_convex']:
                asset_options.convex_decomposition_from_submeshes = False
            else:
                asset_options.convex_decomposition_from_submeshes = True

            # NOTE(ycho): we allow gym=None for mock testing
            if gym is not None:
                obj_asset = gym.load_urdf(sim,
                                          str(Path(filename).parent),
                                          str(Path(filename).name),
                                          asset_options)
                obj_assets[key] = obj_asset
        data['object_asset'] = obj_assets
        return data


def main():
    obj_asset = ObjectAsset()


if __name__ == '__main__':
    main()
