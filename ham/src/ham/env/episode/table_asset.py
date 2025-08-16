#!/usr/bin/env python3

from isaacgym import gymapi
from typing import Tuple, Dict
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import torch as th

from ham.env.episode.spec import DefaultSpec


def _create_table_asset_options():
    asset_options = gymapi.AssetOptions()
    # asset_options.fix_base_link = True
    asset_options.fix_base_link = False
    asset_options.flip_visual_attachments = False
    asset_options.vhacd_enabled = False
    asset_options.thickness = 0.001  # ????
    # asset_options.thickness = 0.02  # ????
    asset_options.convex_decomposition_from_submeshes = False
    # asset_options.convex_decomposition_from_submeshes = True
    asset_options.override_com = False
    asset_options.override_inertia = False
    asset_options.disable_gravity = True
    return asset_options


def _load_table_assets(texts,
                       geoms,
                       gym,
                       sim,
                       as_urdf: bool = False,
                       fuse: bool = False):
    s = geoms.shape
    asset_options = _create_table_asset_options()
    table_assets = []

    with TemporaryDirectory() as tmpdir:
        for i_batch in range(texts.shape[0]):
            if fuse:
                filename = (
                    Path(tmpdir) / F'table_{i_batch:02d}.urdf'
                )
                with open(str(filename), 'w') as fp:
                    fp.write(texts[i_batch])
                table_asset = gym.load_urdf(sim, tmpdir,
                                            filename.name,
                                            asset_options)
                table_assets.append(table_asset)
            else:
                table_assets_i = []
                for i_part in range(texts.shape[1]):
                    if as_urdf:
                        # == as-urdf ==
                        text = texts[i_batch, i_part]
                        filename = (Path(tmpdir) /
                                    F'plate_{i_batch:02d}_{i_part:02d}.urdf')
                        with open(str(filename), 'w') as fp:
                            fp.write(text)
                        # NOTE(ycho): are these options necessary?
                        asset_options.density = 0.0
                        asset_options.override_com = False
                        asset_options.override_inertia = False
                        table_asset = gym.load_urdf(sim, tmpdir,
                                                    filename.name,
                                                    asset_options)
                    else:
                        # == as-box ==
                        # NOTE(ycho): _STRONG_ assumption that
                        # `geoms` are given as axis-aligned boxes
                        # (probably ok for now)
                        asset_options.density = 0.0
                        asset_options.override_com = False
                        asset_options.override_inertia = False
                        x, y, z = [float(x) for x in geoms[i_batch, i_part]]
                        table_asset = gym.create_box(sim, x, y, z,
                                                     asset_options)
                    table_assets_i.append(table_asset)
                table_assets.append(table_assets_i)
    if fuse:
        table_assets = np.reshape(table_assets, (*s[:-2]))
    else:
        table_assets = np.reshape(table_assets, (*s[:-2], -1))
    return table_assets


class TableAsset(DefaultSpec):
    @property
    def setup_keys(self) -> Tuple[str, ...]: return (
        # FIXME(ycho): should maybe be "plate_geom"
        # to avoid confusion with `table_dim`.
        'table_geom',
        # 'table_urdf',
        # TODO(ycho):
        # think about whether this
        # should be included here...!!
        'table_asset'
    )

    @property
    def setup_deps(self) -> Tuple[str, ...]: return ('table_dim',)

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        gen = ctx['gen']
        geom = gen.geom(data['table_dim'])
        data['table_geom'] = geom
        return data

    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        # data = dict(data)

        # == URDF text format ==
        # (gen, tmpdir) -> table_urdf_text
        gen = ctx['gen']
        gym = ctx.get('gym', None)
        tmpdir = ctx['tmpdir']
        outputs = gen.urdf(
            data['table_dim'],
            tmpdir,
            fuse_pose=False,
            geom=data['table_geom']
        )
        # data['table_urdf_text'] = outputs['urdf']

        # == as Isaac Gym asset ==
        # (table_urdf_text, table_geom) -> table_asset
        # NOTE(ycho): we allow gym=None for mock testing
        if gym is not None:
            assets = _load_table_assets(
                # data['table_urdf_text'],
                outputs['urdf'],
                data['table_geom'],
                gym,
                ctx['sim'],
                False,
                False)
            # NOTE(ycho): list -> dict
            data['table_asset'] = {i: v for (i, v) in enumerate(assets)}
        return data


def main():
    table_asset = TableAsset()


if __name__ == '__main__':
    main()
