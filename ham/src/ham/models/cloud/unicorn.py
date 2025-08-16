#!/usr/bin/env python3

from typing import Optional, Tuple
from dataclasses import dataclass

import torch as th
import torch.nn as nn

from ham.models.common import MLP
from ham.util.config import ConfigBase

from ham.models.cloud.point_mae import (
    get_patch_module,
    get_pos_enc_module,
    get_group_module_v2,
    PointMAEEncoder,
    PointMAELayer,
    subsample,
)

from icecream import ic


class PointMAEEncoderWrapper(PointMAEEncoder):
    def forward(self, *args, **kwds):
        return super().forward(*args, **kwds)[0]


class MLPEncoder(nn.Module):
    @dataclass
    class Config(ConfigBase):
        model_dim: int = 128
        patch_size: int = 32
        num_layer: int = 4

        group_type: str = 'fps'
        pos_enc_type: str = 'mlp'
        patch_type: str = 'mlp'
        encoder_type: str = 'mlp'

        num_patch_level: int = 0
        point_dim: int = 3
        pe_dim: Optional[int] = None

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        model_dim = cfg.model_dim

        self.group = get_group_module_v2(
            cfg.group_type, cfg.patch_size, recenter=True)
        # raise ValueError(F'got patch_size={cfg.patch_size}')

        if cfg.num_patch_level == 0:
            self.patch = get_patch_module(
                cfg.patch_type, model_dim, cfg.patch_size, False,
                point_dim=cfg.point_dim)
        else:
            patch = {str(lvl): get_patch_module(cfg.patch_type, model_dim,
                                                cfg.patch_size // (1 << lvl),
                                                point_dim=cfg.point_dim,
                                                pe_dim=cfg.pe_dim)
                     for lvl in range(cfg.num_patch_level)}
            self.patch = nn.ModuleDict(patch)
            self.groups = {str(lvl): get_group_module_v2(
                cfg.group_type, cfg.patch_size // (1 << lvl), recenter=True
            )
                for lvl in range(cfg.num_patch_level)}

        self.pos_enc = get_pos_enc_module(cfg.pos_enc_type, model_dim, 3)
        if cfg.encoder_type == 'mlp':
            self.encoder = MLP((model_dim,) * cfg.num_layer,
                               act_cls=nn.GELU,
                               use_bn=False,
                               use_ln=True)
        elif cfg.encoder_type == 'xfm':
            self.encoder = PointMAEEncoderWrapper(
                PointMAEEncoderWrapper.Config(
                    layer=PointMAELayer.Config(hidden_size=model_dim),
                    num_hidden_layers=cfg.num_layer
                ))
        else:
            raise ValueError(F'Unknown encoder_type={cfg.encoder_type}')

    @property
    def patch_size(self) -> int:
        return self.cfg.patch_size

    def forward(self,
                x: th.Tensor,
                z_ctx: Optional[th.Tensor] = None,
                aux=None,
                patch_level: Optional[int] = None,
                group_level: Optional[int] = None,
                patch_point: Optional[th.Tensor] = None
                ):
        cfg = self.cfg
        _aux = {}

        # group_level determines the _number_ of groups.
        # patch_level determines the _size_ of patches.
        if (group_level is None) or (cfg.num_patch_level == 0):
            p, c = self.group(
                x,
                aux=_aux,
                center=patch_point
            )
        else:
            p, c = self.groups[str(group_level)](
                x,
                aux=_aux,
                center=patch_point
            )

        if aux is not None:
            # mainly adds "patch_index"
            aux.update(_aux)
            aux['patch_center'] = c

        if cfg.num_patch_level == 0:
            # ic(p.shape, c.shape)
            z_pre = self.patch(p - c[..., None, :]) + self.pos_enc(
                    c[..., :3]
            )
        else:
            # mixture of
            # multi-resolution patch embeddings
            # to deal with sparsity problems... maybe
            z_patch = []

            patch_levels = range(cfg.num_patch_level)
            if patch_level is not None:
                patch_levels = [patch_level]

            for lvl in patch_levels:
                patch_size: int = (cfg.patch_size // (1 << lvl))
                p_sub = subsample(p, patch_size)
                z_patch.append(self.patch[str(lvl)](p_sub - c[..., None, :]))

            if len(z_patch) == 1:
                z_patch = z_patch[0]
            else:
                z_patch = th.stack(z_patch, dim=-1)
                indices = th.randint(z_patch.shape[-1],
                                     size=z_patch.shape[:-1],
                                     device=z_patch.device)
                z_patch = th.take_along_dim(z_patch,
                                            indices[..., None],
                                            dim=-1).squeeze(dim=-1)
            z_pos = self.pos_enc(c)
            z_pre = z_patch + z_pos

        if z_ctx is not None:
            z_pre = th.cat([z_pre, z_ctx], dim=-2)

        s_pre = z_pre.shape
        z_pre = z_pre.reshape(-1, *s_pre[-2:])
        z_enc = self.encoder(z_pre)
        z_enc = z_enc.reshape(*s_pre[:-1], z_enc.shape[-1])
        return z_enc


def main():
    encoder = MLPEncoder(MLPEncoder.Config(
        patch_size=128,
        # patch_level=(0, 1, 2, 3)
        num_patch_level=4,
        pe_dim=128
    ))
    print(encoder)
    x = th.randn((1, 2048, 3))
    encoder(x)


if __name__ == '__main__':
    main()
