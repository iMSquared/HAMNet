#!/usr/bin/env python3

import torch as th


def rotation_matrix_2d(x: th.Tensor) -> th.Tensor:
    R = th.empty((*x.shape, 2, 2),
                 dtype=x.dtype,
                 device=x.device)
    c, s = th.cos(x), th.sin(x)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    return R


def sample_cloud_from_triangles(ouv: th.Tensor,
                                num_samples: int,
                                eps: float = 1e-6) -> th.Tensor:
    double_area = th.linalg.norm(
        th.cross(ouv[..., 1, :], ouv[..., 2, :]),
        dim=-1)
    i = th.multinomial(double_area + eps,
                       num_samples,
                       replacement=True)[..., :, None, None]
    ouv = th.take_along_dim(ouv, i, dim=-3)
    o, u, v = th.unbind(ouv, dim=-2)

    # Triangle-sampling parameters
    st = th.rand((*ouv.shape[:-3], num_samples, 2),
                 dtype=ouv.dtype,
                 device=ouv.device)
    in_tri = (st.sum(dim=-1, keepdim=True) <= 1.0)
    st = th.where(in_tri, st, 1 - st)
    s, t = th.unbind(st[..., None], dim=-2)
    p = o + s * u + t * v
    ouv = p.reshape(*ouv.shape[:-3], -1, 3)
    return ouv
