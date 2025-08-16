#!/usr/bin/env python3


import torch
import torch as th


@th.jit.script
def point_line_distance(p: th.Tensor,
                        l: th.Tensor,
                        eps: float = 1e-6):
    """
    Arg:
        p: point cloud of shape (..., P, 3)
        l: line of shape (..., L, 2, 3)
    Return:
        out: _squared_ distance of shape (..., P, L)
    """
    v0, v1 = l.unbind(dim=-2)
    # v0 = (..., L, 3)
    v1v0 = v1 - v0
    # LAYOUT: (..., P, L, 3)
    pv0 = p[..., :, None, :] - v0[..., None, :, :]
    t_bot = th.einsum('...d, ...d -> ...', v1v0, v1v0)  # LD -> L
    t_bot = t_bot[..., None, :]
    t_top = th.einsum('...pld, ...ld -> ...pl',
                      pv0, v1v0)
    tt = (t_top / t_bot).masked_fill_(
        t_bot < eps, 0).clamp_(0, 1)

    p_proj = v0[..., None, :, :] + tt[..., None] * v1v0[..., None, :, :]
    diff = p[..., :, None, :] - p_proj
    dist = th.einsum('...d, ...d -> ...', diff, diff)
    return dist


@th.jit.script
def barycentric_coords(p: th.Tensor, tri: th.Tensor,
                       eps: float = 1e-6):
    """
    Arg:
        p: Point Cloud of shape (..., P, 3)
        t: Triangles of shape (..., T, 3, 3)

    Return:
        b: Barycentric coords of shape (..., P, T, 3)
    """
    v0, v1, v2 = th.unbind(tri, dim=-2)  # T3
    p0 = v1 - v0  # T3
    p1 = v2 - v0  # T3
    p2 = p[..., :, None, :] - v0[..., None, :, :]  # PT3

    d00 = th.einsum('...d, ...d -> ...', p0, p0)
    d01 = th.einsum('...d, ...d -> ...', p0, p1)
    d11 = th.einsum('...d, ...d -> ...', p1, p1)
    d20 = th.einsum('...ptd, ...td -> ...pt', p2, p0)
    d21 = th.einsum('...ptd, ...td -> ...pt', p2, p1)

    denom = d00 * d11 - d01 * d01 + 1e-18  # T
    denom = denom[..., :, None, :]
    w1 = (d11[..., None, :] * d20 - d01[..., None, :] * d21) / denom
    w2 = (d00[..., None, :] * d21 - d01[..., None, :] * d20) / denom
    w0 = 1.0 - w1 - w2
    return th.stack([w0, w1, w2], dim=-1)


@th.jit.script
def is_inside_triangle(p: th.Tensor, tri: th.Tensor,
                       eps: float = 1e-6):
    """
    Arg:
        x: Point Cloud of shape (..., P, 3)
        t: Triangles of shape (..., T, 3, 3)

    Return:
        in: Whether each point is inside each triangle; (..., P, T)
    """
    c = barycentric_coords(p, tri, eps)
    out = ((0 <= c) & (c <= 1)).all(dim=-1)
    return out


@th.jit.script
def is_inside_barycentric_coords_fused(
        p: th.Tensor, tri: th.Tensor):
    v0, v1, v2 = th.unbind(tri, dim=-2)  # T3
    p0 = v1 - v0  # T3
    p1 = v2 - v0  # T3
    p2 = p[..., :, None, :] - v0[..., None, :, :]  # PT3

    d00 = th.einsum('...d, ...d -> ...', p0, p0)
    d01 = th.einsum('...d, ...d -> ...', p0, p1)
    d11 = th.einsum('...d, ...d -> ...', p1, p1)
    d20 = th.einsum('...ptd, ...td -> ...pt', p2, p0)
    d21 = th.einsum('...ptd, ...td -> ...pt', p2, p1)

    denom = d00 * d11 - d01 * d01
    denom = denom[..., :, None, :]
    k1 = (d11[..., None, :] * d20 - d01[..., None, :] * d21)
    k2 = (d00[..., None, :] * d21 - d01[..., None, :] * d20)

    return (
        (0 <= k1) &
        (0 <= k2) &
        (k1 + k2 <= denom)
    )


@th.jit.script
def point_triangle_distance(x: th.Tensor,
                            tri: th.Tensor,
                            eps: float = 1e-8):
    """
    Arg:
        x: Point Cloud of shape (..., P, 3)
        tri: Triangles of shape (..., T, 3, 3)

    Return:
        d: _squared_ distance between points and triangles.
    """
    v0, v1, v2 = tri.unbind(dim=-2)  # T3
    n = th.linalg.cross(v2 - v0, v1 - v0)  # Can be cached if needed
    mag_n = th.linalg.norm(n, dim=-1, keepdim=True) + eps
    # n.div_(mag_n)
    n = n.div(mag_n)

    # LAYOUT: (..., P, T, 3)
    v0x = v0[..., None, :, :] - x[..., :, None, :]
    t = th.einsum('...ptd, ...td -> ...pt', v0x, n)

    inside = is_inside_barycentric_coords_fused(x, tri)
    edge_indices = th.as_tensor([0, 1, 1, 2, 2, 0],
                                dtype=th.long,
                                device=tri.device)
    lines = th.index_select(tri, -2, edge_indices)  # 6 3
    lines = lines.reshape(tri.shape[:-3] + (-1, 2, 3))
    pld = point_line_distance(x, lines, eps)
    pld = pld.reshape(pld.shape[:-1] + (-1, 3)).amin(dim=-1)
    d = th.where(inside & (mag_n.swapaxes(-1, -2) > eps),
                 th.square(t),
                 pld)
    return d
