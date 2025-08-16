#!/usr/bin/env python3

from typing import Tuple, List, Optional
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract_expression
from collections import defaultdict


class MetaLinear(nn.Module):
    def __init__(self, d_i: int,
                 d_o: int,
                 act_cls='tanh',
                 norm_cls='layernorm'):
        super().__init__()
        self.d_i = d_i
        self.d_o = d_o
        if act_cls == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
        if norm_cls == 'layernorm':
            self.norm = nn.LayerNorm(d_o, elementwise_affine=False)
        else:
            self.norm = nn.Identity()

    def forward(self, x: th.Tensor,
                W: th.Tensor,
                b: th.Tensor):
        s = x.shape
        x = x.reshape(-1, s[-1])
        # FIXME(ycho): `squeeze` only applicable for us right now.
        y = self.act(self.norm(F.linear(x, W.squeeze(0), b.squeeze(0))))
        y = y.reshape(*s[:-1], y.shape[-1])
        return y


class MetaMLP(nn.Module):
    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        print(F'MetaMLP got dims = {dims}')
        num_layer: int = len(dims) - 1
        last_idx: int = num_layer - 1
        self.layers = nn.ModuleList([
            MetaLinear(d_i, d_o,
                       act_cls=('tanh' if (l != last_idx) else 'none'),
                       norm_cls='layernorm' if (l != last_idx) else 'none'
                       )
            for l, (d_i, d_o) in enumerate(zip(dims[:-1], dims[1:]))
        ])

    def forward(self,
                x: th.Tensor,
                # weight/bias (modules)
                # shape: Lx(Di,Do)
                Ws: List[th.Tensor],
                # shape: Lx(Do)
                bs: List[th.Tensor]):
        for layer, W, b in zip(self.layers, Ws, bs):
            x = layer(x, W, b)
        return x


class GateModLinear(nn.Module):
    def __init__(self,
                 d_i: int,
                 d_o: int,
                 act_cls='tanh',
                 norm_cls='layernorm',
                 gate: bool = True):
        super().__init__()
        self.d_i = d_i
        self.d_o = d_o
        if act_cls == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
        if norm_cls == 'layernorm':
            self.norm = nn.LayerNorm(d_o, elementwise_affine=False)
        else:
            self.norm = nn.Identity()

        # NOTE(ycho): rough guesses here are fine.
        b = 1024 * 10  # batch size (= num_env x len_rollout)
        i = d_o  # output dim
        m = 8  # num modules
        j = d_i  # input dim
        self.gate = gate
        if gate:
            self.f_Wx = contract_expression(
                'bi,bm,mij,bj->bi',
                (b, i),
                (b, m),
                (m, i, j),
                (b, j),
            )
            self.f_b = contract_expression(
                'bj,bm,mj->bj',
                (b, j),
                (b, m),
                (m, j),
            )
        else:
            self.f_Wx = contract_expression(
                'bm,mij,bj->bi',
                (b, m),
                (m, i, j),
                (b, j),
            )
            self.f_b = contract_expression(
                'bm,mj->bj',
                (b, m),
                (m, j)
            )

    def extra_repr(self):
        return F'{self.d_i}->{self.d_o}'

    def forward(self,
                x: th.Tensor,
                # weight/bias (modules)
                Ws: th.Tensor,
                bs: th.Tensor,

                # module selection probabilities
                pW: th.Tensor,
                pb: th.Tensor,

                # output gates
                gW: Optional[th.Tensor] = None,
                gb: Optional[th.Tensor] = None):
        if self.gate:
            Wx = self.f_Wx(gW, pW, Ws, x)
            b = self.f_b(gb, pb, bs)
        else:
            Wx = self.f_Wx(pW, Ws, x)
            b = self.f_b(pb, bs)
        return self.act(self.norm(Wx + b))


class GateModMLP(nn.Module):
    def __init__(self,
                 dims: Tuple[int, ...],
                 gate: bool = True):
        super().__init__()
        print(F'GateModMLP got dims = {dims}')
        num_layer: int = len(dims) - 1
        last_idx: int = num_layer - 1
        self.gate = gate
        self.layers = nn.ModuleList([
            GateModLinear(d_i, d_o,
                          # FIXME(ycho): hardcoded act_cls
                          act_cls='tanh' if (l != last_idx) else 'none',
                          norm_cls='layernorm' if (l != last_idx) else 'none',
                          gate=gate
                          )
            for l, (d_i, d_o) in enumerate(zip(dims[:-1], dims[1:]))
        ])

    def forward(self, x: th.Tensor,
                # weight/bias (modules)
                # shape: Lx(M,Di,Do)
                Ws: List[th.Tensor],
                # shape: Lx(M,Do)
                bs: List[th.Tensor],

                # module selection probabilities
                # shape: (...,L,M)
                pWs: th.Tensor,
                # shape: (...,L,M)
                pbs: th.Tensor,
                gWs: Optional[th.Tensor] = None,
                gbs: Optional[th.Tensor] = None):
        """
        Args:
            (inputs)
            x: network input -- (..., d_x)

            (parameters)
            Ws: module weights -- L * (M, d_y, d_x)
            bs: module biases -- L * (M, d_y)

            (predictions)
            pWs: weight activation coeffs -- L * (..., M)
            pbs: bias activation coeffs -- L * (..., M)
            gWs: weight gating coeffs -- L * (..., d_y)
            gbs: bias gating coeffs -- L * (..., d_y)

        Shapes:
            L   : number of layers.
            M   : number of modules.
            d_x : input dims.
            d_y : output dims.

        Return:
            y: network output -- (..., d_y)
            modulated prediction according to the given network parameters.
        """
        if self.gate:
            for layer, W, b, pW, pb, gW, gb in zip(self.layers,
                                                   Ws, bs, pWs, pbs, gWs, gbs):
                x = layer(x, W, b, pW, pb, gW.squeeze(dim=-1), gb)
        else:
            for layer, W, b, pW, pb in zip(self.layers, Ws, bs, pWs, pbs):
                x = layer(x, W, b, pW, pb)
        return x
