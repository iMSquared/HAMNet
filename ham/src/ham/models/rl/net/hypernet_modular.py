#!/usr/bin/env python3

from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, fields, replace

import math
import pickle
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops
from functorch import make_functional_with_buffers
from icecream import ic

from ham.models.common import (
    merge_shapes,
    MLP,
    MHAWrapper,
    SplitDim,
    get_activation_function
)
from ham.models.rl.net.base import FeatureBase
from ham.models.rl.net.gate_mod_mlp import GateModMLP, MetaMLP


@dataclass
class ReduceConfig:
    """ Generalized "reduce" config for flattening observations. """
    type: str = 'None'  # Last, cross
    key_cross: Tuple[str, ...] = ()
    dim_local_proj: Tuple[int, ...] = ()
    attn_num_head: int = 8
    attn_emb_dim: int = 512
    nonlinear_tokenize: bool = False
    use_flash_attn: bool = True


class MHAReduce(nn.Module):
    """
    Reduce a set of tokens into a single vector via
    with multi-head cross attention.
    """
    Config = ReduceConfig

    def __init__(self,
                 cfg: Config,
                 embed_dim: int,
                 query_dim: int):
        super().__init__()
        self.cfg = cfg
        self.q_embed = (nn.Linear(query_dim, cfg.attn_emb_dim)
                        if not cfg.nonlinear_tokenize
                        else MLP(
                            (query_dim, cfg.attn_emb_dim),
                            nn.GELU,
                            True,
                            False,
                            use_ln=True))
        self.kv = (nn.Linear(embed_dim, cfg.attn_emb_dim)
                   if not cfg.nonlinear_tokenize
                   else MLP(
                            (embed_dim, cfg.attn_emb_dim),
            nn.GELU,
            True,
            False,
            use_ln=True))
        self.mhca = MHAWrapper(cfg.attn_emb_dim,
                               cfg.attn_num_head,
                               cross_attn=True,
                               use_flash_attn=cfg.use_flash_attn)
        dim_proj = merge_shapes(cfg.attn_emb_dim, cfg.dim_local_proj)
        self.proj = MLP(dim_proj,
                        nn.GELU,
                        use_bn=False,
                        use_ln=True)

    def forward(self,
                ctx: Dict[str, th.Tensor],
                x: th.Tensor):
        cfg = self.cfg
        # NOTE Get last for embeding
        q = th.cat([(ctx[kk][..., -1, :] if 'embed' in kk else ctx[kk])
                    for kk in cfg.key_cross], -1)
        with th.cuda.amp.autocast(cfg.use_flash_attn,
                                  th.float16):
            emb = self.mhca(self.q_embed(q[..., None, :]),
                            self.kv(x)).squeeze(dim=-2)
            emb = self.proj(emb)
        return emb


class NodeModulationNetwork(nn.Module):
    """
    Apply modulation on node-level logits at each layer.
    For a given layer, this means that the prediction needs to output
    m logits, where m is the number of modules.
    """

    def __init__(self,
                 dim_in: int,
                 num_param: int,
                 num_module: int,
                 hidden: Tuple[int, ...]):
        super().__init__()
        self.num_param = num_param
        self.num_module = num_module

        dims = merge_shapes(dim_in, hidden, num_param * num_module)
        self.logits = MLP(dims,
                          act_cls=nn.GELU,
                          use_bn=False,
                          use_ln=True)

    def forward(self, x: th.Tensor):
        y = self.logits(x).reshape(*x.shape[:-1],
                                   self.num_param,
                                   self.num_module)
        p = th.softmax(y, dim=-1)
        return p


class EdgeModulationNetwork(nn.Module):
    """
    Apply modulation on edge-level logits at each layer.
    For a given layer, this means that the prediction needs to output
    (m_y * m_x) logits, where m_y is the number of output modules,
    and m_x is the number of input modules.
    """

    def __init__(self,
                 dim_in: int,
                 num_param: int,
                 num_module: int,
                 hidden: Tuple[int, ...],
                 auto_regressive: bool = False,
                 emb_dim: int = 128):
        super().__init__()
        self.num_module = num_module
        self.auto_regressive = auto_regressive

        # NOTE: Assume actor and critic have same arch
        self.num_layer = num_param // 2

        if auto_regressive:
            # Here, previous logit predictions are also used as input
            dims = merge_shapes(dim_in, hidden, emb_dim)
            self.embed = MLP(dims,
                             act_cls=nn.GELU,
                             use_bn=False,
                             use_ln=True)

            dims = merge_shapes(emb_dim, emb_dim,
                                num_module * num_module)
            self.actor_p_embed = nn.ModuleList(
                [nn.Linear((i_l + 1) * num_module, emb_dim)
                 for i_l in range(self.num_layer - 1)])
            self.critic_p_embed = nn.ModuleList(
                [nn.Linear((i_l + 1) * num_module, emb_dim)
                 for i_l in range(self.num_layer - 1)])
            dims = merge_shapes(emb_dim, emb_dim,
                                num_module)
            actor_logits = [MLP(dims,
                            act_cls=nn.GELU,
                            use_bn=False,
                            use_ln=True)]
            critic_logits = [MLP(dims,
                                 act_cls=nn.GELU,
                                 use_bn=False,
                                 use_ln=True)]
            dims = merge_shapes(emb_dim, emb_dim,
                                num_module * num_module)
            actor_logits += [MLP(dims,
                                 act_cls=nn.GELU, use_bn=False, use_ln=True)
                             for _ in range(self.num_layer - 1)]
            critic_logits += [MLP(dims,
                                  act_cls=nn.GELU, use_bn=False, use_ln=True)
                              for _ in range(self.num_layer - 1)]

            self.actor_logits = nn.ModuleList(actor_logits)
            self.critic_logits = nn.ModuleList(critic_logits)
        else:
            dims = merge_shapes(dim_in, hidden)
            # bb = backbone*
            self.bb = MLP(dims,
                          act_cls=nn.GELU,
                          activate_output=True,
                          use_bn=False,
                          use_ln=True)

            dims = merge_shapes(hidden[-1], num_module)
            # NOTE Assume actor&critic nets have the same number of layers.
            self.actor_logit_first = MLP(dims,
                                         act_cls=nn.GELU,
                                         use_bn=False,
                                         use_ln=True)
            self.critic_logit_first = MLP(dims,
                                          act_cls=nn.GELU,
                                          use_bn=False,
                                          use_ln=True)
            dims = merge_shapes(hidden[-1], num_module * num_module
                                * (self.num_layer - 1))
            self.actor_logits = MLP(dims,
                                    act_cls=nn.GELU,
                                    use_bn=False,
                                    use_ln=True)
            self.critic_logits = MLP(dims,
                                     act_cls=nn.GELU,
                                     use_bn=False,
                                     use_ln=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        s = x.shape
        if self.auto_regressive:
            actor_p = []
            critic_p = []
            emb = self.embed(x)
            al = self.actor_logits[0](emb)  # (B, N)
            cl = self.critic_logits[0](emb)  # (B, N)
            actor_p.append(F.softmax(al, dim=-1))
            critic_p.append(F.softmax(cl, dim=-1))
            for idx, (a, c) in enumerate(
                zip(self.actor_logits[1:],
                    self.critic_logits[1:])):
                actor_cond = self.actor_p_embed[idx](th.cat(actor_p, -1))
                aw = a(emb * actor_cond).reshape(*s[:-1],
                                                 self.num_module,
                                                 self.num_module)  # (B, N, N)
                al = th.einsum('...ij,...j -> ...i', aw, al)
                actor_p.append(F.softmax(al, dim=-1))
                critic_cond = self.critic_p_embed[idx](th.cat(critic_p, -1))
                cw = c(emb * critic_cond).reshape(*s[:-1],
                                                  self.num_module,
                                                  self.num_module)  # (B, N, N)
                cl = th.einsum('...ij,...j -> ...i', cw, cl)
                critic_p.append(F.softmax(cl, dim=-1))
            p = th.stack(actor_p + critic_p,
                         dim=-2)  # (B, 2L = (num_param), N)
        else:
            z = self.bb(x)
            actor_first = self.actor_logit_first(z)  # (B, N)
            # (B, L-1, N, N)
            actor_remain = self.actor_logits(z).reshape(*s[:-1],
                                                        self.num_layer - 1,
                                                        self.num_module,
                                                        self.num_module)
            critic_first = self.critic_logit_first(z)
            # (B, L-1, N, N)
            critic_remain = self.critic_logits(z).reshape(*s[:-1],
                                                          self.num_layer - 1,
                                                          self.num_module,
                                                          self.num_module)

            actor_logits = [actor_first]
            al = actor_first
            critic_logits = [critic_first]
            cl = critic_first
            for i_l in range(self.num_layer - 1):
                al = th.einsum('...ij,...j -> ...i',
                               actor_remain[:, i_l],
                               al)
                actor_logits.append(al)
                cl = th.einsum('...ij,...j -> ...i',
                               critic_remain[:, i_l],
                               cl)
                critic_logits.append(cl)
            logits = th.stack(actor_logits + critic_logits,
                              dim=-2)  # (B, 2L = (num_param), N)
            p = F.softmax(logits, dim=-1)

        return p


class NodeModulationNetworkWithGate(nn.Module):
    def __init__(self,
                 dim_in: int,
                 num_param: int,
                 num_module: int,
                 target_hiddens: Tuple[int, ...],
                 hidden: Tuple[int, ...],
                 init_std: float = 0.008,
                 fused: bool = False,
                 temperature: Optional[float] = None,
                 sharing: bool = True,
                 num_group: Optional[int] = None
                 ):
        super().__init__()
        self.num_param = num_param
        self.num_module = num_module
        self.fused = fused

        self.temperature = None
        if temperature is not None:
            self.temperature = temperature
        self.num_group = num_group

        dims = merge_shapes(dim_in, hidden)

        self.share_bb = sharing
        if self.share_bb:
            self.bb = MLP(dims,
                          act_cls=nn.GELU,
                          activate_output=True,
                          use_bn=False,
                          use_ln=True)
        else:
            self.routing_bb = MLP(dims,
                                  act_cls=nn.GELU,
                                  activate_output=True,
                                  use_bn=False,
                                  use_ln=True)
            self.gating_bb = MLP(dims,
                                 act_cls=nn.GELU,
                                 activate_output=True,
                                 use_bn=False,
                                 use_ln=True)
        self.logits = nn.Linear(hidden[-1], num_param * num_module)
        if fused:
            self.target_hiddens = target_hiddens

            if num_group is None:
                h_out = sum(target_hiddens)
                self.scale_header = nn.Linear(hidden[-1], h_out)
                self.split_header = SplitDim(self.target_hiddens)
            else:
                h_out = num_group * len(target_hiddens)
                self.scale_header = nn.Linear(hidden[-1], h_out)

            with th.no_grad():
                h = self.scale_header
                nn.init.uniform_(h.weight, -init_std, init_std)
                nn.init.zeros_(h.bias)
        else:
            self.scale_header = nn.ModuleList(
                [nn.Linear(hidden[-1],
                           t)
                 for t in target_hiddens])
            for h in self.scale_header:
                nn.init.uniform_(h.weight, -init_std, init_std)
                nn.init.zeros_(h.bias)

    def forward(self, x: th.Tensor):
        if self.share_bb:
            z = self.bb(x)
            z_gating = z
        else:
            z = self.routing_bb(x)
            z_gating = self.gating_bb(x)
        y = self.logits(z).reshape(*x.shape[:-1],
                                   self.num_param,
                                   self.num_module)
        if self.temperature is not None:
            p = th.softmax(y / self.temperature, dim=-1)
        else:
            p = th.softmax(y, dim=-1)
        # ic(p.std(dim=0).mean(dim=-1))
        if self.fused:
            scale = 1 + self.scale_header(z_gating)
            if self.num_group is None:
                scale = self.split_header(scale)
            else:
                scale = einops.rearrange(scale, '... (l g) -> ... l g',
                                         g=self.num_group)
                scale = th.unbind(scale, dim=-2)
                scale = [einops.repeat(x, '... g -> ... (g r)',
                                       r=d // self.num_group)
                         for (x, d) in zip(scale, self.target_hiddens)]
        else:
            scale = [(1 + h(z_gating)) for h in self.scale_header]
        return p, scale


class ActionValueSubnet(nn.Module):

    @dataclass
    class ActionValueSubnetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = ()
        # NOTE(ycho): center + logstd + value
        dim_out: int = 20 + 20 + 1

        action_dim: int = 20
        value_dim: int = 1
        squeeze: bool = True
        log_std_init: float = 0.0
        hidden: Tuple[int, ...] = (64, 128, 64,)
        act_cls: str = 'tanh'
        use_bn: bool = False
        use_ln: bool = True
        affine: bool = True
        output_ls: bool = True
        share_backbone: bool = False
        ff_index: Optional[int] = None
        b_scale: float = 0.001

        def __post_init__(self):
            if self.output_ls:
                self.dim_out = (self.action_dim * 2 +
                                self.value_dim)
            else:
                self.dim_out = (self.action_dim + self.value_dim)

    Config = ActionValueSubnetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        act_cls = get_activation_function(cfg.act_cls)
        self.bb = None
        if cfg.share_backbone:
            dim_bb = merge_shapes(cfg.dim_in,
                                  cfg.hidden)
            self.bb = MLP(dim_bb, act_cls, True, cfg.use_bn,
                          use_ln=cfg.use_ln, affine=cfg.affine)
            dims_actor = merge_shapes(cfg.hidden[-1],
                                      cfg.action_dim)
            self.action_center = MLP(dims_actor, act_cls, False, cfg.use_bn,
                                     use_ln=cfg.use_ln, affine=cfg.affine)
            self.value = None
            if cfg.value_dim > 0:
                dims_value = merge_shapes(cfg.hidden[-1],
                                          cfg.value_dim)
                self.value = MLP(dims_value, act_cls, False, cfg.use_bn,
                                 use_ln=cfg.use_ln, affine=cfg.affine)
        else:
            dims_actor = merge_shapes(cfg.dim_in,
                                      cfg.hidden,
                                      cfg.action_dim)
            self.action_center = MLP(dims_actor, act_cls, False, cfg.use_bn,
                                     use_ln=cfg.use_ln, affine=cfg.affine)
            if cfg.ff_index is not None:
                with th.no_grad():
                    nn.init.normal_(
                        self.action_center.model[cfg.ff_index].linear.weight,
                        std=cfg.b_scale / cfg.dim_in[0])
                    nn.init.uniform_(
                        self.action_center.model[cfg.ff_index].linear.bias, -
                        1.0, 1.0)

            self.value = None
            if cfg.value_dim > 0:
                dims_value = merge_shapes(cfg.dim_in,
                                          cfg.hidden,
                                          cfg.value_dim)
                self.value = MLP(dims_value, act_cls, False, cfg.use_bn,
                                 use_ln=cfg.use_ln, affine=cfg.affine)
        log_std = None
        if cfg.output_ls:
            log_std = nn.Parameter(
                th.full((cfg.action_dim,), cfg.log_std_init),
                requires_grad=True)
        self.register_parameter('log_std', log_std)

    def forward(self, state: th.Tensor,
                ctx: Optional[Dict[str, th.Tensor]] = None):
        """
        inputs:
            object pose -- 3+6
        """
        if self.bb is not None:
            state = self.bb(state)
        center = self.action_center(state)

        value = None
        if self.value is not None:
            value = self.value(state)
        outs = [center]

        if self.cfg.output_ls:
            logstd = self.log_std.expand(center.shape)
            outs.append(logstd)

        if value is not None:
            outs.append(value)
        return th.cat(outs, dim=-1)


class ModularHyperActionValueNet(nn.Module, FeatureBase):

    @dataclass(init=False)
    class ModularHyperActionValueNetConfig(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (8, 256)

        # Should be act_dimx2+1
        dim_out: int = 41

        subnet: ActionValueSubnet.Config = ActionValueSubnet.Config()

        # dimensions etc. for ...
        # `embed_cloud, phys_params`
        key_ctx: Tuple[str, ...] = ()
        dim_ctx: Tuple[int, ...] = ()
        reduce_ctx: Tuple[str, ...] = ()

        key_state: Tuple[str, ...] = ()
        dim_state: Tuple[int, ...] = ()
        reduce_state: Optional[Dict[str, ReduceConfig]] = None

        wnet_dims: Tuple[int, ...] = (256, 256)

        # number of modules per weight
        num_module: int = 4
        num_val_module: Optional[int] = None
        num_group: Optional[int] = None

        # == cross-attn ==
        cross_global: bool = False
        key_robot: str = 'robot_state'
        dim_robot: int = 14
        num_head: int = 2
        use_flash_attn: bool = True
        cross_local: bool = False
        key_local_cross: str = 'object_state'
        cross_big: bool = False
        cross_big_emb_dim: int = 256

        key_cross_task: Tuple[str, ...] = (
            'object_state', 'goal', 'hand_state')
        obj_attn_num_head: int = 8
        obj_attn_emb_dim: int = 512

        use_local_shape: bool = False
        dim_local_proj: Tuple[int, ...] = (256, 128)

        nested_ctx: Optional[Dict[str, List[str]]] = None
        sm_like_modulation: bool = False
        auto_regressive: bool = False

        # reduce_nested_ctx: Optional[Tuple[str, ...]] = None
        reduce_nested_weight: str = 'learned'  # or sum or mean?
        log_attn: bool = False
        log_weights: bool = False
        load_weights: Optional[str] = None

        # scaling with weight only without last layer as default
        # also work with bias / or with last layer
        scaling: bool = False
        scale_bias: bool = False
        scale_last: bool = False
        fuse_scale: bool = False
        gate_share_bb: bool = True

        temperature: Optional[float] = None

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            # Resolve ctx dim as the result of cross-attention.
            if (len(self.reduce_ctx) > 0) and ('embed_cloud' in self.key_ctx):
                if list(self.reduce_ctx)[list(self.key_ctx).index(
                        'embed_cloud')] == 'cross':
                    dim_ctx = list(self.dim_ctx)
                    dim_ctx[list(self.key_ctx).index(
                        'embed_cloud')] = self.obj_attn_emb_dim
                    self.dim_ctx = tuple(dim_ctx)
            if self.use_local_shape:
                dim_state = list(self.dim_state)
                dim_state[list(self.key_state).index(
                    'embed_cloud')] = self.dim_local_proj[-1]
                self.dim_state = tuple(dim_state)
            self.subnet = replace(self.subnet,
                                  dim_in=(sum(self.dim_state),),
                                  )
            self.dim_out = self.subnet.dim_out

    Config = ModularHyperActionValueNetConfig

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Generate & extract `num_module` x parameter sets with functorch.
        paramss = []
        for i in range(cfg.num_module):
            subnet_base = ActionValueSubnet(cfg.subnet)
            func, params, bufs = make_functional_with_buffers(subnet_base)
            params = nn.ParameterList(params)
            paramss.append(params)

        # FIXME(ycho): hard-coded logic to count the number of actor subnets
        num_actor_net = 0
        for k, v in subnet_base.named_parameters():
            if 'action_center' in k:
                num_actor_net += 1

        ps = [p for p in zip(*paramss)]

        if cfg.num_val_module is not None:
            stacked_params = [th.stack(p, dim=0) if (i < num_actor_net)
                              else th.stack(p[:cfg.num_val_module], dim=0)
                              for i, p in enumerate(ps)]
        else:
            stacked_params = [th.stack(p, dim=0) for i, p in enumerate(ps)]

        # num_params X num_modules X param_dims
        self.params = nn.ParameterList(stacked_params)
        self.bufs = bufs

        # Create parameter-less actor/critic meta-networks
        # that receive network weights as extra input.
        sub_cfg = cfg.subnet
        dims_actor = merge_shapes(sub_cfg.dim_in,
                                  sub_cfg.hidden,
                                  sub_cfg.action_dim)
        dims_value = merge_shapes(sub_cfg.dim_in,
                                  sub_cfg.hidden,
                                  sub_cfg.value_dim)
        if (cfg.num_module <= 1) and (not cfg.scaling):
            # "simpler" version, more commonly known as meta-MLP
            self.func_a = MetaMLP(dims_actor)
            self.func_v = MetaMLP(dims_value)
        else:
            self.func_a = GateModMLP(dims_actor,
                                     gate=cfg.scaling)
            self.func_v = GateModMLP(dims_value,
                                     gate=cfg.scaling)

        self.reduce_cfg = list(cfg.reduce_ctx)

        # Cross-attention config to embed global cloud tokens
        use_cross_global = (cfg.cross_global and
                            'embed_peripheral_cloud' in cfg.key_ctx)

        if use_cross_global:
            i_reduce = cfg.key_ctx.index('embed_peripheral_cloud')
            self.reduce_cfg[i_reduce] = 'cross'
            embed_dim = cfg.dim_ctx[i_reduce]
            if cfg.cross_big:
                big_emb_dim = cfg.cross_big_emb_dim
                query_dim = (
                    cfg.dim_robot + cfg.dim_state
                    [cfg.key_state.index(cfg.key_local_cross)])
                self.embed_robot = MLP(
                    (query_dim, big_emb_dim),
                    nn.GELU,
                    True,
                    False,
                    use_ln=True
                )
                self.kv_gr = MLP(
                    (embed_dim, big_emb_dim),
                    nn.GELU,
                    True,
                    False,
                    use_ln=True
                )
                dim_proj = merge_shapes(big_emb_dim, cfg.dim_local_proj)
                self.proj_gr = MLP(
                    dim_proj,
                    nn.GELU,
                    use_bn=False,
                    use_ln=True
                )
                embed_dim = big_emb_dim
            else:
                self.embed_robot = nn.Linear(cfg.dim_robot,
                                             embed_dim)
            self.head_gr = embed_dim // cfg.num_head
            self.mhca_gr = MHAWrapper(embed_dim,
                                      cfg.num_head,
                                      cross_attn=True,
                                      use_flash_attn=cfg.use_flash_attn)
            self._attn = None

        # Cross-attention config to embed local cloud tokens
        use_cross_local = (cfg.cross_local and
                           'embed_focal_cloud' in cfg.key_ctx)
        if use_cross_local:
            i_reduce = cfg.key_ctx.index('embed_focal_cloud')
            self.reduce_cfg[i_reduce] = 'cross'
            embed_dim = cfg.dim_ctx[i_reduce]
            dim_q = cfg.dim_state[cfg.key_state.index(cfg.key_local_cross)]
            if cfg.cross_big:
                big_emb_dim = cfg.cross_big_emb_dim
                query_dim = (cfg.dim_robot + dim_q)
                self.embed_local_query = MLP(
                    (query_dim, big_emb_dim),
                    nn.GELU,
                    True,
                    False,
                    use_ln=True
                )
                self.kv_lr = MLP(
                    (embed_dim, big_emb_dim),
                    nn.GELU,
                    True,
                    False,
                    use_ln=True
                )
                dim_proj = merge_shapes(big_emb_dim, cfg.dim_local_proj)
                self.proj_lr = MLP(
                    dim_proj,
                    nn.GELU,
                    use_bn=False,
                    use_ln=True
                )
                embed_dim = big_emb_dim
            else:
                self.embed_local_query = nn.Linear(dim_q,
                                                   embed_dim)
            self.mhca_lr = MHAWrapper(embed_dim,
                                      cfg.num_head,
                                      cross_attn=True,
                                      use_flash_attn=cfg.use_flash_attn)

        # Cross-attention config to embed object cloud tokens
        if 'embed_cloud' in cfg.key_ctx:
            if self.reduce_cfg[cfg.key_ctx.index(
                    'embed_cloud')] == 'cross':
                query_dim = 0
                embed_dim = cfg.dim_ctx[cfg.key_ctx.index(
                    'embed_peripheral_cloud')]
                for k in cfg.key_cross_task:
                    query_dim += cfg.dim_state[cfg.key_state.index(k)]
                self.embed_query = nn.Linear(query_dim, cfg.obj_attn_emb_dim)
                self.to_kv = nn.Linear(embed_dim, cfg.obj_attn_emb_dim)
                self.mhca_obj = MHAWrapper(cfg.obj_attn_emb_dim,
                                           cfg.obj_attn_num_head,
                                           cross_attn=True,
                                           use_flash_attn=cfg.use_flash_attn)

        if cfg.num_module > 1 or cfg.scaling:
            if cfg.sm_like_modulation:
                dim_ctx = [d * 2 if r == 'mean+max' else d
                           for r, d in zip(self.reduce_cfg, cfg.dim_ctx)]
                self.modulate = EdgeModulationNetwork(sum(dim_ctx),
                                                      len(params),
                                                      cfg.num_module,
                                                      cfg.wnet_dims,
                                                      cfg.auto_regressive)
            else:
                dim_ctx = [d * 2 if (r == 'mean+max' or r == 'mean+glb') else d
                           for r, d in zip(self.reduce_cfg, cfg.dim_ctx)]
                if not cfg.scaling:
                    self.modulate = NodeModulationNetwork(sum(dim_ctx),
                                                          len(params),
                                                          cfg.num_module,
                                                          cfg.wnet_dims)
                else:
                    if not cfg.scale_last:  # exclude last layer
                        scale_target = list(cfg.subnet.hidden) * 2
                    else:
                        scale_target = (
                            list(
                                merge_shapes(
                                    cfg.subnet.hidden,
                                    cfg.subnet.action_dim)) +
                            list(
                                merge_shapes(
                                    cfg.subnet.hidden,
                                    cfg.subnet.value_dim)))
                    if cfg.scale_bias:
                        # with bias
                        scale_target = [
                            s for s in scale_target for _ in (
                                0, 1)]
                    self.modulate = NodeModulationNetworkWithGate(
                        sum(dim_ctx),
                        len(params),
                        cfg.num_module,
                        scale_target,
                        cfg.wnet_dims,
                        fused=cfg.fuse_scale,
                        temperature=cfg.temperature,
                        sharing=cfg.gate_share_bb,
                        num_group=cfg.num_group)
                    # padding for last layer
                    self.register_buffer(
                        'last_actor_scale', th.ones(
                            cfg.subnet.action_dim))
                    self.register_buffer(
                        'last_critic_scale', th.ones(
                            cfg.subnet.value_dim))
                self._weights = None

        if cfg.reduce_state is None:
            # "Old" reduction style (hard-coded)
            if cfg.use_local_shape:
                if 'embed_cloud' in cfg.key_ctx:
                    assert (self.reduce_cfg[cfg.key_ctx.index(
                        'embed_cloud')] != 'cross')
                    embed_dim = cfg.dim_ctx[cfg.key_ctx.index('embed_cloud')]
                else:
                    embed_dim = cfg.dim_state[cfg.key_state.index(
                        'embed_focal_cloud')]

                query_dim = 0
                for k in cfg.key_cross_task:
                    query_dim += cfg.dim_state[cfg.key_state.index(k)]
                self.embed_query = nn.Linear(query_dim, cfg.obj_attn_emb_dim)
                self.to_kv = nn.Linear(embed_dim, cfg.obj_attn_emb_dim)
                self.mhca_obj = MHAWrapper(cfg.obj_attn_emb_dim,
                                           cfg.obj_attn_num_head,
                                           cross_attn=True,
                                           use_flash_attn=cfg.use_flash_attn)
                dim_proj = merge_shapes(
                    cfg.obj_attn_emb_dim, cfg.dim_local_proj)
                self.out_proj = MLP(dim_proj,
                                    nn.GELU,
                                    use_bn=False,
                                    use_ln=True)

        else:
            # "New" reduction style (generalized)...
            # (except `embed_Cloud` hardcoding)
            state_cross = {}
            embed_dim = cfg.dim_state[cfg.key_state.index('embed_cloud')]
            for k, v in cfg.reduce_state.items():
                if v.type == 'cross':
                    query_dim = 0
                    for kk in v.key_cross:
                        query_dim += cfg.dim_state[cfg.key_state.index(kk)]
                    state_cross[k] = MHAReduce(v,
                                               embed_dim,
                                               query_dim)
            self.__state_cross = nn.ModuleDict(state_cross)

    def embed_context(self, x: th.Tensor,
                      ctx: Dict[str, th.Tensor]) -> th.Tensor:
        cfg = self.cfg
        (len(ctx['embed_cloud'].shape) - 2)

        ctx = dict(ctx)
        assert (len(cfg.key_ctx) == len(self.reduce_cfg))
        # FIXME(ycho): hard-coded keys...!!
        for k, reduce in zip(cfg.key_ctx, self.reduce_cfg):
            if reduce == 'cross':
                if k == 'embed_peripheral_cloud':
                    with th.cuda.amp.autocast(cfg.use_flash_attn,
                                              th.float16):
                        if cfg.log_attn:
                            q = self.mhca_gr.Wq(self.embed_robot(
                                ctx[cfg.key_robot][..., None, :]))
                            kv = self.mhca_gr.Wkv(ctx[k])
                            q = einops.rearrange(
                                q, '... (h d) -> ... h d',
                                d=self.head_gr)
                            kv = einops.rearrange(
                                kv, '... (two h d) -> ... two h d',
                                two=2, d=self.head_gr)
                            ke, v = kv.unbind(dim=-3)
                            scale: float = 1.0 / math.sqrt(q.shape[-1])
                            dots = th.einsum(
                                '...qhd,...khd->...hqk', q, ke * scale)
                            self._attn = th.softmax(
                                dots, dim=-1, dtype=v.dtype)
                        kv = self.kv_gr(ctx[k]) if cfg.cross_big else ctx[k]
                        if cfg.cross_big:
                            qq = th.cat([ctx[cfg.key_robot],
                                        ctx[cfg.key_local_cross]],
                                        dim=-1)
                            q = self.embed_robot(qq[..., None, :])
                        else:
                            q = self.embed_robot(
                                ctx[cfg.key_robot][..., None, :])
                        ctx[k] = self.mhca_gr(q, kv).squeeze(dim=-2)
                    if cfg.cross_big:
                        ctx[k] = self.proj_gr(
                            ctx[k].to(ctx[cfg.key_robot].dtype))
                elif k == 'embed_cloud':
                    q = th.cat([ctx[kk] for kk in cfg.key_cross_task], -1)
                    with th.cuda.amp.autocast(cfg.use_flash_attn,
                                              th.float16):
                        ctx[k] = self.mhca_obj(
                            self.embed_query(q[..., None, :]),
                            self.to_kv(ctx[k])).squeeze(dim=-2)
                elif k == 'embed_focal_cloud':
                    with th.cuda.amp.autocast(cfg.use_flash_attn,
                                              th.float16):
                        kv = self.kv_lr(ctx[k]) if cfg.cross_big else ctx[k]
                        if cfg.cross_big:
                            qq = th.cat([ctx[cfg.key_robot],
                                        ctx[cfg.key_local_cross]],
                                        dim=-1)
                            q = self.embed_local_query(qq[..., None, :])
                        else:
                            q = self.embed_local_query(
                                ctx[cfg.key_local_cross][..., None, :])
                        ctx[k] = self.mhca_lr(q, kv).squeeze(dim=-2)
                    if cfg.cross_big:
                        ctx[k] = self.proj_lr(
                            ctx[k].to(ctx[cfg.key_local_cross].dtype))
            elif reduce == 'last':
                # last
                ctx[k] = ctx[k][..., -1, :]
            elif reduce.isdigit():
                # index
                ctx[k] = ctx[k][..., int(reduce), :]
            elif reduce == 'mean+max':  # ~P2V
                z_max = th.amax(ctx[k], dim=-2)
                z_avg = th.mean(ctx[k], dim=-2)
                # ic(z_max.shape, z_avg.shape)
                ctx[k] = th.cat([z_avg, z_max], dim=-1)
            elif reduce == 'mean+glb':
                z_avg = th.mean(ctx[k], dim=-2)
                z_glb = ctx[k][..., -1, :]
                # ic(z_max.shape, z_avg.shape)
                ctx[k] = th.cat([z_avg, z_glb], dim=-1)
            elif reduce == 'none':
                pass
            else:
                raise ValueError(F'Unknown reduce type = {reduce}')
        ctx_emb = th.cat([ctx[k] for k in cfg.key_ctx], dim=-1)
        return ctx_emb

    def get_network_modulation(self, x: th.Tensor, ctx: Dict[str, th.Tensor]):
        cfg = self.cfg
        NB = (len(ctx['embed_cloud'].shape) - 2)

        # Extract flat context.
        # FIXME(ycho):
        # this is an ugly hack, to
        # extract only the global embedding
        ctx = dict(ctx)
        ctx_emb = self.embed_context(x, ctx)
        ctx_emb_flat = ctx_emb.reshape(-1, *ctx_emb.shape[NB:])

        # Modulation: determine module selection weights.
        weights = None
        scales = None
        if cfg.num_module > 1 or cfg.scaling:
            scales = None
            if not cfg.scaling:
                weights = self.modulate(ctx_emb_flat)
            else:
                weights, scales = self.modulate(ctx_emb_flat)
                scales = list(scales)
            if cfg.log_weights:
                self._weights = weights.clone()

            if cfg.load_weights is not None:
                with open(cfg.load_weights, 'rb') as fp:
                    ws = pickle.load(fp)
                weights[...] = th.as_tensor(ws,
                                            dtype=weights.dtype,
                                            device=weights.device)

            # Flatten batch dimensions
            weights = weights.reshape(-1, *weights.shape[-2:])
            # (num_params x (batch...) x num_module)
            weights = weights.unbind(dim=-2)

        return ctx_emb, weights, scales

    def forward(self, x: th.Tensor,
                ctx: Dict[str, th.Tensor]) -> th.Tensor:
        cfg = self.cfg
        _, weights, scales = self.get_network_modulation(x, ctx)

        # Postprocess scales.
        cfg.scaling
        if cfg.scaling:
            if not cfg.scale_last:
                # no scaling for last layer
                num_scale = len(scales)
                num_padding = 2 if cfg.scale_bias else 1
                for _ in range(num_padding):
                    scales.insert(num_scale // 2, self.last_actor_scale[None])
                    scales.append(self.last_critic_scale[None])
            if not cfg.scale_bias:
                ss = []
                for s in scales:
                    ss.append(s)
                    ss.append(None)
            else:
                ss = []
                for p, s in zip(self.params, scales):
                    if len(p.shape) == 2:
                        # bias
                        ss.append(s)
                    else:
                        # weight
                        ss.append(s[..., None])

        if cfg.reduce_state is None:
            # Assume 1D state, except keys marked with `embed`
            # NOTE(ycho): Get last for embeding (hardcoded logic)
            s = [(ctx[k][..., -1, :] if 'embed' in k else ctx[k])
                 for k in cfg.key_state]
            if cfg.use_local_shape:
                q = th.cat([ctx[kk] for kk in cfg.key_cross_task], -1)
                with th.cuda.amp.autocast(cfg.use_flash_attn,
                                          th.float16):
                    emb = self.mhca_obj(
                        self.embed_query(q[..., None, :]),
                        self.to_kv(ctx['embed_cloud'])).squeeze(dim=-2)
                    emb = self.out_proj(emb)
                s[list(cfg.key_state).index('embed_cloud')] = emb
            s = th.cat(s, dim=-1)
        else:
            # "generalized" reduction.
            local_ctx = dict(ctx)
            for k in cfg.key_state:
                reduce = cfg.reduce_state[k]
                if reduce.type == 'cross':
                    local_ctx[k] = self.__state_cross[k](ctx, ctx[k])
                elif reduce.type == 'last':
                    # last
                    local_ctx[k] = ctx[k][..., -1, :]
                elif reduce.type.isdigit():
                    # index
                    local_ctx[k] = ctx[k][..., int(reduce), :]
                elif reduce.type == 'mean+max':  # ~P2V
                    z_max = th.amax(ctx[k], dim=-2)
                    z_avg = th.mean(ctx[k], dim=-2)
                    local_ctx[k] = th.cat([z_avg, z_max], dim=-1)
                elif reduce.type == 'mean+glb':
                    z_avg = th.mean(ctx[k], dim=-2)
                    z_glb = ctx[k][..., -1, :]
                    local_ctx[k] = th.cat([z_avg, z_glb], dim=-1)
                elif reduce.type == 'none':
                    pass
            s = th.cat([local_ctx[k] for k in cfg.key_state], dim=-1)

        NB = len(s.shape) - 1
        b = s.shape[:-1]
        s = s.reshape(-1, *s.shape[NB:])
        N: int = len(self.params)

        # NOTE(ycho): The below logic reinterprets
        # flattened dimension. In particular:
        #   :N//2 --> take "actor network" slices
        #   N//2: --> take "value network" slices
        #   ::2   --> take "weight" slices
        #   1::2  --> take "bias" slices
        if cfg.num_module > 1 or cfg.scaling:
            gate_W_act = ss[:N // 2:2] if cfg.scaling else None
            gate_b_act = ss[1:N // 2:2] if cfg.scaling else None
            act = self.func_a(s,
                              self.params[:N // 2:2],
                              self.params[1:N // 2:2],
                              weights[:N // 2:2],
                              weights[1:N // 2:2],
                              gate_W_act,
                              gate_b_act
                              )
            gate_W_val = ss[N // 2:N:2] if cfg.scaling else None
            gate_b_val = ss[N // 2 + 1:N:2] if cfg.scaling else None
            val = self.func_v(s,
                              self.params[N // 2:N:2],
                              self.params[N // 2 + 1:N:2],
                              weights[N // 2:N:2],
                              weights[N // 2 + 1:N:2],
                              gate_W_val,
                              gate_b_val
                              )
        else:
            act = self.func_a(s,
                              self.params[:N // 2:2],
                              self.params[1:N // 2:2])
            val = self.func_v(s,
                              self.params[N // 2:N:2],
                              self.params[N // 2 + 1:N:2]
                              )
        outputs = th.cat([act, val], dim=-1)
        # restore batch dimensions.
        outputs = outputs.reshape(*b, *outputs.shape[1:])
        return outputs


def main():

    from omegaconf import OmegaConf

    num_time: int = 37
    batch_size: int = 71
    num_query: int = 4
    device: str = 'cuda:0'
    x = None

    num_patch: int = 16
    embed_dim: int = 128
    obs = dict(
        embed_cloud=th.randn(
            (num_time, batch_size, num_patch + 1, embed_dim),
            device=device),
        embed_focal_cloud=th.randn(
            (num_time, batch_size, num_patch + 1, embed_dim),
            device=device),
        embed_peripheral_cloud=th.randn(
            (num_time, batch_size, num_patch + 1, embed_dim),
            device=device),
        object_state=th.randn((num_time, batch_size, 13),
                              device=device),
        hand_state=th.randn((num_time, batch_size, 9),
                            device=device),
        robot_state=th.randn((num_time, batch_size, 14),
                             device=device),
        goal=th.randn((num_time, batch_size, 9),
                      device=device),
        previous_action=th.randn((num_time, batch_size, 20),
                                 device=device),
        phys_params=th.randn((num_time, batch_size, 5),
                             device=device),
    )
    cfg = ModularHyperActionValueNet.Config(
        dim_in=(8, 128),
        dim_out=41,

        key_ctx=('embed_cloud',
                 'embed_focal_cloud',
                 'embed_peripheral_cloud', 'phys_params'),
        dim_ctx=(128, 128, 128, 5),
        reduce_ctx=('mean+max', 'mean+max', 'mean+max', 'none'),

        key_state=(
            'object_state',
            'goal',
            'previous_action',
            'hand_state',
            'robot_state'),
        dim_state=(13, 9, 20, 9, 14),

        cross_global=True,
    )
    OmegaConf.save(cfg, '/tmp/hnet.yaml')

    model = ModularHyperActionValueNet(cfg).to(device)
    ic(model)
    y = model(x, ctx=obs)
    ic(y.shape)  # 7,6,128
    ic(y.reshape(-1, y.shape[-1]).mean(dim=0))
    ic(y.reshape(-1, y.shape[-1]).std(dim=0))
    # ic(model)


if __name__ == '__main__':
    main()
