#!/usr/bin/env python3

import torch as th
import torch.nn as nn

from typing import (Tuple, Dict, Optional, Union, Mapping, List)
from dataclasses import dataclass, replace, InitVar
from ham.util.config import ConfigBase

from ham.util.config import recursive_replace_map
from ham.models.common import (merge_shapes, transfer, map_struct)
from ham.models.rl.ppo_config import DomainConfig
from ham.models.rl.net.base import (AggregatorBase, FeatureBase, FuserBase)
from ham.models.rl.net.mlp import (MLPFeatNet, MLPAggNet, MLPFuser)
from ham.models.rl.net.gru import (GRUAggNet,)
from ham.models.rl.net.no_op import (NoOpFeatNet, NoOpAggNet, CatFuser)
from ham.models.rl.net.hypernet_modular import ModularHyperActionValueNet

from icecream import ic
import nvtx

S = Union[int, Tuple[int, ...]]
T = th.Tensor
rrm = recursive_replace_map

FEAT_KEY: str = '__feat__'
STATE_KEY: str = '__state__'


class GenericStateEncoder(nn.Module):
    def __init__(self,
                 feature_encoders: Dict[str, nn.Module],
                 feature_aggregators: Dict[str, nn.Module],
                 feature_fuser: nn.Module,
                 state_aggregator: nn.Module):
        super().__init__()
        if isinstance(feature_encoders, Mapping):
            self.feature_encoders = nn.ModuleDict(
                feature_encoders)
        else:
            self.feature_encoders = feature_encoders

        if isinstance(feature_aggregators, Mapping):
            self.feature_aggregators = nn.ModuleDict(
                feature_aggregators)
        else:
            self.feature_aggregators = feature_aggregators

        self.feature_fuser = feature_fuser
        self.state_aggregator = state_aggregator

    def init_hidden(self, batch_shape: S,
                    *args, **kwds):
        # FIXME(ycho): introspection into cfg.dim_out
        # probably better to define @abstractproperty(dim_out)->S
        shapes = {}
        shapes[FEAT_KEY] = map_struct(
            self.feature_aggregators,
            lambda src, _: merge_shapes(batch_shape, src.cfg.dim_out),
            base_cls=AggregatorBase
        )
        shapes[STATE_KEY] = merge_shapes(batch_shape,
                                         self.state_aggregator.cfg.dim_out
                                         )
        return map_struct(shapes,
                          lambda src, _: th.zeros(src, *args, **kwds),
                          # FIXME(ycho): doesn't work for np.int* I guess
                          base_cls=(int, Tuple, List)
                          )

    def feature(self,
                observation: Dict[str, th.Tensor],
                ) -> Dict[str, th.Tensor]:
        out = {}
        with nvtx.annotate("feature"):
            # NOTE(ycho): observation _has to be_ a dictionary.
            for k, v in observation.items():
                with nvtx.annotate(k):
                    if k not in self.feature_encoders:
                        continue
                    encoder = self.feature_encoders[k]

                    # resolve encoder type
                    enc_type = encoder
                    if isinstance(encoder, nn.DataParallel):
                        enc_type = encoder.module

                    # FIXME(ycho):
                    # hardcoded classes that requires extra context!
                    if isinstance(enc_type, (ModularHyperActionValueNet,)):
                        out[k] = encoder(v, ctx=observation)
                    else:
                        # requires just the current key
                        out[k] = encoder(v)
        return out

    def state(self,
              hidden: Dict[str, th.Tensor],
              action: th.Tensor,
              feature: Dict[str, th.Tensor]):
        """ Compute state autoregressively. """

        # zip h, x
        hx = map_struct(
            hidden[FEAT_KEY],
            lambda h, x: (h, x),
            feature,
            base_cls=th.Tensor
        )
        new_feat = map_struct(
            self.feature_aggregators,
            lambda f, hx: f(hx[0], None, hx[1]),
            hx,
            base_cls=AggregatorBase
        )
        fused_feature = self.feature_fuser(new_feat)
        new_state = self.state_aggregator(
            hidden[STATE_KEY],
            action,
            fused_feature
        )
        new_hidden = {
            FEAT_KEY: new_feat,
            STATE_KEY: new_state
        }
        return (new_state, new_hidden)

    def state_seq(self,
                  feature: Dict[str, th.Tensor],
                  done: th.Tensor,
                  loop: bool = False
                  ):
        """
        Compute state directly from sequence.
        Requires that there's no feature-level aggregation.
        """
        new_feat = map_struct(
            self.feature_aggregators,
            lambda f, x: f(None, None, x),
            feature,
            base_cls=AggregatorBase
        )
        fused_feature = self.feature_fuser(new_feat)
        new_state = self.state_aggregator(
            None,
            None,
            fused_feature
        )
        return new_state

    def forward(self,
                hidden: Dict[str, th.Tensor],
                action: th.Tensor,
                observation: Dict[str, th.Tensor]):
        feature = self.feature(observation)
        out = self.state(hidden, action, feature)
        return out


class MLPStateEncoder(GenericStateEncoder):
    """
    MLP state encoder with dictionary observations.
    """
    @dataclass
    class Config(ConfigBase):
        # domain: DomainConfig = DomainConfig()
        feature: Optional[Dict[str, FeatureBase.Config]] = None
        aggregator: Optional[Dict[str, AggregatorBase.Config]] = None
        fuser: Optional[FuserBase.Config] = None
        state: Optional[AggregatorBase.Config] = None
        obs_space: InitVar[Union[int, Dict[str, int], None]] = None
        act_space: InitVar[Optional[int]] = None

        def __post_init__(self, obs_space, act_space):
            # apply `dim_in` from `obs_space`
            if obs_space is not None:
                def _map_feature(src, dst):
                    if src is not None:
                        dim_in = merge_shapes(src)
                    else:
                        dim_in = src
                    if isinstance(dst, FeatureBase.Config):
                        return replace(dst, dim_in=dim_in)
                    raise ValueError(F'Invalid cfg = {src}, {dst}')

                # NOTE(ycho): `List,Tuple \in base_cls`
                # indicates that two separate observations
                # e.g. (state, goal) cannot be passed in as an iterable.
                # Instead, use a dictionary: like
                # {"state:(shape), "goal":(shape)}.
                self.feature = map_struct(
                    obs_space,  # src
                    _map_feature,
                    self.feature,
                    base_cls=(int, List, Tuple)
                )

            def _map_gru(src, dst):
                # First figure out `dim_obs`
                if not isinstance(src, FeatureBase.Config):
                    return dst
                dim_obs = merge_shapes(src.dim_out)
                if not isinstance(dst, AggregatorBase.Config):
                    return dst
                # NOTE(ycho): we can't incorporate
                # action inputs during feature aggreation!
                return replace(dst, dim_obs=dim_obs, dim_act=[])

            self.aggregator = map_struct(
                self.feature,
                _map_gru,
                self.aggregator,
                base_cls=FeatureBase.Config
            )

            # Configure aggregator.
            if isinstance(self.aggregator, Mapping):
                def _get_input_dims(src, dst):
                    if src is None:
                        return merge_shapes(0)
                    else:
                        return merge_shapes(src.dim_out)
                input_dims = map_struct(
                    self.aggregator,
                    _get_input_dims,
                    base_cls=AggregatorBase.Config
                )
                fuser = replace(
                    self.fuser,
                    input_dims=input_dims
                )
                self.fuser = fuser
            if isinstance(self.state, AggregatorBase.Config):
                dim_act: int = ([] if act_space is None else act_space)
                self.state = replace(
                    self.state,
                    dim_obs=merge_shapes(self.fuser.dim_out),
                    dim_act=dim_act)

    @classmethod
    def from_domain(cls, domain_cfg: DomainConfig, **kwds):
        cfg = recursive_replace_map(cls.Config(domain=domain_cfg),
                                    kwds)
        return cls.from_config(cfg)

    @classmethod
    def from_config(cls, cfg: Config):
        def feat_cls_fn(c: FeatureBase.Config):
            # FIXME(ycho): subclasses need to be listed before their
            # base classes to avoid breaking this logic.
            if isinstance(c, MLPFeatNet.Config):
                return MLPFeatNet(c)
            if isinstance(c, ModularHyperActionValueNet.Config):
                return ModularHyperActionValueNet(c)
            elif isinstance(c, NoOpFeatNet.Config):
                return NoOpFeatNet(c)
            else:
                raise KeyError(F'Unknown feat_cls={c}')

        def agg_fn(c: AggregatorBase.Config):
            if isinstance(c, MLPAggNet.Config):
                return MLPAggNet(c)
            elif isinstance(c, GRUAggNet.Config):
                return GRUAggNet(c)
            elif isinstance(c, NoOpAggNet.Config):
                return NoOpAggNet(c)
            else:
                raise KeyError(F'Unknown cfg={c}')

        feature_encoders = map_struct(
            cfg.feature,
            lambda src, dst: feat_cls_fn(src),
            base_cls=FeatureBase.Config
        )

        feature_aggregators = map_struct(
            cfg.aggregator,
            lambda src, dst: agg_fn(src),
            base_cls=AggregatorBase.Config
        )

        def _get_input_dims(src, dst):
            return merge_shapes(src.dim_out)
        input_dims = map_struct(
            cfg.aggregator,
            _get_input_dims,
            base_cls=AggregatorBase.Config
        )
        if isinstance(cfg.fuser, MLPFuser.Config):
            feature_fuser = MLPFuser(cfg.fuser,
                                     input_dims=input_dims)
        elif isinstance(cfg.fuser, CatFuser.Config):
            feature_fuser = CatFuser(cfg.fuser,
                                     input_dims=input_dims)
        else:
            raise KeyError(F'Unknown cfg.fuser={cfg.fuser}')

        state_aggregator = agg_fn(cfg.state)

        return cls(
            feature_encoders,
            feature_aggregators,
            feature_fuser,
            state_aggregator
        )


def test_inference():
    from ham.env.random_env import RandomEnv
    domain_cfg = DomainConfig(
        num_env=8,
        obs_space={'a': 64, 'b': 32},
        num_act=6
    )
    state_encoder = MLPStateEncoder.from_domain(
        domain_cfg)
    ic(state_encoder)
    env = RandomEnv(domain_cfg)

    prev_hidden = state_encoder.init_hidden(
        batch_shape=domain_cfg.num_env)
    action = th.zeros((domain_cfg.num_env, domain_cfg.num_act),
                      device=env.device)
    obs = env.reset()
    curr_state, hidden = state_encoder(
        prev_hidden, action, obs)


def test_create_with_noop():
    domain_cfg = DomainConfig(
        num_env=8,
        obs_space={
            'object_state': 7,
            'cube_state': 7,
            'goal': 3
        },
        num_act=6
    )
    state_cfg = MLPStateEncoder.Config(
        domain=domain_cfg)
    state_cfg.mlp['goal'] = NoOpFeatNet.Config(
        dim_in=state_cfg.mlp['goal'].dim_in,
        dim_out=state_cfg.mlp['goal'].dim_out,
    )
    state_cfg.gru['goal'] = NoOpAggNet.Config(
        dim_obs=state_cfg.mlp['goal'].dim_out,
    )
    state_cfg.__post_init__()
    ic(state_cfg)

    state_net = MLPStateEncoder.from_config(state_cfg)
    ic(state_net)


def test_linear_agg():
    domain_cfg = DomainConfig(
        num_env=8,
        obs_space={
            'object_state': 7,
            'cube_state': 7,
            'goal': 3
        },
        num_act=6
    )

    ic('default')
    state_cfg = MLPStateEncoder.Config(
        domain=domain_cfg)
    ic(state_cfg)

    # state_cfg = recursive_replace_map(state_cfg, {
    #     'feat_agg_cls': 'mlp'}
    # )
    state_cfg = replace(state_cfg, feat_agg_cls='mlp')
    ic(state_cfg)


def test_override_defaults():
    domain_cfg = DomainConfig(
        num_env=8,
        obs_space={
            'object_state': 7,
            'cube_state': 7,
            'goal': 3
        },
        num_act=6
    )

    ic('default')
    state_cfg = MLPStateEncoder.Config(
        domain=domain_cfg)
    ic(state_cfg)

    ic('update default act_cls')
    state_cfg = recursive_replace_map(state_cfg, {
        'default_mlp.act_cls': 'tanh'}
    )
    ic(state_cfg)

    ic('manually update goal mlp')
    updated_mlp_cfg = dict(state_cfg.mlp)
    updated_mlp_cfg['goal'] = MLPFeatNet.Config(
        dim_in=3,
        dim_out=18,
        act_cls='relu'
    )
    state_cfg = replace(state_cfg, mlp=updated_mlp_cfg)
    ic(state_cfg)

    ic('update default act_cls again')
    state_cfg = recursive_replace_map(state_cfg, {
        'default_mlp.act_cls': 'relu'}
    )
    ic(state_cfg)

    ic('update default act_cls again')
    state_cfg = recursive_replace_map(state_cfg, {
        'default_mlp.act_cls': 'tanh'}
    )
    ic(state_cfg)


def test_transfer():
    a_cfg = DomainConfig(
        num_env=8,
        obs_space={'goal': 3, 'object_state': 7},
        num_act=6)
    a_encoder = MLPStateEncoder.from_domain(a_cfg)

    ab_cfg = DomainConfig(
        num_env=16,
        obs_space={'goal': 3, 'object_state': 7,
                   'robot_state': 7},
        num_act=6)
    ab_encoder = MLPStateEncoder.from_domain(ab_cfg)

    if True:
        source_dict = {k: v for (k, v) in a_encoder.state_dict().items()
                       if (
            'feature_encoders.goal' in k or
            'feature_encoders.object_state' in k
        )}
        ab_encoder.load_state_dict(
            source_dict,
            strict=False)

    if True:
        keys = transfer(
            ab_encoder,
            a_encoder.state_dict(),
            substrs=[
                'feature_encoders.goal',
                'feature_encoders.object_state',
                'feature_aggregators.goal',
                'feature_aggregators.object_state',
            ])
        print('== TRANSFER ==')
        print('missing keys', keys.missing_keys)
        print('unexpected keys', keys.unexpected_keys)
        print('==============')
