#!/usr/bin/env python3

from abc import ABC, abstractproperty
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import inspect
import logging

import torch as th
import torch.nn as nn
from icecream import ic


class RewardBase(ABC):
    @abstractproperty
    def dim(self) -> int: ...


class SuccessReward(nn.Module, RewardBase):
    @dataclass
    class Config:
        pass

    @property
    def dim(self) -> int:
        return 1

    def forward(self, *, succ: th.Tensor, **kwds) -> th.Tensor:
        # NOTE(ycho): since the new API assumes that the
        # rewards are given as [num_env, num_rew] form,
        # we have to return a vectorized version of the rewards.
        return succ.float()[..., None]


class FailureReward(nn.Module, RewardBase):
    """ Add penalty per each timestep """
    @dataclass
    class Config:
        pass

    @property
    def dim(self) -> int:
        return 1

    def forward(self, *, succ: th.Tensor, done: th.Tensor,
                **kwds) -> th.Tensor:
        fail = (~succ & done)
        # NOTE(ycho): since the new API assumes that the
        # rewards are given as [num_env, num_rew] form,
        # we have to return a vectorized version of the rewards.
        return -fail.float()[..., None]


class TimestepReward(nn.Module, RewardBase):
    """ Add penalty per each timestep """
    @dataclass
    class Config:
        pass

    @property
    def dim(self) -> int:
        return 1

    def forward(self, *, done: th.Tensor, **kwds) -> th.Tensor:
        # NOTE(ycho): since the new API assumes that the
        # rewards are given as [num_env, num_rew] form,
        # we have to return a vectorized version of the rewards.
        return -th.ones_like(done, dtype=th.float32)[..., None]


class PotExpRew(nn.Module):
    @dataclass
    class Config:
        reduce: bool = True
        gamma: float = 0.999

    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.gamma >= 1.0:
            raise ValueError(F'cfg.gamma={cfg.gamma} >= 1.0')
        self.cfg = cfg

    def forward(self, *, dist: th.Tensor, **kwds):
        cfg = self.cfg
        pot = th.pow(cfg.gamma, cfg.k * dist)
        if cfg.reduce:
            pot = pot.mean(dim=-1)
        return (cfg.gamma * pot[1:] - pot[:-1])


class PotLogRew(nn.Module):
    @dataclass
    class Config:
        reduce: bool = True
        # gamma: float = 0.999
        gamma: float = 1.0
        k: float = 3.0

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, *, dist: th.Tensor, **kwds):
        """
        dist: [T, ...]
        * distance from the goal
        * arranged in terms of temporal dimension
        """
        cfg = self.cfg
        pot = -th.log1p(cfg.k * dist)
        if cfg.reduce:
            pot = pot.mean(dim=-1)
        return (cfg.gamma * pot[1:] - pot[:-1])


class ReachReward(nn.Module, RewardBase):
    """ map dist(obj, goal) -> rewd """

    @dataclass
    class Config:
        rewd_type: str = 'pot_log'  # log-type pot.rew
        pot_log: PotLogRew.Config = PotLogRew.Config()
        pot_exp: PotExpRew.Config = PotExpRew.Config()

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        if cfg.rewd_type == 'pot_log':
            self.rewd = PotLogRew(cfg.pot_log)
        elif cfg.rewd_type == 'pot_exp':
            self.rewd = PotExpRew(cfg.pot_exp)
        else:
            raise ValueError(F'Unknown rewd_type={cfg.rewd_type}')

    @property
    def dim(self) -> int:
        return 1

    def forward(self,
                *,
                dist: th.Tensor,
                **kwds):
        # you might ask, "what the hell is happening here?" well...
        # [1] the "path" is usually given as (2, ...) tensor
        # so we squeeze (dim=0) from [1:]-[:-1] artifact
        # [2] and then we add the [..., None]
        # to return vector-form rewards.
        rewd = self.rewd(dist=dist).squeeze(dim=0)[..., None]
        return rewd


class AddReward(nn.Module):
    """ composite reward: sum """
    @dataclass
    class Config:
        coef: Optional[Dict[str, float]] = None

    def __init__(self, cfg: Config, rewds: Dict[str, nn.Module]):
        super().__init__()
        self.cfg = cfg
        self.rewds = nn.ModuleDict({k: v for k, v in rewds.items()
                                    if k in cfg.coef})
        for k, v in rewds.items():
            if v.dim > 1:
                ic(
                    F'rewards for non-scalar reward {k}={v.dim} will be summed'
                )

    @property
    def dim(self) -> int:
        return 1

    def forward(self, *args, **kwds):
        cfg = self.cfg
        rewd = 0
        info = {}
        for k in self.rewds.keys():
            coef = kwds.get(F'add_rew_coef_{k}', cfg.coef[k])
            d = coef * self.rewds[k](*args, **kwds).sum(
                dim=-1, keepdim=True)
            info[k] = d
            rewd = rewd + d
        return rewd, info


class CatReward(nn.Module, RewardBase):
    """ composite reward: cat """
    @dataclass
    class Config:
        keys: Tuple[str, ...] = ()

    def __init__(self,
                 cfg: Config,
                 rewds: Dict[str, nn.Module]):
        super().__init__()
        self.cfg = cfg

        # Try to warn against "dupliacted" arguments
        names = {}
        for k, r in rewds.items():
            sig = inspect.signature(r.forward)
            for pname in sig.parameters.keys():
                p = sig.parameters[pname]
                if p.kind not in (inspect.Parameter.VAR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY):
                    raise ValueError(
                        F'Invalid non-kw-only function {k}: {sig}')
                if p.kind == inspect.Parameter.KEYWORD_ONLY:
                    if p.name in names:
                        ic(
                            '(CatReward) Duplicated keyword may clash:'
                            + F'"{p.name}" for {k} and {names[p.name]}'
                        )
                        names[p.name].append(k)
                    names[p.name] = [k]

        self.rewds = nn.ModuleDict({k: v for k, v in rewds.items()
                                    if k in cfg.keys})

    @property
    def dim(self) -> int:
        # NOTE(ycho):
        # can be quite hard to debug,
        # in case one of the rewards fails to define `dim`.
        return sum(v.dim for v in self.rewds.values())

    def forward(self, *args, **kwds):
        rewds = []
        info = {}
        for k in self.rewds.keys():
            d = self.rewds[k](*args, **kwds)
            rewds.append(d)
            # print('CatRewawrd', k, d.shape)
            info[k] = d
        rewds = th.cat(rewds, dim=-1)
        return rewds, info


class TaskReward(nn.Module, RewardBase):
    @dataclass
    class Config:
        dim: int = -1

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    @property
    def dim(self):
        return self.cfg.dim

    def forward(self, *, task: th.Tensor, **kwds) -> th.Tensor:
        return task


class SoftConstraintReward(nn.Module, RewardBase):

    @dataclass
    class Config:
        split: bool = True
        pot: bool = True
        add_count: bool = False

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.reward = ReachReward(ReachReward.Config(
            rewd_type='pot_log'
        ))

    @property
    def dim(self) -> int:
        return (2 if self.cfg.split else 1)

    def forward(self,
                *,
                hand_depth: th.Tensor,
                else_depth: th.Tensor,
                hand_count: Optional[th.Tensor] = None,
                else_count: Optional[th.Tensor] = None,
                **kwds
                ):
        cfg = self.cfg
        assert (cfg.split)
        if cfg.pot:
            # for now
            assert (hand_count is None)
            assert (else_count is None)
            hand_reward = self.reward(dist=-hand_depth)
            else_reward = self.reward(dist=-else_depth)
        else:
            assert (len(hand_depth) == 2)
            hand_reward = hand_depth[1]
            else_reward = else_depth[1]
            if cfg.add_count:
                hand_reward -= hand_count[1]
                else_reward -= else_count[1]
        return th.cat([hand_reward, else_reward], dim=-1)


class HardConstraintReward(nn.Module, RewardBase):

    @dataclass
    class Config:
        split: bool = True

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    @property
    def dim(self) -> int:
        return (2 if self.cfg.split else 1)

    def forward(self,
                *,
                task: th.Tensor,
                **kwds
                ):
        cfg = self.cfg
        assert (cfg.split)
        if cfg.split:
            return th.zeros_like(task[..., :1]).expand(
                *task.shape[:-1], 2).clone()
        else:
            return th.zeros_like(task[..., :1]).expand(
                *task.shape[:-1], 1).clone()


class EnergyRegularizingReward(nn.Module, RewardBase):
    @dataclass
    class Config:
        pass

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    @property
    def dim(self) -> int:
        return 1

    def forward(self, *, energy: th.Tensor,
                **kwds):
        return energy[..., None]
