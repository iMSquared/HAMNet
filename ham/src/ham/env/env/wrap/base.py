#!/usr/bin/env python3

from abc import abstractproperty

from typing import Optional, Iterable, Callable, Tuple

from ham.env.env.iface import EnvIface

import torch as th
from gym import spaces
from functools import partial
# from icecream import ic
# from ham.models.common import map_tensor


def _create_new(old, new, key: str):
    return {key: new}


def _update_as_child(old, new, key: str):
    return {**old, key: new}


def _update_as_sibling(old, new, key: str):
    return {'raw': old, key: new}


def add_obs_field(
        old_space: spaces.Space,
        key: str,
        item: spaces.Space,
        key_old: str = 'raw'
) -> Tuple[spaces.Dict, Callable]:
    """
    add a field to an observation space.
    """
    if old_space is None:
        obs_space = spaces.Dict({key: item})
        update_fn = partial(_create_new, key=key)
        return obs_space, update_fn

    make_dict: bool = (not isinstance(old_space, spaces.Dict))
    if make_dict:
        # Create a new dictionary-based observation space.
        obs_space = spaces.Dict({
            key_old: old_space,
            key: item
        })
        update_fn = partial(_update_as_sibling, key=key)
    else:
        # Let's try to prevent nested mappings, if possible...
        obs_space = spaces.Dict({
            **old_space.spaces,
            key: item,
        })
        update_fn = partial(_update_as_child, key=key)

    return obs_space, update_fn


class WrapperEnv(EnvIface):
    def __init__(self, env: EnvIface):
        super().__init__()
        self.env = env

    @property
    def num_env(self):
        return self.env.num_env

    @property
    def num_rew(self):
        return self.env.num_rew

    @property
    def num_type(self):
        return self.env.num_type

    @property
    def device(self):
        return self.env.device

    @property
    def timeout(self):
        return self.env.timeout

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def setup(self):
        return self.env.setup()

    def reset(self):
        obs = self.env.reset()
        # ic(type(self), map_tensor(obs, lambda src,_ : src.shape))
        return obs

    def step(self, actions: th.Tensor):
        obs, rew, done, info = self.env.step(actions)
        # ic(type(self), map_tensor(obs, lambda src,_ : src.shape))
        return (obs, rew, done, info)

    # def pre_physics_step(self, actions: th.Tensor):
    #     return self.env.pre_physics_step(actions)

    # def physics_step(self):
    #     return self.env.physics_step()

    # def post_physics_step(self):
    #     return self.env.post_physics_step()

    def reset_indexed(self, indices: Optional[Iterable[int]] = None):
        return self.env.reset_indexed(indices)

    def apply_actions(self, actions: th.Tensor):
        """ (act) -> () """
        return self.env.apply_actions(actions)

    def compute_observations(self):
        """ () -> (obs) """
        return self.env.compute_observations()

    def compute_feedback(self):
        """ (obs,act) -> (rew,done,info) """
        return self.env.compute_feedback()

    def unwrap(self, target=None):
        if isinstance(self, target):
            return self
        env = self.env
        if (target is not None) and isinstance(env, target):
            return env
        if isinstance(env, WrapperEnv):
            return env.unwrap(target)
        return env

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)


class ObservationWrapper(WrapperEnv):
    def __init__(self, env: EnvIface,
                 obs_wrapper: Callable[[th.Tensor], th.Tensor]):
        super().__init__(env)
        self.env = env
        self.obs_wrapper = obs_wrapper

    @abstractproperty
    def observation_space(self):
        return

    @property
    def action_space(self):
        return self.env.action_space

    def setup(self):
        return self.env.setup()

    def reset(self):
        out = self.obs_wrapper(super().reset())
        # ic(type(self), map_tensor(out, lambda src,_ : src.shape))
        return out

    def reset_indexed(self, indices: Optional[th.Tensor]):
        return self.env.reset_indexed(indices)

    def step(self, actions: th.Tensor):
        obs, rew, done, info = super().step(actions)
        w_obs = self.obs_wrapper(obs)
        # ic(type(self), w_obs)
        return (w_obs, rew, done, info)

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)


class ActionWrapper(WrapperEnv):
    def __init__(self, env: EnvIface,
                 act_wrapper: Callable[[th.Tensor], th.Tensor]):
        super().__init__(env)
        self.env = env
        self.act_wrapper = act_wrapper

    @property
    def observation_space(self):
        return self.env.observation_space

    @abstractproperty
    def action_space(self):
        pass

    def setup(self):
        return self.env.setup()

    def reset(self):
        return self.env.reset()

    def reset_indexed(self, indices: Optional[th.Tensor]):
        return self.env.reset_indexed(indices)

    def step(self, actions: th.Tensor):
        w_actions = self.act_wrapper(actions)
        return super().step(w_actions)

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)
