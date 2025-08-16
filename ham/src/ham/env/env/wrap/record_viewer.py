#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from pathlib import Path

from dataclasses import dataclass
from ham.util.config import ConfigBase

from ham.env.env.iface import EnvIface
from ham.env.env.wrap.base import WrapperEnv
from ham.util.path import ensure_directory


class RecordViewer(WrapperEnv):
    """
    Wrapper to record images from gym viewer.
    """

    @dataclass
    class Config(ConfigBase):
        record_dir: str = '/tmp/ham/record'
        record_reward: bool = True

    def __init__(self, cfg: Config, env: EnvIface):
        super().__init__(env)
        assert (self.viewer is not None)

        self._record_dir: Path = ensure_directory(
            cfg.record_dir)

        # Minimize the likelihood of colliding
        # with other properties...
        self._record_step: int = 0

    def step(self, *args, **kwds):
        obs, rew, done, info = super().step(*args, **kwds)

        filename: str = str(self._record_dir / F'{self._record_step:04d}.png')
        self.gym.write_viewer_image_to_file(
            self.viewer, filename)
        self._record_step += 1

        return (obs, rew, done, info)
