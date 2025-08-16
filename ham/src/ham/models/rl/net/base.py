#!/usr/bin/env python3

from abc import ABC, abstractproperty
from typing import Tuple, Union
from dataclasses import dataclass
from ham.util.config import ConfigBase


class FeatureBase(ABC):
    @dataclass
    class FeatureBaseConfig(ConfigBase):
        dim_in: Tuple[int, ...] = ()
        dim_out: int = -1
    Config = FeatureBaseConfig


class AggregatorBase(ABC):
    @dataclass
    class Config(ConfigBase):
        dim_obs: Tuple[int, ...] = ()
        dim_act: Tuple[int, ...] = ()
        dim_out: int = -1

    @abstractproperty
    def dim_out(self) -> int:
        # return self.cfg.dim_out
        pass


class FuserBase(ABC):
    @dataclass
    class Config(ConfigBase):
        # dim_in: Tuple[int, ...] = ()
        dim_out: int = -1
