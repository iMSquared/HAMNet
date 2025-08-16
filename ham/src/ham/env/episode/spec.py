#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple, Dict
import torch as th


class Spec(ABC):
    @abstractproperty
    def setup_keys(self) -> Tuple[str, ...]: ()

    @abstractproperty
    def setup_deps(self) -> Tuple[str, ...]: ()

    @abstractproperty
    def reset_keys(self) -> Tuple[str, ...]: ()

    @abstractproperty
    def reset_deps(self) -> Tuple[str, ...]: ()

    @abstractmethod
    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return data

    @abstractmethod
    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return data

    @abstractmethod
    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        return

    @abstractmethod
    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        return


class DefaultSpec(Spec):
    """
    Mostly here to provide default implementations, for convenience
    """
    @property
    def setup_keys(self) -> Tuple[str, ...]: return ()
    @property
    def setup_deps(self) -> Tuple[str, ...]: return ()
    @property
    def reset_keys(self) -> Tuple[str, ...]: return ()
    @property
    def reset_deps(self) -> Tuple[str, ...]: return ()

    def sample_setup(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return data

    def sample_reset(self,
                     ctx,
                     data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return data

    def apply_setup(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        return data

    def apply_reset(self,
                    ctx,
                    data: Dict[str, th.Tensor]):
        return data


class WrapSpec(DefaultSpec):
    """ Spec that wraps other specs.  """

    def __init__(self, specs):
        self.specs = specs

    def find(self, match_fn, recurse: bool = True):
        # NOTE(ycho):
        # the problem with relying on
        # `WrapSpec` is that
        # find() queries will _not_ work with
        # non-wrap specs...
        out = []
        if match_fn(self):
            out.append(self)
        for spec in self.specs:
            if isinstance(spec, WrapSpec) and recurse:
                out.extend(spec.find(match_fn, recurse))
                continue
            if match_fn(spec):
                out.append(spec)
        return out
