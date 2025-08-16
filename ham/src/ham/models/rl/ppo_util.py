#!/usr/bin/env python3

from typing import Dict, Optional

import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import einops

from gym import spaces

from ham.util.torch_util import (
    masked_mean, masked_var_mean)
from ham.models.rl.ppo_config import DomainConfig
from ham.models.rl.util import explained_variance


def ornstein_uhlenbeck_noise(
        x: th.Tensor,
        scale: th.Tensor,
        theta: float = 0.15,  # might be too small
        dt: float = 1e-2,
        in_place: bool = True) -> th.Tensor:
    if not in_place:
        x = x.clone()
    x += (-theta * dt) * x + scale * np.sqrt(dt) * th.randn_like(x)
    return x


def gae_sb3(rewd: th.Tensor,
            val0: th.Tensor,
            val1: th.Tensor,
            end0: th.Tensor,
            end1: th.Tensor,
            gamma: float,
            lmbda: float):
    """ GAE from SB3 """
    # TD target
    nend1f = (1.0 - end1.float())
    delta = rewd + gamma * val1 * nend1f - val0
    T: int = delta.shape[1]

    advantage = th.zeros_like(rewd)
    advantage[:, T - 1] = delta[:, T - 1]
    for step in range(T - 2, -1, -1):
        advantage[:, step] = ((gamma * lmbda) * nend1f[:, step - 1]
                              * advantage[:, step + 1] + delta[:, step - 1])
    return advantage


# @th.jit.script
def gae_rlg(rewd: th.Tensor,
            val0: th.Tensor,
            val1: th.Tensor,
            end0: th.Tensor,
            end1: th.Tensor,
            gamma: float,
            lmbda: float):
    T: int = rewd.shape[1]
    advantage = th.zeros_like(rewd)
    nend1f = (1.0 - end1.float())
    delta = rewd + gamma * val1 * nend1f - val0
    advantage[:, T - 1] = delta[:, T - 1]
    for t in range(T - 2, -1, -1):
        advantage[:, t] = delta[:, t] + (
            gamma * lmbda * nend1f[:, t] * advantage[:, t + 1])
    return advantage


# @th.compile
@th.jit.script
def gae_ax1(
        rewd: th.Tensor,
        val0: th.Tensor,
        val1: th.Tensor,
        end0: th.Tensor,
        end1: th.Tensor,
        gamma: float,
        lmbda: float,
        sigma: float) -> th.Tensor:
    """
    Generalized advantage estimation.
    In this special case, assumes that
    data is arranged as (N, T, ...)
    """
    # TD target
    nend1f = (1.0 - end1.float())
    target = rewd + gamma * val1 * nend1f
    delta = target - val0
    if sigma > 0:
        delta /= sigma

    # if we're doing return-based scaling then:
    # sigma = var(R) + var(gamma) * var(Return)
    # sigma_hat = max(var_stat(R), 1e-2, rewd.var())
    # delta = delta / sigma_hat

    # delta = rewd + gamma * val1 * nend1f - val0
    # end0f = end0.float()
    # nend0f = (1.0 - end0.float())
    advantage = th.zeros_like(rewd)
    T: int = delta.shape[1]
    # adv = th.zeros_like(rewd[:, 0])
    # advantage[:, T - 1] = nend0f[:, T - 1] * delta[:, T - 1]
    advantage[:, T - 1] = delta[:, T - 1]
    for step in range(T - 2, -1, -1):
        # adv[...] = (nend0f[:, step] *
        #             ((gamma * lmbda) * nend1f[:, step] * adv + delta[:, step]))
        # advantage[:, step] = adv
        # advantage[:, step] = (
        #     nend0f[:, step] * (
        #         (gamma * lmbda) * nend1f[:, step] * advantage[:, step + 1]
        #         + delta[:, step])
        # )

        advantage[:, step] = (
            (gamma * lmbda) * nend1f[:, step] * advantage[:, step + 1]
            + delta[:, step]
        )
    return advantage


# @th.jit.script
def smooth_clamp(x: th.Tensor, mi: float, mx: float) -> th.Tensor:
    """ smooth clamping with gradients; from RLGames """
    return 1 / (1 + th.exp((-(x - mi) / (mx - mi) + 0.5) * 4)) * (mx - mi) + mi


class BoundLoss(nn.Module):
    def __init__(self, lo: th.Tensor, hi: th.Tensor):
        super().__init__()
        self.register_buffer('lo',
                             th.as_tensor(lo, dtype=th.float),
                             persistent=False)
        self.register_buffer('hi',
                             th.as_tensor(hi, dtype=th.float),
                             persistent=False)

    def forward(self, mu: th.Tensor,
                mask: Optional[th.Tensor] = None) -> th.Tensor:
        if mu is None:
            return 0.0

        # Non-zero loss if `mu` above max bounds
        b_loss_hi = th.square(th.clamp_min(
            mu - self.hi,
            0.0))
        # Non-zero loss if `mu` below min bounds.
        b_loss_lo = th.square(th.clamp_min(
            self.lo - mu,
            0.0))
        if mask is not None:
            b_loss = masked_mean((b_loss_hi + b_loss_lo),
                                 mask[..., None])
        else:
            b_loss = th.mean((b_loss_hi + b_loss_lo))
        return b_loss


class CBoundLoss(nn.Module):
    def __init__(self, wrap_act, max_dx):
        super().__init__()
        self._wrap_act = wrap_act
        self._max_dx = max_dx

    def forward(self,
                mu: th.Tensor,
                jac: th.Tensor,
                mask: Optional[th.Tensor] = None,
                ) -> th.Tensor:
        if mu is None:
            return 0.0
        dq = self._wrap_act(mu)
        jac = einops.rearrange(jac,
                               '... (x q) -> ... x q',
                               x=6, q=7)
        dx = th.linalg.norm(th.einsum('... x q, ... q -> ... x',
                                      jac[..., :3, :], dq[..., :7]), dim=-1)
        b_loss = th.square(th.clamp_min(dx - self._max_dx, 0.0))
        if mask is not None:
            b_loss = masked_mean(b_loss, mask)
        else:
            b_loss = th.mean(b_loss)
        return b_loss


def get_bound_loss(
        action_space: spaces.Space,
        domain_cfg: DomainConfig,
        k_bound: float,
        scale: float = 1.1
) -> Optional[BoundLoss]:
    # Automatically compute or query action bounds.
    bound = None
    if isinstance(action_space, spaces.Box):
        bound = (action_space.low,
                 action_space.high)
    elif domain_cfg.clip_action is not None:
        bound = domain_cfg.clip_action

    # Check if bound loss should be enabled.
    use_bound_loss = (
        (k_bound > 0) and
        (bound is not None)
    )
    if use_bound_loss:
        bound_loss = BoundLoss(
            th.as_tensor(domain_cfg.clip_action[0], dtype=th.float) * scale,
            th.as_tensor(domain_cfg.clip_action[1], dtype=th.float) * scale
        )
    else:
        bound_loss = None
    return bound_loss


def get_cbound_loss(
        wrap_act,
        max_dx: float):
    return CBoundLoss(wrap_act, max_dx)

class EnergyLoss(nn.Module):
    def __init__(self, wrap_act,
                 wrap_obs):
        super().__init__()
        self._wrap_act = wrap_act
        self._wrap_obs = wrap_obs

    def forward(self,
                mu: th.Tensor,
                qs: th.Tensor,
                mask: Optional[th.Tensor] = None,
                ) -> th.Tensor:
        if mu is None:
            return 0.0
        a = self._wrap_act(mu)
        dq = a[..., :7]
        kp = a[..., 7:14]
        rho = a[..., 14:21]
        kd = th.sqrt(kp) * rho
        with th.no_grad():
            q_state = self._wrap_obs(qs)['robot_state']
        v = q_state[..., 1::2]

        tau = kp *dq - kd * v
        power = th.norm(tau * v, dim=-1)
        
        if mask is not None:
            b_loss = masked_mean(power, mask)
        else:
            b_loss = th.mean(power)
        return b_loss

def get_energy_loss(
        wrap_act,
        wrap_obs):
    return EnergyLoss(wrap_act, wrap_obs)

# @th.jit.script


def _ppo_loss_and_logs_old(
    action: th.Tensor,
    adv: th.Tensor,
    logp0: th.Tensor,
    logp1: th.Tensor,
    ret: th.Tensor,
    val: th.Tensor,
    msk: th.Tensor,
    old_val: Optional[th.Tensor],
    normalize_adv: bool,
    adv_eps: float,
    clip: float,
    max_dv: Optional[float],
    env_type: Optional[th.Tensor]
):
    # OUTPUTS
    logs = {}
    losses = {}

    # == ADVANTAGE ==
    if normalize_adv:
        with th.no_grad():
            if env_type is None:
                adv_var, adv_mean = masked_var_mean(
                    adv.ravel(),
                    msk.ravel())
                print('(old)var,mean',
                      adv_var,
                      adv_mean)
            else:
                # per-type advantage normalization
                msk1 = msk[..., None]  # T x B x 1
                msk2 = F.one_hot(env_type)[None]  # 1xBxD
                msk3 = th.logical_and(msk1, msk2)  # TxBxD
                advm = adv[..., None].expand_as(msk3)
                adv_var, adv_mean = masked_var_mean(
                    advm.reshape(-1, advm.shape[-1]),
                    msk3.reshape(-1, msk3.shape[-1]),
                    dim=0)  # D
                adv_var = adv_var[None, env_type]  # 1,B
                adv_mean = adv_mean[None, env_type]  # 1,B
            adv_std = th.nan_to_num_(th.sqrt(adv_var))
            adv_mean = th.nan_to_num_(adv_mean)
            adv.sub_(adv_mean).div_(adv_std + adv_eps)

    # == POLICY ==
    log_ratio = (logp1 - logp0)
    rm1 = th.expm1(log_ratio)
    surr1_m_adv = rm1 * adv
    surr2_m_adv = th.clamp(rm1, -clip, clip) * adv
    policy_loss = -(adv + th.minimum(surr1_m_adv, surr2_m_adv))
    print('(old)',
          adv,
          msk,
          rm1,
          surr1_m_adv,
          surr2_m_adv,
          policy_loss)
    policy_loss = masked_mean(policy_loss, msk)
    losses['policy'] = policy_loss

    # == VALUE ==
    value_loss = th.square(val - ret)
    if (max_dv is not None) and (old_val is not None):
        val_clipped = (
            old_val + th.clamp(val - old_val, -max_dv, max_dv)
        )
        value_loss_clipped = th.square(val_clipped - ret)
        # NOTE(ycho): shouldn't this be `min`?
        with th.no_grad():
            logs['vclip_frac'] = masked_mean(
                (value_loss > value_loss_clipped).float(), msk)
        value_loss = th.maximum(value_loss, value_loss_clipped)
    value_loss = 0.5 * masked_mean(value_loss, msk)
    losses['value'] = value_loss

    # == ENTROPY ==
    approx_ent = -masked_mean(logp1, msk)
    ent_loss = -approx_ent
    losses['ent'] = ent_loss

    # == EXTRA LOGS ==
    with th.no_grad():
        approx_kl = masked_mean(rm1 - log_ratio, msk)
        explained_var = explained_variance(val, ret)
        clip_mask = th.abs(rm1) > clip
        clip_frac = masked_mean(clip_mask.float(), msk)

        logs['approx_kl'] = (approx_kl.mean().detach())
        logs['approx_ent'] = (approx_ent.mean().detach())
        logs['explained_var'] = (explained_var.mean().detach())
        logs['clip_frac'] = (clip_frac.mean().detach())
        logs['avg_val'] = (val.mean().detach())
        logs['avg_ret'] = (ret.mean().detach())
        logs['std_val'] = (val.std().detach())
        logs['std_ret'] = (ret.std().detach())

    return (losses, logs)


# @th.jit.script
def _ppo_loss_and_logs(
    action: th.Tensor,
    adv: th.Tensor,
    logp0: th.Tensor,
    logp1: th.Tensor,
    ret: th.Tensor,
    val: th.Tensor,
    msk: th.Tensor,
    old_val: Optional[th.Tensor],
    normalize_adv: bool,
    adv_eps: float,
    clip: float,
    max_dv: Optional[float],
    env_type: Optional[th.Tensor]
):
    SUM_REW: bool = True
    # REWD_WEIGHT: th.Tensor = th.as_tensor(
    #     [1.0, 0.15, 0.03, 0.005, 0.005, 0.0],
    #     dtype=th.float32,
    #     device=ret.device
    # )
    # OUTPUTS
    logs = {}
    losses = {}

    # == ADVANTAGE ==
    if normalize_adv:
        with th.no_grad():
            if env_type is None:
                # We make a strong-ish assumption
                # that layout of adv = (..., R) for num_rew
                # appended at the back
                adv_var, adv_mean = masked_var_mean(
                    adv.reshape(-1, adv.shape[-1]),
                    msk.reshape(-1, 1)
                )
                # print('(new)var,mean',
                #       adv_var,
                #       adv_mean)
            else:
                # per-type advantage normalization
                msk1 = msk[..., None]  # T x B x 1
                msk2 = F.one_hot(env_type)[None]  # 1xBxD
                msk3 = th.logical_and(msk1, msk2)  # TxBxD
                advm = adv[..., None].expand_as(msk3)
                adv_var, adv_mean = masked_var_mean(
                    advm.reshape(-1, advm.shape[-1]),
                    msk3.reshape(-1, msk3.shape[-1]),
                    dim=0)  # D
                adv_var = adv_var[None, env_type]  # 1,B
                adv_mean = adv_mean[None, env_type]  # 1,B

            if SUM_REW:
                # variance is computed on the entire return,
                # not for individual terms.
                adv_std = th.nan_to_num_(
                    th.sqrt(adv_var.sum(dim=-1, keepdim=True)))
            else:
                adv_std = th.nan_to_num_(th.sqrt(adv_var))
            adv_mean = th.nan_to_num_(adv_mean)
            adv.sub_(adv_mean).div_(adv_std + adv_eps)

    # == POLICY ==
    log_ratio = (logp1 - logp0)
    rm1 = th.expm1(log_ratio)
    surr1_m_adv = rm1[..., None] * adv
    surr2_m_adv = th.clamp(rm1[..., None], -clip, clip) * adv
    policy_loss = -(adv + th.minimum(surr1_m_adv, surr2_m_adv))
    # print('(new)',
    #       adv.squeeze(dim=-1),
    #       msk.squeeze(dim=-1),
    #       rm1.squeeze(dim=-1),
    #       surr1_m_adv.squeeze(dim=-1),
    #       surr2_m_adv.squeeze(dim=-1),
    #       policy_loss.squeeze(dim=-1))
    policy_loss = masked_mean(policy_loss, msk)
    losses['policy'] = policy_loss

    # == VALUE ==
    value_loss = th.square(val - ret)
    if (max_dv is not None) and (old_val is not None):
        val_clipped = (
            old_val + th.clamp(val - old_val, -max_dv, max_dv)
        )
        value_loss_clipped = th.square(val_clipped - ret)
        # NOTE(ycho): shouldn't this be `min`?
        with th.no_grad():
            logs['vclip_frac'] = masked_mean(
                (value_loss > value_loss_clipped).float(), msk)
        value_loss = th.maximum(value_loss, value_loss_clipped)
    value_loss = 0.5 * masked_mean(value_loss, msk)
    losses['value'] = value_loss

    # == ENTROPY ==
    approx_ent = -masked_mean(logp1, msk.squeeze(dim=-1))
    ent_loss = -approx_ent
    losses['ent'] = ent_loss

    # == EXTRA LOGS ==
    with th.no_grad():
        approx_kl = masked_mean(rm1 - log_ratio, msk.squeeze(dim=-1))
        explained_var = explained_variance(val, ret)
        clip_mask = th.abs(rm1) > clip
        clip_frac = masked_mean(clip_mask.float(), msk.squeeze(dim=-1))

        logs['approx_kl'] = (approx_kl.mean().detach())
        logs['approx_ent'] = (approx_ent.mean().detach())
        logs['explained_var'] = (explained_var.mean().detach())
        logs['clip_frac'] = (clip_frac.mean().detach())
        logs['avg_val'] = (val.mean().detach())
        logs['avg_ret'] = (ret.mean().detach())
        logs['std_val'] = (val.std().detach())
        logs['std_ret'] = (ret.std().detach())

    return (losses, logs)


class PPOLossAndLogs(nn.Module):
    """
    PPO loss and logs.

    loss:
        policy, value, ent
    logs:
        approx_kl, explained_var, clip_frac, avg_val, avg_ret, std_val, std_ret,


    """

    def __init__(self,
                 normalize_adv: bool,
                 adv_eps: float,
                 clip: float,
                 max_dv: float = 0.0,
                 ):
        super().__init__()
        self.normalize_adv: bool = normalize_adv
        self.adv_eps: float = adv_eps
        self.clip: float = clip
        self.max_dv: float = max_dv

    def forward(self,
                action: th.Tensor,
                adv: th.Tensor,
                logp0: th.Tensor,
                logp1: th.Tensor,
                ret: th.Tensor,
                val: th.Tensor,
                msk: th.Tensor,
                old_val: Optional[th.Tensor] = None,
                env_type: Optional[th.Tensor] = None,
                use_old: bool = False
                ):
        if use_old:
            return _ppo_loss_and_logs_old(action, adv,
                                          logp0, logp1, ret, val, msk,
                                          old_val,
                                          self.normalize_adv,
                                          self.adv_eps,
                                          self.clip,
                                          self.max_dv,
                                          env_type=env_type)
        return _ppo_loss_and_logs(action, adv,
                                  logp0, logp1, ret, val, msk,
                                  old_val,
                                  self.normalize_adv,
                                  self.adv_eps,
                                  self.clip,
                                  self.max_dv,
                                  env_type=env_type)
