from typing import Tuple, Iterable, List, Optional, Dict, Union

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import torch_utils

import numpy as np
import torch as th
import torch.nn.functional as F
import einops
from scipy.spatial.transform import Rotation as R
import pkg_resources


import trimesh
from pytorch3d.ops import knn_points
from yourdfpy import URDF

from ham.util.math_util import (quat_from_axa, quat_inverse,
                                quat_multiply, quat_rotate,
                                matrix_from_quaternion,
                                apply_pose_tq,
                                matrix_from_pose
                                )
from ham.data.transforms.aff import get_link_mesh

from ham.env.util import (
    draw_cloud_with_sphere,
)

from pdb import set_trace as bp

import nvtx
from icecream import ic


def align_vectors(a, b, eps: float = 0.00001):
    """
    Return q: rotate(q, a) == b
    """
    dot = th.einsum('...j, ...j->...', a, b)
    parallel = (dot > (1 - eps))
    opposite = (dot < (-1 + eps))

    not_alligend = th.logical_not(th.logical_or(parallel, opposite))
    cross = th.cross(a, b, dim=-1)
    # sin(\theta) = 2 sin(0.5*theta) cos(0.5*theta)
    # 1 + cos(\theta) # = 2 cos^2(0.5*theta)
    out = th.cat([cross, (1 + dot)[..., None]], dim=-1)
    out /= (1e-6 + out.norm(p=2, dim=-1, keepdim=True))

    # Handle aligned cases.
    out[parallel] = th.as_tensor((0, 0, 0, 1),
                                 dtype=out.dtype,
                                 device=out.device)
    out[opposite] = th.as_tensor((1, 0, 0, 0),
                                 dtype=out.dtype,
                                 device=out.device)

    return out


def check_collision(ee_pose: th.Tensor, ee_cloud,
                    obj_pose: th.Tensor, obj_cloud,
                    brute: bool = False,
                    eps: float = 1e-3):
    """
    ee_pose:   [N,7]
    ee_cloud:  [M,3], M=number of points in ee mesh
    obj_pose:  [N,7]
    obj_cloud: [N,O,3], O=number of points in obj mesh
    brute: compute distance as min(NxM) instead of KNN lookup
    eps: contact threshold; if distance is less than eps, we consider it as collision.
    """
    cloud_ee = apply_pose_tq(ee_pose[..., None, :], ee_cloud[None])
    cloud_obj = apply_pose_tq(obj_pose[..., None, :], obj_cloud)

    query, target = cloud_ee, cloud_obj
    if query.shape[-2] > target.shape[-2]:
        query, target = target, query
    # Will this be faster than brute min(NxM) query?
    if brute:
        delta = query[..., None, :, :]
        - target[..., :, None, :]
        dists = th.linalg.norm(delta, dim=-1)
        dists = (dists.reshape(*query.shape[:-2], -1)
                 .min(dim=-1)
                 .values)
    else:
        dists, _, _ = knn_points(query, target, K=1,
                                 return_nn=False,
                                 return_sorted=False)
        dists = dists.squeeze_(dim=-1).min(dim=-1).values
    return (dists < eps), dists

def sample_gripper_poses(num_samples,
                         obj_pose: th.Tensor,
                         obj_cloud: th.Tensor,
                         obj_normal: th.Tensor,
                         cone_max_theta: float,
                         offset: float,
                         randomize_yaw: bool):
    device = obj_pose.device
    num_env: int = obj_pose.shape[0]
    num_point: int = obj_cloud.shape[-2]

    point_idx = th.randint(0,
                           num_point,
                           size=(num_env, num_samples,),
                           device=device).long()
    sample_cloud = th.take_along_dim(obj_cloud,
                                     point_idx[..., None], -2)
    sample_normals = th.take_along_dim(obj_normal,
                                       point_idx[..., None], -2)

    z_axis = th.zeros((num_env, num_samples, 3),
                      device=device)
    z_axis[..., -1] = 1

    if cone_max_theta > 0:
        theta = th.tensor(cone_max_theta, device=device)
        min_z = th.cos(theta)
        delta = 1 - th.cos(theta)
        z = min_z + delta * th.rand((num_env, num_samples,),
                                    device=device)
        axis = th.stack([th.zeros_like(z),
                         th.sqrt(1 - z**2), z], dim=-1)
    else:
        axis = z_axis.clone()

    sample_cloud += sample_normals * offset

    # N, S,
    sample_cloud = apply_pose_tq(obj_pose[..., None, 0:7],
                                 sample_cloud)
    sample_normals = quat_rotate(obj_pose[..., None, 3:7],
                                 sample_normals.reshape(sample_cloud.shape))

    EE_ori = align_vectors(axis, -sample_normals)
    delta = quat_rotate(EE_ori, axis) + sample_normals

    if randomize_yaw:
        random_yaw = 2 * np.pi * th.rand((num_samples, 1), device=device)
        yaw_quat = quat_from_axa(z_axis * random_yaw)
        EE_ori = quat_multiply(EE_ori, yaw_quat)
    # get EE
    # print(sampled_points.shape)
    # print(EE_ori.shape)
    # print(obj_pose.shape)
    EE_pose = th.cat([sample_cloud, EE_ori], axis=-1)
    return (EE_pose)


@th.jit.script
def orientation_error_osc(q_src: th.Tensor, q_dst: th.Tensor):
    s1, c1 = q_src[..., :3], q_src[..., 3:4]
    s2, c2 = q_dst[..., :3], q_dst[..., 3:4]
    return -(s1 * c2 - s2 * c1 + th.cross(s2, s1, dim=-1))


def find_actor_indices(gym, envs: Iterable, name: str,
                       domain=gymapi.IndexDomain.DOMAIN_SIM) -> List[int]:
    return [gym.find_actor_index(envs[i], name, domain)
            for i in range(len(envs))]


def find_actor_handles(gym, envs: Iterable, name: str) -> List[int]:
    return [gym.find_actor_handle(envs[i], name)
            for i in range(len(envs))]


class CartesianControlError:
    def __init__(self,
                 accumulate: bool,
                 target_shape: Optional[Tuple[int, ...]] = None,
                 dtype=th.float,
                 device=None,
                 clip_pos: Optional[float] = 0.1,
                 clip_orn: Optional[float] = 0.1,
                 pos_bound: th.Tensor = None,
                 local: bool = False
                 ):
        if local:
            raise ValueError('local is broken :(')
        self.accumulate = accumulate
        self.target = None

        if self.accumulate:
            self.target = th.zeros(target_shape,
                                   dtype=dtype,
                                   device=device)
            self.hasref = th.zeros(target_shape[:1],
                                   dtype=bool,
                                   device=device)
        self.clip_pos = clip_pos
        self.clip_orn = clip_orn
        self.pos_bound = pos_bound
        self.local = local

    def update_pos_bound(self, pos_bound: th.Tensor):
        self.pos_bound = pos_bound

    def clear(self, indices: th.Tensor):
        if not self.accumulate:
            return
        self.hasref[indices] = False

    def reset(self,
              state: th.Tensor,
              indices: th.Tensor,
              apply_index: bool = True
              ):
        if not self.accumulate:
            return

        if apply_index:
            if indices.dtype == th.bool:
                # self.target = th.where(indices, state, self.target)
                # self.target.masked_scatter_(indices, state)
                self.target[indices] = state[indices]
            else:
                self.target[indices] = state[indices]
        else:
            self.target[indices] = state

    @nvtx.annotate("Error.update")
    def update(self,
               state: th.Tensor,
               action: th.Tensor,
               relative: bool):
        if not relative:
            self.target = action
            return

        if self.accumulate:
            # Reset target with references.
            with nvtx.annotate("reset_target"):
                # self.target.masked_scatter_(~self.hasref[..., None], state)
                mask = (~self.hasref[..., None]).expand(self.target.shape)
                self.target[mask] = state[..., :7][mask]
                self.hasref.fill_(True)

            with nvtx.annotate("convert_quat"):
                if (action.shape[-1] == 6 or action.shape[-1] == 18 or
                        action.shape[-1] == 20):
                    # quat = axisaToquat(action[..., 3:6], action.device)
                    quat = quat_from_axa(action[..., 3:6])
                else:
                    quat = action[..., 3:7]

            # Compose transform to `target`.
            with nvtx.annotate("apply_delta"):
                self.target[..., 0:3] += action[..., 0:3]
                if quat.shape[-1] == 4:
                    if self.local:
                        self.target[..., 3:7] = quat_multiply(
                            self.target[..., 3:7], quat)
                    else:
                        self.target[..., 3:7] = quat_multiply(
                            quat, self.target[..., 3:7])

        else:
            # We directly save the `action.
            self.target = state.clone()
            self.target[..., 0:3] += action[..., 0:3]
            if (action.shape[-1] == 6 or action.shape[-1] == 18
                or action.shape[-1] == 20):
                # quat = axisaToquat(action[..., 3:6], action.device)
                quat = quat_from_axa(action[..., 3:6])
            else:
                quat = action[..., 3:7]

            if quat.shape[-1] >= 4:
                if self.local:
                    self.target[..., 3:7] = quat_multiply(
                        self.target[..., 3:7], quat)
                else:
                    self.target[..., 3:7] = quat_multiply(
                        quat, self.target[..., 3:7])
        if self.clip_pos is not None:
            # First, conditionally clamp target to workspace.
            with nvtx.annotate("clip_pos"):
                if self.pos_bound is not None:
                    self.target[..., :3].clamp_(self.pos_bound[..., 0, :3],
                                                self.pos_bound[..., 1, :3])
                # Clamp translational offset from ee state.
                self.target[..., :3].clamp_(state[..., :3] - self.clip_pos,
                                            state[..., :3] + self.clip_pos)

        if self.clip_orn is not None:
            # clamp rotational offset.
            with nvtx.annotate("clip_orn"):
                if self.local:
                    d_qxn = quat_multiply(quat_inverse(state[..., 3:7]),
                                          self.target[..., 3:7])
                else:
                    d_qxn = quat_multiply(self.target[..., 3:7],
                                          quat_inverse(state[..., 3:7]))
                d_axa = axis_angle_from_quat(d_qxn)
                angle = th.abs(
                    (th.linalg.norm(d_axa, dim=-1, keepdim=True) + th.pi) %
                    (2 * th.pi) - th.pi)
                oob_mask = (angle > self.clip_orn).squeeze(dim=-1)
                d_axa[oob_mask] = (d_axa * self.clip_orn / angle)[oob_mask]

                if self.local:
                    self.target[..., 3:7] = quat_multiply(state[..., 3:7],
                                                          quat_from_axa(d_axa))
                else:
                    self.target[..., 3:7] = quat_multiply(quat_from_axa(d_axa),
                                                          state[..., 3:7])

    @nvtx.annotate("CartesianControlError.error")
    def __call__(self, state: th.Tensor,
                 axa: bool = True):
        # if not self.accumulate:
        #     dt = self.target[..., :3] - state[..., :3]
        #     if axa:
        #         dr = -(state[:,-4:]*self.target[...,-1:]-self.target[...,-4:]*state[:,-1:]+\
        #                 th.cross(self.target[...,-4:], state[...,-4:],-1))
        #     else:
        #         dr = quat_multiply(self.target[..., 3:7],
        #                     quat_inverse(state[..., 3:7]))
        #     return th.cat([dt, dr], dim=-1)
        # else:
        #     dt = self.target[..., :3] - state[..., :3]
        #     dq = quat_multiply(self.target[..., 3:7],
        #                     quat_inverse(state[..., 3:7]))
        #     # print(self.target.shape)
        #     # print(state.shape)
        #     # print('dt', dt)
        #     # print('dq', dq)
        #     if axa:
        #         dr = axis_angle_from_quat(dq)
        #     else:
        #         dr = dq
        dt = self.target[..., :3] - state[..., :3]
        if axa:
            if False:  # not self.local:
                # "shortcut" only enabled for
                # AXA + non-local path
                dr = orientation_error_osc(
                    state[..., 3: 7],
                    self.target[..., 3: 7]
                )
            else:
                if self.local:
                    # R dR = R'
                    # dR = R^{-1} R'
                    dq = quat_multiply(quat_inverse(state[..., 3:7]),
                                       self.target[..., 3:7])
                else:
                    # dR R = R'
                    # dR = R' R^{-1}
                    dq = quat_multiply(self.target[..., 3:7],
                                       quat_inverse(state[..., 3:7]))
                dr = axis_angle_from_quat(dq)
            # dr = -(state[..., 3:6] * self.target[..., -1:] - self.target[..., 3:6]
            #        * state[:, -1:] + th.cross(self.target[..., 3:6], state[..., 3:6], -1))
        else:
            dr = quat_multiply(self.target[..., 3:7],
                               quat_inverse(state[..., 3:7]))
        return th.cat([dt, dr], dim=-1)


class JointControlError:
    def __init__(self,
                 accumulate: bool,
                 target_shape: Optional[Tuple[int, ...]] = None,
                 dtype=th.float,
                 device=None,
                 clip_pos: Optional[float] = 0.1,
                 clip_orn: Optional[float] = 0.1,
                 pos_bound: th.Tensor = None,
                 local: bool = False
                 ):
        self.pose_error = CartesianControlError(accumulate,
                                                target_shape,
                                                dtype,
                                                device,
                                                clip_pos,
                                                clip_orn,
                                                pos_bound,
                                                local)
        # self.pos_bound = pos_bound
        self.local = local
        self.device = device
        self.clip_pos = clip_pos
        self.q_target = None

        self._lmda = 0.01 * th.eye(6,
                                  dtype=th.float,
                                  device=device)

    @property
    def pos_bound(self):
        return self.pose_error.pos_bound

    def update_pos_bound(self, pos_bound: th.Tensor):
        return self.pose_error.update_pos_bound(pos_bound)

    def clear(self, index):
        return self.pose_error.clear(index)

    @classmethod
    @nvtx.annotate("get_delta_dof_pos")
    def _get_delta_dof_pos(cls,
                           delta_pose:th.Tensor,
                           jacobian:th.Tensor,
                           ik_method:str,
                           lmda:Optional[float]=None):
        """Get delta Franka DOF position from delta pose using specified IK method."""
        # References:
        # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
        # 2)
        # https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
        # (p. 47)

        if ik_method == 'pinv':  # Jacobian pseudoinverse
            k_val = 1.0
            jacobian_pinv = th.linalg.pinv(jacobian)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'trans':  # Jacobian transpose
            k_val = 1.0
            jacobian_T = th.transpose(jacobian, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'dls':  # damped least squares (Levenberg-Marquardt)
            delta_dof_pos = dls_ik(jacobian, delta_pose, lmda)
            # lambda_val = 0.1
            # jacobian_T = th.transpose(jacobian, dim0=1, dim1=2)
            # lambda_matrix = (
            #     lambda_val ** 2) * th.eye(n=jacobian.shape[1], device=self.device)
            # A = jacobian @ jacobian_T + lambda_matrix
            # rhs = th.linalg.solve(A, delta_pose)
            # delta_dof_pos = th.einsum('...jp,...p->...j', jacobian_T, rhs)
        elif ik_method == 'svd':  # adaptive SVD
            k_val = 1.0
            U, S, Vh = th.linalg.svd(jacobian)
            S_inv = 1. / S
            min_singular_value = 1.0e-5
            S_inv = th.where(
                S > min_singular_value,
                S_inv,
                th.zeros_like(S_inv))
            jacobian_pinv = th.transpose(Vh, dim0=1, dim1=2)[
                :, :, :6] @ th.diag_embed(S_inv) @ th.transpose(U, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        return delta_dof_pos

    @nvtx.annotate("IKController.update()")
    def update(self,
               state: th.Tensor,
               q_pos: th.Tensor,
               action: th.Tensor,
               j_eef,
               relative: bool = None,
               indices: Optional[th.Tensor] = None,
               update_pose: bool = True,
               recompute_orn: bool = False
               ):
        if indices is None:
            indices = th.arange(len(action),
                                dtype = th.long,
                                device = action.device)
        if update_pose:
            self.pose_error.update(state,
                                action,
                                relative)
        if self.q_target is None:
            self.q_target = q_pos.clone()
        if len(indices):
            if not recompute_orn:
                clamped_error = action[indices, :6].clone()
                clamped_error[:, :3] = self.pose_error(state)[indices, :3]
            else:
               clamped_error = self.pose_error(state)[indices]
        
            self.q_target[indices] = self._get_delta_dof_pos(
                clamped_error, j_eef[indices], 'dls', self._lmda) + q_pos[indices]

    @property
    def target(self):
        return self.q_target
        # return self.pose_error.target

    def __call__(self, state, q_pos: th.Tensor, j_eef):
        # delta_pose = self.pose_error(state)
        # return self._get_delta_dof_pos(delta_pose, 'dls', j_eef)
        return self.q_target - q_pos


def _matmul(a, b):
    return th.einsum('...ij,...j->...i', a, b)


class CartesianImpedanceController:
    """
    Maybe correct
    """

    def __init__(self,
                 kp_pos: th.Tensor,
                 kd_pos: th.Tensor,
                 kp_orn: th.Tensor,
                 kd_orn: th.Tensor,
                 max_u: th.Tensor,
                 device=None,
                 OSC=True):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_orn = kp_orn
        self.kd_orn = kd_orn
        self.max_u = max_u
        self.device = device
        self.OSC = OSC

    def update_gain(self, gains):
        self.kp_pos = gains[:, :3]
        self.kp_orn = gains[:, 3:6]
        self.kd_pos = gains[:, 6:9] * th.sqrt(self.kp_pos)
        self.kd_orn = gains[:, 9:] * th.sqrt(self.kp_orn)

    def __call__(self,
                 mm: th.Tensor,
                 J_e: th.Tensor,
                 ee_vel: th.Tensor,
                 pose_error: th.Tensor,
                 gains: Optional[th.Tensor] = None,
                 cache: Optional[Dict[str, th.Tensor]] = None
                 ):

        # Convert error -> control target.
        # We consider positional/orientation error differently.
        xddt = th.zeros_like(pose_error)
        xddt[:, :3] = (self.kp_pos *
                       pose_error[:, :3] -
                       self.kd_pos *
                       ee_vel[:, :3])
        xddt[:, 3:] = (self.kp_orn *
                       pose_error[:, 3:] -
                       self.kd_orn *
                       ee_vel[:, 3:])
        J_e_T = th.transpose(J_e, 1, 2)

        if self.OSC:
            # MM is PSD, so we can use cholesky_solve().
            mm_L = th.linalg.cholesky(mm)
            # mm_inv = th.cholesky_inverse(mm)
            # m_eef_inv = J_e @ mm_inv @ J_e_T
            m_eef_inv = J_e @ th.cholesky_solve(J_e_T, mm_L)
            if cache is not None:
                cache['m_eef_inv'] = m_eef_inv

            # if True:
            #     ii = th.linalg.inv(m_eef_inv)
            #     print(th.linalg.eigvals(ii))

            if True:
                wrench = th.linalg.solve(m_eef_inv, xddt)
            else:
                lhs = J_e @ th.linalg.inv(mm_L) @ J_e_T
                wrench = (th.linalg.inv(lhs) @ xddt[..., None]).squeeze(dim=-1)

        else:
            wrench = xddt

        # u = J_e^{T}@(J_e@MM^{-1}@J_e^T)^{-1} @ xddt
        u = th.einsum(
            '...ij, ...j -> ...i', J_e_T, wrench)

        # Clip the values to be within valid effort range
        u.clamp_(-self.max_u, self.max_u)
        return u


class IKController:
    """
    Maybe correct
    """

    def __init__(self,
                 kp: th.Tensor,
                 max_u: th.Tensor,
                 num_env,
                 device=None):
        self.kp = th.full((num_env, 7), kp,
                          dtype=th.float,
                          device=device)
        self.kd =  th.full((num_env, 7), np.sqrt(kp),
                          dtype=th.float,
                          device=device)
        self.max_u = max_u
        self.device = device

    def update_gain(self, gains,
                    indices = None):
        if indices is None:
            self.kp = gains[:, :7]
            self.kd = gains[:, 7:] * th.sqrt(self.kp)
        else:
            if len(indices)>0:
                self.kp[indices, :] = gains[indices, :7]
                self.kd[indices, :] = (gains[indices, 7:] *
                                        th.sqrt(self.kp[indices, :]))

    def __call__(self,
                 mm: th.Tensor,
                 q_vel: th.Tensor,
                 joint_error: th.Tensor,
                 cache: Optional[Dict[str, th.Tensor]] = None
                 ):
        u = (self.kp * joint_error - self.kd * q_vel)
        # .unsqueeze(-1)
        # u = (mm @ u).squeeze(-1)

        # Clip the values to be within valid effort range
        u.clamp_(-self.max_u, self.max_u)
        return u


@th.jit.script
def dls_ik(J: th.Tensor, p: th.Tensor, lmda: th.Tensor) -> th.Tensor:
    J_T = th.swapaxes(J, -2, -1)
    A = (J @ J_T).add_(lmda)
    # u = th.linalg.cholesky(A)
    # return (J_T @ th.cholesky_solve(p, u.unsqueeze(-1))).squeeze(-1)
    return (J_T @ th.linalg.solve(A, p)[..., None])[..., 0]


def get_analytic_jacobian(
        fingertip_quat, fingertip_jacobian, num_envs, device):
    """Convert geometric Jacobian to analytic Jacobian."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    # NOTE: Gym returns world-space geometric Jacobians by default

    batch = num_envs

    # Overview:
    # x = [x_p; x_r]
    # From eq. 2.189 and 2.192, x_dot = J_a @ q_dot = (E_inv @ J_g) @ q_dot
    # From eq. 2.191, E = block(E_p, E_r); thus, E_inv = block(E_p_inv, E_r_inv)
    # Eq. 2.12 gives an expression for E_p_inv
    # Eq. 2.107 gives an expression for E_r_inv

    # Compute E_inv_top (i.e., [E_p_inv, 0])
    I = th.eye(3, device=device)
    E_p_inv = I.repeat((batch, 1)).reshape(batch, 3, 3)
    E_inv_top = th.cat(
        (E_p_inv, th.zeros((batch, 3, 3), device=device)), dim=2)

    # Compute E_inv_bottom (i.e., [0, E_r_inv])
    fingertip_axis_angle = axis_angle_from_quat(fingertip_quat)
    fingertip_axis_angle_cross = get_skew_symm_matrix(
        fingertip_axis_angle, device=device)
    fingertip_angle = th.linalg.vector_norm(fingertip_axis_angle, dim=1)
    factor_1 = 1 / (fingertip_angle ** 2)
    factor_2 = 1 - fingertip_angle * 0.5 * th.sin(
        fingertip_angle) / (1 - th.cos(fingertip_angle))
    factor_3 = factor_1 * factor_2
    E_r_inv = I - 1 * 0.5 * fingertip_axis_angle_cross + (
        fingertip_axis_angle_cross @ fingertip_axis_angle_cross) * factor_3.unsqueeze(-1).repeat((1, 3 * 3)).reshape((batch, 3, 3))
    E_inv_bottom = th.cat(
        (th.zeros((batch, 3, 3), device=device), E_r_inv), dim=2)

    E_inv = th.cat(
        (E_inv_top.reshape((batch, 3 * 6)),
         E_inv_bottom.reshape((batch, 3 * 6))),
        dim=1).reshape(
        (batch, 6, 6))

    J_a = E_inv @ fingertip_jacobian

    return J_a


def get_skew_symm_matrix(vec, device):
    """Convert vector to skew-symmetric matrix."""
    # Reference:
    # https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication

    batch = vec.shape[0]
    I = th.eye(3, device=device)
    skew_symm = th.transpose(
        th.cross(
            vec.repeat((1, 3)).reshape((batch * 3, 3)),
            I.repeat((batch, 1))).reshape(batch, 3, 3),
        dim0=1, dim1=2)

    return skew_symm


def translate_along_local_z(pos, quat, offset, device):
    """Translate global body position along local Z-axis and express in global coordinates."""

    num_vecs = pos.shape[0]
    offset_vec = offset * th.tensor([0.0, 0.0, 1.0],
                                    device=device).repeat((num_vecs, 1))
    _, translated_pos = torch_utils.tf_combine(q1=quat, t1=pos, q2=th.tensor(
        [0.0, 0.0, 0.0, 1.0], device=device).repeat((num_vecs, 1)), t2=offset_vec)

    return translated_pos


def axis_angle_from_euler(euler):
    """Convert tensor of Euler angles to tensor of axis-angles."""

    quat = torch_utils.quat_from_euler_xyz(
        roll=euler[:, 0],
        pitch=euler[:, 1],
        yaw=euler[:, 2])
    quat = quat * th.sign(quat[:, 3]).unsqueeze(-1)  # smaller rotation
    axis_angle = axis_angle_from_quat(quat)

    return axis_angle


def axis_angle_from_quat(quat, eps=1.0e-6):
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference:
    # https://github.com/facebookresearch/pytorch3d/blob/bee31c48d3d36a8ea268f9835663c52ff4a476ec/pytorch3d/transforms/rotation_conversions.py#L516-L544

    if True:
        axis = F.normalize(quat[..., 0:3])

        half_angle = th.acos(quat[..., 3:].clamp(-1.0, +1.0))
        angle = (2.0 * half_angle + th.pi) % (2 * th.pi) - th.pi
        return axis * angle

    if False:
        # axis = quat[..., :3] # sin(h/2) * axis
        # w = quat[..., 3:]
        h = 2.0 * th.acos(quat[..., 3:].clamp(-1.0, +1.0)) / th.pi
        return quat[..., :3] / th.sinc(h)

        # scale factor: h/2 / cos(h/2)

        # [1]
        # norm(quat[..., :3]) = sin(h/2)

        # out[..., :3] = uvec(quat[..., :3])
        # out[..., 3:] = 2 * th.arccos(np.clip(w, -1.0, 1.0))
        # return quat[..., :3] * (2 * th.acos(w.clamp(-1.0, +1.0))) / th.linalg.norm(
        #         quat[..., :3], dim=-1, keepdim=True))
    else:
        mag = th.linalg.norm(quat[..., 0:3], dim=-1,
                             keepdim=True)
        half_angle = th.atan2(mag, quat[..., 3])

        axis = F.normalize(quat[..., 0:3])

        # print('half-angle', half_angle)
        # angle = 2.0 * half_angle
        # print('angle', angle)
        # th.sinc(half_angle / th.pi)  = sin(half_angle) / half_angle
        sin_half_angle_over_angle = th.where(th.abs(angle) > eps,
                                             th.sin(half_angle) / angle,
                                             1 / 2 - angle ** 2.0 / 48)
        # sin_half_angle_over_angle = 0.5 * th.sinc(half_angle / th.pi)
        # print('sin_half', sin_half_angle_over_angle)
        axis_angle = quat[:, 0:3] / sin_half_angle_over_angle

    return axis_angle


def axisaToquat(axisA: th.Tensor, device):

    num_rotations = axisA.shape[0]
    angle = th.norm(axisA, dim=-1)
    small_angle = (angle <= 1e-3)
    large_angle = ~small_angle

    scale = th.empty((num_rotations,), device=device, dtype=th.float)
    scale[small_angle] = (0.5 - angle[small_angle] ** 2 / 48 +
                          angle[small_angle] ** 4 / 3840)
    scale[large_angle] = (th.sin(angle[large_angle] / 2) /
                          angle[large_angle])
    quat = th.empty((num_rotations, 4), device=device, dtype=th.float)
    quat[:, :3] = scale[:, None] * axisA
    quat[:, -1] = th.cos(angle / 2)
    return quat
