#!/usr/bin/env python3

from yourdfpy import URDF
from typing import Tuple, Dict, Optional
import numpy as np
import torch as th
import pickle
import einops

try:
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose as CuroboPose
    from curobo.wrap.model.robot_world import (
        RobotConfig,
        RobotWorld,
        RobotWorldConfig,
        WorldConfig,
        WorldCollisionConfig)
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig, IKResult
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.sdf.world_mesh import (
        WorldMeshCollision,
        WorldPrimitiveCollision
    )
    from curobo.util_file import load_yaml
    from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
    from curobo.types.tensor import T_BDOF, T_BValue_bool, T_BValue_float
    from curobo.rollout.rollout_base import tensor_repeat_seeds

    def solve_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        goal_pose: CuroboPose,
        num_seeds: int,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, CuroboPose]] = None,
        env_idx: Optional[th.Tensor] = None
    ) -> IKResult:
        # create goal buffer:
        goal_buffer = self.update_goal_buffer(solve_state, goal_pose,
                                              retract_config, link_poses)
        if env_idx is not None:
            # print('A', goal_buffer.batch_world_idx)
            goal_buffer.batch_world_idx = (
                tensor_repeat_seeds(env_idx[..., None], num_seeds)
            ).to(goal_buffer.batch_world_idx.dtype).reshape(
                goal_buffer.batch_world_idx.shape
            )
        coord_position_seed = self.get_seed(
            num_seeds, goal_buffer.goal_pose, use_nn_seed, seed_config
        )

        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = newton_iters
        self.solver.reset()
        result = self.solver.solve(goal_buffer, coord_position_seed)
        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = self.og_newton_iters
        ik_result = self.get_result(
            num_seeds,
            result,
            goal_buffer.goal_pose,
            return_seeds)
        if ik_result.goalset_index is not None:
            ik_result.goalset_index[ik_result.goalset_index >=
                                    goal_pose.n_goalset] = 0

        return ik_result

    def solve_batch_env(
            self: IKSolver,
        goal_pose: CuroboPose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, CuroboPose]] = None,
        env_idx: Optional[th.Tensor] = None
    ) -> IKResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV,
            num_ik_seeds=num_seeds,
            batch_size=goal_pose.batch,
            n_envs=goal_pose.batch,
            n_goalset=1,
        )
        return solve_from_solve_state(
            self,
            solve_state,
            goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
            env_idx=env_idx
        )
except BaseException:
    print('no-curobo')
    pass

from ham.util.torch_util import dcn
from ham.util.path import get_path
from ham.util.math_util import invert_pose_tq, xyzw2wxyz
from ham.env.episode.spec import DefaultSpec
from ham.env.episode.util import upsert


def _finalize_curobo_pose(robot_pos,
                          x: th.Tensor,
                          i: th.Tensor):
    """
    Does the following five operations:
        1. (get) pose = pose[i]
        2. (to robot frame) pose = robot_from_world @ pose
        3. (invert) pose = pose^{I}
        4. (roll quat) pose.quat = xyzw2wxyz(pose.quat)
        5. (set) pose[i] = pose
    """
    # extract corresponding indices
    x_i = x[i, ..., 0:7]

    # Clone if necessary, since we're going to
    # modify x_i in-place.
    if (x_i.data_ptr() == x.data_ptr()):
        x_i = x_i.clone()

    # [0] convert world-frame pose
    # to be relative to the robot,
    # since in curobo robot is located
    # at the origin (0,0,0).
    # FIXME(ycho): consider actually applying
    # the inverse transform...!
    x_i[..., :3] -= robot_pos[..., None, :3]

    # [1] invert to match curobo convention:
    # pose = `obj_from_world` transform
    x_i = invert_pose_tq(x_i)
    # [2] match curobo convention:
    # quat = (wxyz) ordering
    x_i[..., 3:7] = xyzw2wxyz(x_i[..., 3:7])
    # Update in-place.
    x[i, ..., 0:7] = x_i
    return x


class CollisionFree(DefaultSpec):

    @property
    def setup_keys(self) -> Tuple[str, ...]: return ()

    @property
    def setup_deps(self) -> Tuple[str, ...]: return ('reset_ids',
                                                     'table_geom',
                                                     'obj_ctx',
                                                     'rel_scale'
                                                     )

    @property
    def reset_deps(self) -> Tuple[str, ...]: return ('reset_ids',
                                                     'robot_pos',
                                                     'obj_poses',
                                                     'goal_poses',
                                                     'table_pos',
                                                     'curr_pose_meta',
                                                     'robot_dof',
                                                     )

    @property
    def reset_keys(self) -> Tuple[str, ...]: return (
        'obj_pose',
        'goal_pose',
        'col_free',
        'has_dof',
        'robot_dof',
    )

    def sample_setup(
            self, ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        # Load geometries for curobo
        tmpdir = ctx['tmpdir']
        num_env = ctx['num_env']
        device = ctx['device']
        obj_set = ctx['obj_set']
        use_cuda_graph = ctx['use_cuda_graph']
        near_obj = ctx['near_obj']

        # FIXME(ycho): _basically_ hardcoded...!!!
        robot_file = ctx.get('robot_file',
                             # 'assets/real/franka_custom.yml'
                             'assets/real/fr3_custom.yml'
                             )
        robot_urdf = ctx.get('robot_urdf', get_path(
            # 'assets/franka_description/robots/franka_panda_custom_v3.urdf'
            'assets/franka_description/robots/fr3_custom_reduceq0.urdf'
        ))

        obj_ctx = data['obj_ctx']
        obj_names = obj_ctx['obj_name']
        box_dims = dcn(data['table_geom'])
        obj_scale = dcn(data['rel_scale'])

        world_data = []
        for i in range(num_env):
            urdf_file = obj_set.urdf(obj_names[i])
            urdf = URDF.load(urdf_file,
                             build_scene_graph=False,
                             load_meshes=False)

            mesh_data = {}
            for k, v in urdf.link_map.items():
                for j, c in enumerate(v.collisions):
                    # print(c.origin) # should be identity...!!!
                    mesh_file = c.geometry.mesh.filename
                    mesh_scale = c.geometry.mesh.scale * obj_scale[i]
                    mesh_data[F'obj_{i:03d}_{j:02d}'] = {
                        'pose': [0, 0, 0,
                                 1, 0, 0, 0],
                        'scale': mesh_scale.tolist(),
                        'file_path': mesh_file
                    }
            cuboid_data = {
                F'box_{i:03d}_{j:02d}': {
                    'dims': box_dims[i, j].tolist(),
                    'pose': [0, 0, 0,
                             1, 0, 0, 0]
                }
                for j in range(box_dims.shape[1])
            }
            world_datum = {
                'mesh': mesh_data,
                'cuboid': cuboid_data
            }
            world_data.append(world_datum)

        world_cfg = [WorldConfig.from_dict(d) for d in world_data]
        world_cfg = [WorldConfig.create_collision_support_world(c)
                     for c in world_cfg]
        tensor_args = TensorDeviceType(device=device)

        # == load robot ==
        robot_file = get_path(robot_file)
        robot_dict = load_yaml(robot_file)["robot_cfg"]
        robot_dict['kinematics']['external_robot_configs_path'] = get_path(
            'assets')
        robot_dict['kinematics']['asset_root_path'] = get_path('assets')
        robot_dict['kinematics']['urdf_path'] = robot_urdf
        robot_cfg = RobotConfig.from_dict(robot_dict)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            rotation_threshold=(2 * np.pi),
            collision_checker_type=CollisionCheckerType.MESH,
            position_threshold=(0.01),
            num_seeds=1,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            # Cuda graph I think does _not_ work...???
            use_cuda_graph=use_cuda_graph,
            # collision_activation_distance=0.10,
            regularization=True,
            use_particle_opt=True,
        )
        ik_solver = IKSolver(ik_config)

        ik_config_2 = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            # world_cfg,
            # collision_checker_type=CollisionCheckerType.MESH,
            world_coll_checker=ik_solver.world_coll_checker,
            rotation_threshold=(2 * np.pi),
            position_threshold=(0.03),
            num_seeds=1,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
            # Cuda graph I think does _not_ work...???
            use_cuda_graph=use_cuda_graph,
            # collision_activation_distance=0.10,
            regularization=True,
            use_particle_opt=True,
        )
        ik_solver_2 = IKSolver(ik_config_2)

        # Position-only IK
        ik_solver.update_pose_cost_metric(
            PoseCostMetric(
                reach_partial_pose=True,
                reach_vec_weight=tensor_args.to_device(
                    [0, 0, 0,
                     1, 1, 1]
                )
            )
        )
        ik_solver_2.update_pose_cost_metric(
            PoseCostMetric(
                reach_partial_pose=True,
                reach_vec_weight=tensor_args.to_device(
                    [0, 0, 0,
                     1, 1, 1]
                )
            )
        )

        # Update handles for later use
        ctx['ik_solver'] = ik_solver
        ctx['ik_solver_2'] = ik_solver_2
        return data

    def sample_reset(self,
                     ctx, data: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        ik_solver = ctx['ik_solver']
        ik_solver_2 = ctx['ik_solver_2']
        near_obj = ctx['near_obj']
        col_world = ik_solver.world_coll_checker
        reset_ids = data['reset_ids']
        num_reset: int = len(reset_ids)
        if num_reset <= 0:
            return data
        # barrier_ids = ctx['barrier_id'][reset_ids].long()
        robot_pos = data['robot_pos'][reset_ids]

        curr_pose_meta = data['curr_pose_meta']

        # FIXME(ycho): we basically have to recompute
        # `plate_pose` ... which is kind of inconvenient.
        plate_pose = curr_pose_meta['pose'].clone()
        plate_pose[..., :2] += data['table_pos'][reset_ids, ..., None, :2]

        # Update box poses from env
        col_world._cube_tensor_list[1][reset_ids, ..., 0:7] = (
            plate_pose
        )
        out = _finalize_curobo_pose(robot_pos,
                                    col_world._cube_tensor_list[1],
                                    reset_ids)

        # Compute target positions
        # (also, relative to the robot)
        init_poses = data['obj_poses'][reset_ids]
        goal_poses = data['goal_poses'][reset_ids]
        target_xyz = th.cat([
            # FIXME(ycho): obj_poses[...,2] -= z_eps
            # to be _technically_ correct.
            init_poses[..., :3],
            goal_poses[..., :3]
        ], dim=-2)
        target_xyz[..., :3] -= robot_pos[..., None, :3]
        # print('target_xyz', target_xyz.shape)
        ik_target_pose = CuroboPose(
            target_xyz.reshape(-1, 1, 3),
            None
        )

        # Temporarily disable object collision queries
        # And check generated init/goal poses.
        query_size: int = target_xyz.shape[-2]
        col_world._mesh_tensor_list[2].fill_(0)
        with th.enable_grad():
            result = solve_batch_env(
                ik_solver,
                ik_target_pose,
                num_seeds=1,
                return_seeds=1,
                env_idx=einops.repeat(reset_ids, '... -> (... r)',
                                      r=query_size)
            )
        col_world._mesh_tensor_list[2].fill_(1)

        # Format output
        success = result.success.reshape(num_reset,
                                         query_size)
        solution = result.solution.reshape(num_reset,
                                           query_size,
                                           -1)
        last_pos = result.goal_pose.position.reshape(
            num_reset, query_size, -1)

        # Split into init/goal components.
        N: int = init_poses.shape[-2]
        init_suc, goal_suc = success[..., :N], success[..., N:]
        init_sol, goal_sol = solution[..., :N, :], solution[..., N:, :]
        init_pos, goal_pos = last_pos[..., :N, :], last_pos[..., N:, :]
        init_orn = init_poses[..., 3:7]
        goal_orn = goal_poses[..., 3:7]

        # Select collision-free solutions.
        # NOTE(ycho): we take the poses from `init_poses`,
        # rather than `last_pos`, so there's no need to
        # add `robot_pos` again.
        init_best = th.max(init_suc,
                           dim=-1,
                           keepdim=True)
        init_pose = th.take_along_dim(init_poses,
                                      init_best.indices[..., None],
                                      dim=-2).squeeze(dim=-2)
        goal_best = th.max(goal_suc,
                           dim=-1,
                           keepdim=True)
        goal_pose = th.take_along_dim(goal_poses,
                                      goal_best.indices[..., None],
                                      dim=-2).squeeze(dim=-2)

        # Find a collision-free initial configuration
        # for the robot near the object (otherwise set it random)
        if near_obj:
            col_world._mesh_tensor_list[1][reset_ids, ..., 0, 0:7] = (
                init_pose
            )
            _finalize_curobo_pose(robot_pos,
                                  col_world._mesh_tensor_list[1][..., 0:7],
                                  reset_ids)
            obj_pos = (init_pose[..., : 3].reshape(-1, 1, 3) -
                       robot_pos[..., None, : 3])
            ik_target_pose = CuroboPose(
                obj_pos,
                None)
            # unused for now
            # seed_config = th.take_along_dim(init_sol,
            #                                 init_best.indices[..., None],
            #                                 dim=-2).squeeze(dim=-2)
            pos_thresh = ik_solver_2.position_threshold
            ik_solver_2.position_threshold = 0.1
            with th.enable_grad():
                result = ik_solver_2.solve_batch_env(
                    ik_target_pose,
                    num_seeds=64,
                    return_seeds=1,
                    env_idx=reset_ids
                )
            ik_solver_2.position_threshold = pos_thresh
            has_dof = result.success.squeeze(dim=-1)
            robot_dof = result.solution.squeeze(dim=-2)[has_dof]
        else:
            col_world._mesh_tensor_list[1][reset_ids, ..., 0, 0:7] = (
                init_pose
            )
            _finalize_curobo_pose(robot_pos,
                                  col_world._mesh_tensor_list[1][..., 0:7],
                                  reset_ids)
            q_lo = ik_solver_2.kinematics.get_joint_limits().position[0]
            q_hi = ik_solver_2.kinematics.get_joint_limits().position[1]
            rand_config = th.rand((num_reset, 64, 7),
                                  dtype=th.float32,
                                  device=reset_ids.device) * (q_hi - q_lo) + q_lo
            robot_world = RobotWorld(RobotWorldConfig.load_from_config(
                ik_solver_2.robot_config,
                world_collision_checker=ik_solver_2.world_coll_checker,
                collision_activation_distance=0.0,
                self_collision_activation_distance=0.0))
            sd, wd = robot_world.get_world_self_collision_distance_from_joint_trajectory(
                rand_config, reset_ids.to(dtype=th.int32))
            feasible = (sd <= 0) & (wd <= 0)
            has_sol = th.max(feasible, dim=-1)
            has_dof = has_sol.values
            robot_dof = th.take_along_dim(rand_config,
                                          has_sol.indices[..., None, None],
                                          dim=-2).squeeze(dim=-2)

        # Is the scene "feasible" ?
        col_free = th.logical_and(
            init_best.values.squeeze(dim=-1),
            goal_best.values.squeeze(dim=-1)
        )

        # And then update data.
        upsert(data, reset_ids, 'obj_pose', init_pose)
        upsert(data, reset_ids, 'goal_pose', goal_pose)
        upsert(data, reset_ids, 'col_free', col_free)
        upsert(data, reset_ids, 'has_dof', has_dof)
        upsert(data, reset_ids[has_dof], 'robot_dof', robot_dof)

        return data
