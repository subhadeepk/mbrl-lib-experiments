# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


import mbrl.types

from . import Model


class modquad_ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    This class can wrap a dynamics model to be used as an environment. The only requirement
    to use this class is for the model to use this wrapper is to have a method called
    ``predict()``
    with signature `next_observs, rewards = model.predict(obs,actions, sample=, rng=)`

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        model (:class:`mbrl.models.Model`): the model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
            Defaults to ``None``, in which case predicted rewards will be used.
        generator (torch.Generator, optional): a torch random number generator (must be in the
            same device as the given model). If None (default value), a new generator will be
            created using the default torch seed.
    """

    def __init__(
        self,
        env: gym.Env,
        model: Model,
        generator: Optional[torch.Generator] = None,
    ):
        self.dynamics_model = model
        # self.termination_fn = termination_fn
        # self.reward_fn = reward_fn
        self.device = model.device
        self.dt = 0.01

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        self.current_pos = None
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._return_as_np = True
        self.trajectory_step = 0

    def reset(
        self, initial_obs_batch: np.ndarray, return_as_np: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Resets the model environment.

        Args:
            initial_obs_batch (np.ndarray): a batch of initial observations. One episode for
                each observation will be run in parallel. Shape must be ``B x D``, where
                ``B`` is batch size, and ``D`` is the observation dimension.
            return_as_np (bool): if ``True``, this method and :meth:`step` will return
                numpy arrays, otherwise it returns torch tensors in the same device as the
                model. Defaults to ``True``.

        Returns:
            (dict(str, tensor)): the model state returned by `self.dynamics_model.reset()`.
        """
        if isinstance(self.dynamics_model, mbrl.models.OneDTransitionRewardModel):
            assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        with torch.no_grad():
            model_state = self.dynamics_model.reset(
                initial_obs_batch.astype(np.float32), rng=self._rng
            )
        self._return_as_np = return_as_np
        return model_state if model_state is not None else {}

    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
        planning_step: int = 0,
    ) -> Tuple[mbrl.types.TensorType, mbrl.types.TensorType, np.ndarray, Dict]:
        """Steps the model environment with the given batch of actions.

        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be ``B x A``, where ``B`` is the batch size (i.e., number of episodes),
                and ``A`` is the action dimension. Note that ``B`` must correspond to the
                batch size used when calling :meth:`reset`. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.
            model_state (dict(str, tensor)): the model state as returned by :meth:`reset()`.
            sample (bool): if ``True`` model predictions are stochastic. Defaults to ``False``.

        Returns:
            (tuple): contains the predicted next observation, reward, done flag and metadata.
            The done flag is computed using the termination_fn passed in the constructor.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            # if actions is tensor, code assumes it's already on self.device
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            (
                next_observs,
                pred_rewards,
                pred_terminals,
                next_model_state,
            ) = self.dynamics_model.sample(
                actions,
                model_state,
                deterministic=not sample,
                rng=self._rng,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs, planning_step)
            )
            dones = self.termination_fn(actions, next_observs, planning_step)

            if pred_terminals is not None:
                raise NotImplementedError(
                    "ModelEnv doesn't yet support simulating terminal indicators."
                )

            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, next_model_state

    def render(self, mode="human"):
        pass

    def evaluate_action_sequences(
        self,
        action_sequences: torch.Tensor,
        initial_state: np.ndarray,
        num_particles: int,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.
            initial_state (np.ndarray): the initial state for the trajectories.
            num_particles (int): number of times each action sequence is replicated. The final
                value of the sequence will be the average over its particles values.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        with torch.no_grad():
            assert len(action_sequences.shape) == 3
            population_size, horizon, action_dim = action_sequences.shape
            # either 1-D state or 3-D pixel observation
            assert initial_state.ndim in (1, 3)
            tiling_shape = (num_particles * population_size,) + tuple(
                [1] * initial_state.ndim
            )
            initial_obs_batch = np.tile(initial_state, tiling_shape).astype(np.float32)
            model_state = self.reset(initial_obs_batch, return_as_np=False)
            batch_size = initial_obs_batch.shape[0]
            total_rewards = torch.zeros(batch_size, 1).to(self.device)
            terminated = torch.zeros(batch_size, 1, dtype=bool).to(self.device)
            for time_step in range(horizon):
                # Fix: Ensure planning_step doesn't exceed trajectory bounds
                # planning_step = min(self.trajectory_step + time_step + 1, self.num_waypoints - 1)
                planning_step = self.trajectory_step
                action_for_step = action_sequences[:, time_step, :]
                action_batch = torch.repeat_interleave(
                    action_for_step, num_particles, dim=0
                )
                _, rewards, dones, model_state = self.step(
                    action_batch, model_state, sample=True, planning_step=planning_step
                )
                rewards[terminated] = 0
                terminated |= dones
                total_rewards += rewards

            total_rewards = total_rewards.reshape(-1, num_particles)
            return total_rewards.mean(dim=1)


    def set_desired_trajectory(self, total_time, pos_traj, orient_traj):
        self.x, self.y, self.z, self.dx, self.dy, self.dz, self.ddx, self.ddy, self.ddz = pos_traj
        self.roll, self.pitch, self.yaw, self.droll, self.dpitch, self.dyaw, self.ddroll, self.ddpitch, self.ddyaw = orient_traj
        self.num_waypoints = len(self.x)
        print("x_shape", len(self.x))
        trajectory = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.dx, self.dy, self.dz, self.droll, self.dpitch, self.dyaw]
        self.desired_trajectory = np.array(trajectory).T
        print("trajectory shape", self.desired_trajectory.shape)
        # print(trajectory[0])
        # self.desired_trajectory = trajectory.reshape(-1, 12)
        # print("desired traj shape", self.desired_trajectory.shape)

    def _reward_fn(self,actions, next_observs, planning_step):
        """
        Calculates the reward for the current planning step, for all the particles in the batch.
        The reward is the sum of the absolute differences between the desired and predicted observations.
        Trajectory structure: 
        x=0, y=1, z=2, roll=3, pitch=4, yaw=5, dx=6, dy=7, dz=8, droll=9, dpitch=10, dyaw=11
        Observation structure: dx=0, dy=1, dz=2, droll=3, dpitch=4, dyaw=5, quat=6,7,8,9
        """

        desired_obs = self.desired_trajectory[self.trajectory_step]
        desired_obs = torch.from_numpy(desired_obs).to(self.device)
        next_observs = next_observs.to(self.device)

        diff = desired_obs - next_observs
        # Step 1: L2 norm of first three elements of next_observs
        l2_next_first3 = torch.norm(next_observs[:, :3], dim=1)  # shape (1000,)

        # Step 2: L2 norm of certain elements of diff (e.g., indices 4, 6, 8)
        cols = [3, 4, 5, 6, 7, 8, 9, 10, 11]
        l2_diff_selected = torch.norm(diff[:, cols], dim=1)      # shape (1000,)

        # Step 3: Weighted sum
        w1, w2 = 0.5, 0.5
        weighted_sum = w1 * l2_next_first3 + w2 * l2_diff_selected  # shape (1000,)

        # Step 4: Expand to (batch size, 1)
        reward = - weighted_sum.unsqueeze(1)

    
        return reward #makes the reward dimensions (batch_size, 1 )

    def _termination_fn(self, actions, next_observs, planning_step):

        """
        if the planning step is the last waypoint index, then the episode is terminated.
        """
        # Fix: Ensure planning_step doesn't exceed trajectory bounds
        planning_step = min(planning_step, self.num_waypoints - 1)
        if planning_step >= self.num_waypoints - 1:
            return torch.ones(actions.shape[0], 1, dtype=bool).to(self.device)
        else:
            return torch.zeros(actions.shape[0], 1, dtype=bool).to(self.device)
        

    def reward_fn(self, actions, next_observs, planning_step):
        """
        next_observs: [B, 10], actions: [B, A]  -> returns [B]
        Higher is better.

        Calculates the reward for the current planning step, for all the particles in the batch.
        The reward is the sum of the absolute differences between the desired and predicted observations.
        Trajectory structure: 
        x=0, y=1, z=2, roll=3, pitch=4, yaw=5, dx=6, dy=7, dz=8, droll=9, dpitch=10, dyaw=11
        Observation structure: dx=0, dy=1, dz=2, droll=3, dpitch=4, dyaw=5, quat=6,7,8,9
        """
        
        device = self.device
        B = next_observs.shape[0]
        # print("next_observs shape", next_observs.shape)
        # print("actions shape", actions.shape)

        # Desired 12-D state at this step. If your trajectory stores one 12-D vector per step:
        desired_obs = self.desired_trajectory[self.trajectory_step]
        desired_obs = torch.from_numpy(desired_obs).to(self.device)

        # next_observs_seq: [B, T, 10] (first 3 = velocities)
        v = next_observs[:, :3]                          # [B, 3]
        x0 = torch.from_numpy(self.current_pos).to(self.device)                            
        pos = x0 + self.dt * v
        # print("pos shape", pos.shape)
        next_obs_euler = self.quat_xyzw_to_euler_zyx(next_observs[:, 6:10])  # [B, 3]

        # Calculate errors for position, orientation, velocity, and angular velocity
        position_error = torch.norm(pos - desired_obs[:3], dim=1)
        orientation_error = torch.norm(next_obs_euler - desired_obs[3:6], dim=1)
        velocity_error = torch.norm(next_observs[:, 0:3] - desired_obs[6:9], dim=1)
        angular_velocity_error = torch.norm(next_observs[:, 3:6] - desired_obs[9:12], dim=1)

        # Reward = negative cost
        reward = -(position_error + orientation_error + velocity_error + angular_velocity_error)
        # Ensure reward has shape (B, 1) for consistency
        return reward.unsqueeze(1)
    
    def termination_fn(self, actions, next_observs, planning_step):
        device = self.device
        B = next_observs.shape[0]

        vel = next_observs[:, 0:3]      
        x0 = torch.from_numpy(self.current_pos).to(self.device)                            
        pos = x0 + self.dt * vel
        att   = self.quat_xyzw_to_euler_zyx(next_observs[:, 6:10])
        rates = next_observs[:, 3:6]

        # Bounds (adjust to your platform & safety)
        # Position limits: [x_min, y_min, z_min], [x_max, y_max, z_max]
        pos_lower = torch.tensor([-5.0, -2.0, 0], device=device)   # m
        pos_upper = torch.tensor([5.0, 2.0, 2.0], device=device)      # m
        
        # Attitude limits: [roll_min, pitch_min, yaw_min], [roll_max, pitch_max, yaw_max]
        att_lower = torch.tensor([-0.17, -0.37, -0.5], device=device)  # ~-34° for roll/pitch, -180° for yaw
        att_upper = torch.tensor([0.17, 0.37, 0.5], device=device)     # ~34° for roll/pitch, 180° for yaw

        # Velocity limits: [dx_min, dy_min, dz_min], [dx_max, dy_max, dz_max]
        vel_lower = torch.tensor([-0.8, -0.4, -0.7], device=device)   # m/s
        vel_upper = torch.tensor([0.8, 0.4, 0.7], device=device)      # m/s
        
        # Angular rates limits: [droll_min, dpitch_min, dyaw_min], [droll_max, dpitch_max, dyaw_max]
        rates_lower = torch.tensor([-0.6, -0.6, -0.1], device=device)     # rad/s
        rates_upper = torch.tensor([0.6, 0.6, 0.1], device=device)        # rad/s

        bad = (
            (pos < pos_lower).any(dim=1) | (pos > pos_upper).any(dim=1) |
            (vel < vel_lower).any(dim=1) | (vel > vel_upper).any(dim=1) |
            (att < att_lower).any(dim=1) | (att > att_upper).any(dim=1) |
            (rates < rates_lower).any(dim=1) | (rates > rates_upper).any(dim=1)
        )
        # if bad.any():
        #     print("bad", bad)
        done_waypoints = self.trajectory_step >= (self.num_waypoints - 1)
        done = (bad | done_waypoints).view(B, 1)
        # done = (done_waypoints).view(B, 1)
        # done = torch.ones(B, 1, dtype=bool).to(self.device)

        return done
    

    def quat_xyzw_to_euler_zyx(self, q_xyzw: torch.Tensor, degrees: bool = False) -> torch.Tensor:
        # q_xyzw: [..., 4] in (x, y, z, w). Kornia expects (w, x, y, z).
        eulers = R.from_quat(q_xyzw.numpy()).as_euler('xyz', degrees=False)
        eulers = torch.from_numpy(eulers)
        return eulers