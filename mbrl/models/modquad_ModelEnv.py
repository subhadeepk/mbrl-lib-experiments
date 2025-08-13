# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

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

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
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
            dones = self.termination_fn(actions, next_observs)

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
                planning_step = self.trajectory_step #+ time_step + 1
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
        trajectory = np.concatenate([self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.dx, self.dy, self.dz, self.droll, self.dpitch, self.dyaw])
        self.desired_trajectory = trajectory.reshape(-1, 12)

    def reward_fn(self,actions, next_observs, planning_step):
        """
        Calculates the reward for the current planning step, for all the particles in the batch.
        The reward is the sum of the absolute differences between the desired and predicted observations.
        """
        desired_obs = self.desired_trajectory[planning_step]
        desired_obs = desired_obs.reshape(-1, 10)
        desired_obs = desired_obs.to(self.device)
        next_observs = next_observs.to(self.device)

        # reward = torch.sum(torch.abs(next_observs[:3] - desired_obs[:3]), dim=1)
        # reward = torch.sum(torch.abs(next_observs[:, 0:3]), dim=1)

        #Using euclidian norm
        reward1 = - torch.norm(next_observs[:, 0:3], dim=1)
        reward2 = - torch.norm(next_observs[:, 3:12] - desired_obs[:, 3:12], dim=1)
        reward = 0.7*reward1 + 0.3*reward2
        return reward

    def termination_fn(self, actions, next_observs, planning_step):
        """
        if the planning step is the last waypoint index, then the episode is terminated.
        """
        if planning_step == self.num_waypoints:
            return torch.ones(actions.shape[0], 1, dtype=bool).to(self.device)
        else:
            return torch.zeros(actions.shape[0], 1, dtype=bool).to(self.device)