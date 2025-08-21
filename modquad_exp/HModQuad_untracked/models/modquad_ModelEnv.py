# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Optional, Tuple, Any, List

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


import mbrl.types

from . import Model
from modquad_logger import ModQuadLogger


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
        enable_logging: bool = True,
    ):
        self.dynamics_model = model
        # self.termination_fn = termination_fn
        # self.reward_fn = reward_fn
        self.device = model.device
        print(model.device)
        self.dt = 0.01
        self._int_pos = None

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
        self.u_hover = torch.tensor([3.24433114,0,0,0], device=self.device)
        self.last_action_taken = None
        # Logging control
        self.enable_logging = enable_logging
        
        # Initialize logger only if logging is enabled
        if self.enable_logging:
            self.logger = ModQuadLogger(log_dir="modquad_model_logs", enable_file_logging=True)
            print(f"ModQuad logging enabled. Logs will be saved to: modquad_model_logs/")
        else:
            self.logger = None
            print("ModQuad logging disabled.")

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

            # ---- NEW: initialize per-particle integrated position ----
        if self.current_pos is None:
            raise RuntimeError("Set self.current_pos (np.array shape [3,]) before calling reset/evaluate.")

        B = initial_obs_batch.shape[0]  # num_particles * population_size
        pos0 = torch.from_numpy(self.current_pos).to(self.device).expand(B, 3).clone()

        if model_state is None:
            model_state = {}
        model_state["int_pos"] = pos0    # track absolute position per particle
        self._int_pos = pos0             # keep a copy for reward/termination
        # Initialize last_action as a Bx4 tensor, broadcasting from last_action_taken if available
        if self.last_action_taken is not None:
            last_action_tensor = torch.tensor(self.last_action_taken).to(self.device)
            self.last_action = last_action_tensor.expand(B, -1)  # Broadcast to [B, 4]
        else:
            self.last_action = torch.zeros(B, 4).to(self.device)

        return model_state if model_state is not None else {}

    def step(
        self,
        actions: mbrl.types.TensorType,
        model_state: Dict[str, torch.Tensor],
        sample: bool = False,
        horizon_step: int = 0,
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

            # ---- NEW: integrate per-particle position using predicted linear velocity ----
            # next_observs layout: [dx, dy, dz, droll, dpitch, dyaw, qx, qy, qz, qw]
            v_lin = next_observs[:, 0:3]  # [B,3]
            if "int_pos" not in model_state:
                # Safety init (should already exist if reset did its job)
                B = v_lin.shape[0]
                pos0 = torch.from_numpy(self.current_pos).to(self.device).expand(B, 3).clone()
                model_state["int_pos"] = pos0
            

            prev_pos = model_state["int_pos"]              # [B,3]
            next_pos = prev_pos + self.dt * v_lin          # [B,3]
            next_model_state["int_pos"] = next_pos         # carry forward
            self._int_pos = next_pos                       # expose to reward/termination
            # self.current_thrust = actions[:,0]
            # print("int_pos", self._int_pos[:,2].max(), self._int_pos[:,2].min())

            # Compute Euler once here and stash (you already do this in reward_fn; either place works)
            # If you prefer to leave it in reward_fn, you can skip this.
            # q_xyzw = next_observs[:, 6:10]
            # self.next_obs_euler = self.quat_xyzw_to_euler_xyz_torch(q_xyzw)


            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs, horizon_step)
            )
            dones = self.termination_fn(actions, next_observs, horizon_step)

            if pred_terminals is not None:
                raise NotImplementedError(
                    "ModelEnv doesn't yet support simulating terminal indicators."
                )

            # Log the model step (only if logging is enabled)
            if self.enable_logging and self.logger is not None:
                self.logger.log_model_step(
                    actions=actions,
                    observations=actions,  # We don't have current obs, using actions as placeholder
                    rewards=rewards,
                    dones=dones,
                    model_state=next_model_state,
                    planning_step=planning_step,
                    trajectory_step=self.trajectory_step,
                    next_observations=next_observs,
                    predicted_rewards=pred_rewards,
                    predicted_terminals=pred_terminals
                )
            self.last_action = actions.detach()
            
            if self._return_as_np:
                next_observs = next_observs.cpu().numpy()
                rewards = rewards.cpu().numpy()
                dones = dones.cpu().numpy()
            return next_observs, rewards, dones, next_model_state

    def render(self, mode="human"):
        pass
    
    def start_trajectory_logging(self, trial_number: int = 0):
        """Start logging a new trajectory"""
        if self.enable_logging and self.logger is not None:
            return self.logger.start_trajectory(trial_number)
        else:
            print("Logging is disabled. Cannot start trajectory logging.")
            return -1
    
    def end_trajectory_logging(self, success: bool, termination_reason: str, 
                              waypoints_reached: int, total_waypoints: int, 
                              final_reward: float = 0.0):
        """End logging the current trajectory"""
        if self.enable_logging and self.logger is not None:
            self.logger.end_trajectory(success, termination_reason, waypoints_reached, 
                                      total_waypoints, final_reward)
        else:
            print("Logging is disabled. Cannot end trajectory logging.")
    
    def log_planning_time(self, planning_time: float):
        """Log planning time for the current trajectory step"""
        if self.enable_logging and self.logger is not None:
            self.logger.log_planning_time(planning_time)
    
    def log_step_reward(self, reward: float):
        """Log reward for the current trajectory step"""
        if self.enable_logging and self.logger is not None:
            self.logger.log_step_reward(reward)
    
    def log_trajectory_step(self, trajectory_step: int):
        """Log the current trajectory step progression"""
        if self.enable_logging and self.logger is not None:
            self.logger.log_trajectory_step(trajectory_step)
    
    def log_cem_iteration(self, metrics: Dict[str, Any]):
        """Log CEM iteration metrics"""
        if self.enable_logging and self.logger is not None:
            self.logger.log_cem_iteration(metrics)
    
    def log_experiment_parameters(self, cem_params: Dict[str, Any], 
                                 model_params: Dict[str, Any],
                                 planning_params: Dict[str, Any],
                                 env_params: Dict[str, Any]):
        """Log experiment configuration parameters"""
        if self.enable_logging and self.logger is not None:
            self.logger.log_experiment_parameters(cem_params, model_params, planning_params, env_params)
    
    def start_trial_logging(self, trial_number: int):
        """Start logging a new trial"""
        if self.enable_logging and self.logger is not None:
            self.logger.start_trial(trial_number)
    
    def log_planning_step(self, trial_number: int, planning_step: int,
                          best_predicted_reward: float, top_k_rewards: List[float],
                          top_k_average_reward: float, num_cem_iterations: int,
                          cem_converged: bool, actual_action_taken: List[float],
                          simulator_reward: float, planning_time: float,
                          population_evolution: Dict[str, Any]):
        """Log a complete planning step within a trial"""
        if self.enable_logging and self.logger is not None:
            self.logger.log_planning_step(
                trial_number, planning_step, best_predicted_reward, top_k_rewards,
                top_k_average_reward, num_cem_iterations, cem_converged,
                actual_action_taken, simulator_reward, planning_time, population_evolution
            )
    
    def end_trial_logging(self, trial_number: int, success: bool, termination_reason: str,
                          waypoints_reached: int, total_waypoints: int, total_reward: float):
        """End logging the current trial"""
        if self.enable_logging and self.logger is not None:
            self.logger.end_trial(trial_number, success, termination_reason,
                                 waypoints_reached, total_waypoints, total_reward)
    
    def get_logger(self):
        """Get the logger instance for external access"""
        if self.enable_logging and self.logger is not None:
            return self.logger
        else:
            print("Logging is disabled. No logger available.")
            return None
    
    def enable_logging(self, enable: bool = True):
        """Enable or disable logging dynamically"""
        if enable and not self.enable_logging:
            # Enable logging
            self.enable_logging = True
            if self.logger is None:
                self.logger = ModQuadLogger(log_dir="modquad_model_logs", enable_file_logging=True)
                print("ModQuad logging enabled. Logs will be saved to: modquad_model_logs/")
            else:
                print("ModQuad logging already enabled.")
        elif not enable and self.enable_logging:
            # Disable logging
            self.enable_logging = False
            print("ModQuad logging disabled.")
        else:
            print(f"Logging is already {'enabled' if enable else 'disabled'}.")
    
    def is_logging_enabled(self) -> bool:
        """Check if logging is currently enabled"""
        return self.enable_logging and self.logger is not None

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
                    action_batch, model_state, sample=True, horizon_step=time_step
                )
                rewards[terminated] = 0
                terminated |= dones
                total_rewards += rewards

            total_rewards = total_rewards.reshape(-1, num_particles)
            
            # Log the model evaluation (only if logging is enabled)
            if self.enable_logging and self.logger is not None:
                self.logger.log_model_evaluation(
                    action_sequences=action_sequences,
                    total_rewards=total_rewards,
                    num_particles=num_particles
                )
            
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
   
    def termination_fn(self, actions, next_observs, planning_step):
        device = self.device
        B = next_observs.shape[0]

        if self._int_pos is None:
            # Fallback (shouldn't happen if step ran)
            vel = next_observs[:, 0:3]
            x0 = torch.from_numpy(self.current_pos).to(device)
            pos = x0 + self.dt * vel
        else:
            pos = self._int_pos  # [B,3]
        vel = next_observs[:, 0:3]                                 
        att   = self.next_obs_euler #(next_observs[:, 6:10])
        rates = next_observs[:, 3:6]

        # Bounds (adjust to your platform & safety)
        # Position limits: [x_min, y_min, z_min], [x_max, y_max, z_max]
        pos_lower = torch.tensor([-10.0, -10.0, 0], device=device)   # m
        pos_upper = torch.tensor([10.0, 10.0, 5.0], device=device)      # m
        
        # Attitude limits: [roll_min, pitch_min, yaw_min], [roll_max, pitch_max, yaw_max]
        att_lower = torch.tensor([-0.17, -0.37, -0.5], device=device)  # ~-34° for roll/pitch, -180° for yaw
        att_upper = torch.tensor([0.17, 0.37, 0.5], device=device)     # ~34° for roll/pitch, 180° for yaw

        # Velocity limits: [dx_min, dy_min, dz_min], [dx_max, dy_max, dz_max]
        vel_lower = torch.tensor([-0.8, -0.51, -0.51], device=device)   # m/s
        vel_upper = torch.tensor([0.8, 0.4, 1.3], device=device)      # m/s
        
        # Angular rates limits: [droll_min, dpitch_min, dyaw_min], [droll_max, dpitch_max, dyaw_max]
        rates_lower = torch.tensor([-0.37, -0.9, -0.1], device=device)     # rad/s
        rates_upper = torch.tensor([0.37, 0.9, 0.09], device=device)        # rad/s

        bad = (
            # (pos[:,2] > self.desired_trajectory[self.trajectory_step][2]) |
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
    
    def quat_xyzw_to_euler_xyz_torch(self, q_xyzw: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of quaternions (x, y, z, w) to a batch of Euler angles (yaw, pitch, roll).
        The input is expected to be in x, y, z, w format.
        The output is in zyx (yaw, pitch, roll) format.
        
        Args:
            q_xyzw: A torch.Tensor of shape (B, 4) representing quaternions.
            
        Returns:
            A torch.Tensor of shape (B, 3) representing Euler angles.
        """
        x, y, z, w = q_xyzw.unbind(dim=-1)
        
        # Calculate roll (rotation around x-axis)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
        
        # Calculate pitch (rotation around y-axis)
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch_y = torch.asin(t2)
        
        # Calculate yaw (rotation around z-axis)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
        
        eulers = torch.stack((roll_x, pitch_y, yaw_z ), dim=-1)
        return eulers
    
    def _angle_wrap(self, a: torch.Tensor) -> torch.Tensor:
        # Wrap to (-pi, pi]
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _squared_l2(self, x: torch.Tensor) -> torch.Tensor:
        return (x ** 2).sum(dim=1)  # [B]

    # --- Latest reward function ---
    def reward_fn(self, actions, next_observs, planning_step):
        """
        next_observs: [B, 10]
        layout: [dx, dy, dz, droll, dpitch, dyaw, qx, qy, qz, qw]
        desired_trajectory row layout (12):
        [x, y, z, roll, pitch, yaw, dx, dy, dz, droll, dpitch, dyaw]
        Returns: [B, 1] (higher is better)
        """

        device = self.device
        B = next_observs.shape[0]
        ground_truth_pos = self.current_pos 
        ground_truth_velocity = self.current_velocity
        # print("horizon step", planning_step)


        ht_error_now = abs(ground_truth_pos[2] - self.desired_trajectory[self.trajectory_step][2])
        velocity_error_now = abs(ground_truth_velocity[2] - self.desired_trajectory[self.trajectory_step][6])
        if velocity_error_now < 0.02 and ht_error_now < 0.1:
            w_u = 100
            print("stabilize now")
        else:
            w_u = 0.0

        pos = self._int_pos

        # if pos_error_now < 0.1:
        #     w_p = 0.0
        # else:
        #     w_p = 5.0
        
        # if velocity_error_now < 0.05 or pos_error_now < 0.2:
        #     w_v = 0.0
        # else:
        #     w_v = 1.0

        # print(self.last_action[0], actions[0])

        # --- weights (tune as needed) ---
        w_p   = 3.0     # position
        w_v   = 0.1   # linear velocity
        w_att = 0.0 #2.0   # attitude (roll, pitch, yaw)
        w_w   = 0.0 #0.3   # body rates
        # w_u   = 0.0 #0.05  # control effort (around hover)
        w_du  = 0.1  #0.001 #0.025 #0.15   # control smoothness
        w_prog = 0.0
        
        # --- position error threshold for zeroing w_p ---
        pos_threshold = 0.1  # meters - adjust this value as needed
        velocity_threshold = 0.01  # m/s - adjust this value as needed
        # Desired 12-D state at this planning step
        desired = torch.from_numpy(self.desired_trajectory[self.trajectory_step]).to(device)  # [12]
        
        # Parse observation
        v_lin   = next_observs[:, 0:3]    # [B,3] dx,dy,dz
        w_body  = next_observs[:, 3:6]    # [B,3] droll,dpitch,dyaw
        q_xyzw  = next_observs[:, 6:10]   # [B,4] x,y,z,w

        # One-step position estimate around current_pos (as you had)
        x0 = torch.from_numpy(self.current_pos).to(device)  # [3]
                               # [B,3]

        # Euler from quaternion. Your function returns [roll, pitch, yaw]
        eul = self.quat_xyzw_to_euler_xyz_torch(q_xyzw)     # [B,3]
        self.next_obs_euler = eul                           # keep for termination_fn

        # Desired splits
        p_ref   = desired[0:3]      # [3]
        att_ref = desired[3:6]      # [3] roll, pitch, yaw
        v_ref   = desired[6:9]      # [3]
        w_ref   = desired[9:12]     # [3]

        # Errors (wrap angles for attitude)
        e_p   = p_ref - pos                    # [B,3]
        e_v   = v_lin - v_ref                  # [B,3]
        e_att = self._angle_wrap(eul - att_ref)  # [B,3]
        e_w   = w_body - w_ref                 # [B,3]   <-- FIX vs your old code
        e_du  = actions - self.last_action    # [B,4]

        # print("e_p avg", e_p[2].mean())
        # print("e_v avg", e_v[2].mean())
        # print("e_att avg", e_att[2].mean())
        # print("e_w avg", e_w[2].mean())
        # --- adaptive position weight based on error magnitude ---
        pos_error_magnitude = torch.abs(e_p[:, 2])  # [B] - Z-axis position error
        velocity_error_magnitude = torch.abs(e_v[:, 2])  # [B] - Z-axis velocity error

        # make scalars tensors once
        w_p_t  = torch.tensor(w_p,  device=device)
        w_v_t  = torch.tensor(w_v,  device=device)
        w_att_t= torch.tensor(w_att,device=device)
        w_w_t  = torch.tensor(w_w,  device=device)

        w_p_adaptive = torch.where(
            pos_error_magnitude <= pos_threshold,
            torch.tensor(0.0, device=device),
            w_p_t
        )
        w_v_adaptive = torch.where(
            pos_error_magnitude <= pos_threshold,
            w_v_t,
            torch.tensor(0.0, device=device)
        )

        # effort/smoothness only when settled
        w_du_adaptive = torch.where(
            (pos_error_magnitude <= pos_threshold) & (velocity_error_magnitude <= velocity_threshold),
            torch.tensor(w_du, device=device),
            torch.tensor(0.0, device=device)
        )
        w_u_adaptive = torch.where(
            (pos_error_magnitude <= pos_threshold) & (velocity_error_magnitude <= velocity_threshold),
            torch.tensor(w_u, device=device),
            torch.tensor(0.0, device=device)
        )

        # --- use the adaptive weights here ---
        cost = (
            w_p * self._squared_l2(e_p) +
            w_v * self._squared_l2(e_v) +
            w_att_t      * self._squared_l2(e_att) +
            w_w_t        * self._squared_l2(e_w)
        )

        cost = cost + w_du * self._squared_l2(e_du)
        reward = -cost
        
        return reward.unsqueeze(1)
    
