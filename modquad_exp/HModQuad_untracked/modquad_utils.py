import math

import torch

def modquad_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    Termination function for ModQuad environment.
    Args:
        act: [batch_size, action_dim] (not used here)
        next_obs: [batch_size, obs_dim] (see ModQuadEnv._get_obs)
    Returns:
        done: [batch_size, 1] boolean tensor, True if episode should terminate
    """
    assert len(next_obs.shape) == 2

    # Extract robot state from observation
    # obs = [pos1(3), ang1(3), vel1(3), ang_vel1(3), pos2(3), ang2(3), vel2(3), ang_vel2(3)]
    pos = next_obs[:, 0:3]
    ang = next_obs[:, 3:6]
    # pos2 = next_obs[:, 12:15]  # target position, if you want to use it

    # Example thresholds
    max_xy = 5.0      # meters
    min_z = 0.05      # meters (crashed)
    max_z = 10.0      # meters (too high)
    max_angle = 45 * torch.pi / 180  # radians

    # Out of bounds checks
    out_of_bounds = (
        (pos[:, 0].abs() > max_xy) |
        (pos[:, 1].abs() > max_xy) |
        (pos[:, 2] < min_z) |
        (pos[:, 2] > max_z) |
        (ang.abs() > max_angle).any(dim=1)
    )

    done = out_of_bounds[:, None]
    return done
    
def modquad_reward(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    """
    Reward function for ModQuad: negative distance between robot and target positions.
    Args:
        act: [batch_size, action_dim] (not used)
        next_obs: [batch_size, obs_dim] (see ModQuadEnv._get_obs)
    Returns:
        reward: [batch_size, 1] float tensor
    """
    assert len(next_obs.shape) == 2
    # obs = [pos1(3), ang1(3), vel1(3), ang_vel1(3), pos2(3), ang2(3), vel2(3), ang_vel2(3)]
    pos = next_obs[:, 0:3]
    pos_des = next_obs[:, 12:15]
    dist = (pos - pos_des).norm(dim=1, keepdim=True)
    return -dist 

def modquad_reward_with_setpoints(act: torch.Tensor, next_obs: torch.Tensor, setpoint_trajectory=None, current_step=0) -> torch.Tensor:
    """
    Enhanced reward function for ModQuad that can handle changing setpoints during trajectory evaluation.
    Args:
        act: [batch_size, action_dim] (not used)
        next_obs: [batch_size, obs_dim] (see ModQuadEnv._get_obs)
        setpoint_trajectory: [horizon, 3] array of setpoints for the trajectory
        current_step: current step in the trajectory (0-indexed)
    Returns:
        reward: [batch_size, 1] float tensor
    """
    assert len(next_obs.shape) == 2
    # obs = [pos1(3), ang1(3), vel1(3), ang_vel1(3), pos2(3), ang2(3), vel2(3), ang_vel2(3)]
    pos = next_obs[:, 0:3]
    
    if setpoint_trajectory is not None and current_step < len(setpoint_trajectory):
        # Use the setpoint from the trajectory
        pos_des = torch.tensor(setpoint_trajectory[current_step], device=next_obs.device, dtype=next_obs.dtype)
        pos_des = pos_des.expand(next_obs.shape[0], -1)  # Expand to batch size
    else:
        # Fallback to using target position from observation
        pos_des = next_obs[:, 12:15]
    
    dist = (pos - pos_des).norm(dim=1, keepdim=True)
    return -dist

class ModQuadRewardWrapper:
    """
    Wrapper class to handle setpoint trajectories in reward computation.
    This can be used with ModelEnv to provide context-aware rewards.
    """
    def __init__(self, setpoint_trajectory=None):
        self.setpoint_trajectory = setpoint_trajectory
        self.current_step = 0
    
    def __call__(self, act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        reward = modquad_reward_with_setpoints(act, next_obs, self.setpoint_trajectory, self.current_step)
        self.current_step += 1
        return reward
    
    def reset(self, setpoint_trajectory=None):
        """Reset the wrapper for a new trajectory."""
        self.setpoint_trajectory = setpoint_trajectory
        self.current_step = 0 

def create_modquad_trajectory_eval_fn(model_env, setpoint_trajectory, num_particles=20):
    """
    Creates a custom trajectory evaluation function for ModQuad with setpoint tracking.
    
    Args:
        model_env: The ModelEnv instance
        setpoint_trajectory: [horizon, 3] array of setpoints for the trajectory
        num_particles: Number of particles for uncertainty estimation
    
    Returns:
        A function that evaluates action sequences with setpoint-aware rewards
    """
    def trajectory_eval_fn(initial_state, action_sequences):
        """
        Custom trajectory evaluation function that uses setpoint-aware rewards.
        
        Args:
            initial_state: Current observation
            action_sequences: [batch_size, horizon, action_dim] tensor of action sequences
            
        Returns:
            [batch_size] tensor of total rewards for each action sequence
        """
        with torch.no_grad():
            population_size, horizon, action_dim = action_sequences.shape
            
            # Create reward wrapper with setpoint trajectory
            reward_wrapper = ModQuadRewardWrapper(setpoint_trajectory)
            
            # Create a temporary ModelEnv with our custom reward function
            temp_model_env = type(model_env)(
                model_env.dynamics_model,
                model_env.termination_fn,
                reward_fn=reward_wrapper,
                generator=model_env._rng
            )
            
            # Evaluate using the temporary model environment
            return temp_model_env.evaluate_action_sequences(
                action_sequences, 
                initial_state=initial_state, 
                num_particles=num_particles
            )
    
    return trajectory_eval_fn 

