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

