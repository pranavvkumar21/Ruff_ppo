from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils import configclass
import torch

def velocity_tracking(env, key="x", slope=-3):
    direction = {"x": 0, "y": 1, "z": 2}
    if key=="z":
        vel = env.scene["ruff"].data.root_ang_vel_b[:, direction[key]]
    else:
        vel = env.scene["ruff"].data.root_lin_vel_b[:, direction[key]]
    cmd_vel = mdp.generated_commands(env, command_name="velocity_command")[:, direction[key]]
    c_v = 1.0 / cmd_vel.abs().clamp_min(1e-3)
    vel_reward = torch.exp(slope * c_v * (vel - cmd_vel) ** 2)
    return vel_reward

def balance_reward(env):
    vz = env.scene["ruff"].data.root_lin_vel_b[:, 2]
    wx = env.scene["ruff"].data.root_ang_vel_b[:, 0]
    wy = env.scene["ruff"].data.root_ang_vel_b[:, 1]
    cmd_x = mdp.generated_commands(env, command_name="velocity_command")[:, 0]
    cx = 1.0 / cmd_x.abs().clamp_min(1e-3)
    w_xy = torch.sqrt(wx ** 2 + wy ** 2)
    balance = (torch.exp(-2.5 * cx * vz ** 2) + torch.exp(-2.0 * cx * w_xy ** 2))
    return balance

def twist(env):
    quat = env.scene["ruff"].data.root_quat_w               # already on GPU                # runs on GPU
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    cmd_x = mdp.generated_commands(env, command_name="velocity_command")[:, 0]
    cx = 1.0 / cmd_x.abs().clamp_min(1e-3)
    norm = torch.sqrt((roll ** 2 + pitch ** 2)) * cx
    return norm
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    forward_vel = RewTerm(func=velocity_tracking, params={"key": "x", "slope": -3}, weight=2.0)
    lateral_vel = RewTerm(func=velocity_tracking, params={"key": "y", "slope": -3}, weight=2.0)
    angular_vel = RewTerm(func=velocity_tracking, params={"key": "z", "slope": -1.5}, weight=1.5)
    balance = RewTerm(func=balance_reward, weight=2.0)
    twist = RewTerm(func=twist, weight=-0.6)
