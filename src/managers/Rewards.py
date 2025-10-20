from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils import configclass
import torch
import yaml
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "ruff_reward_config.yaml", "r") as f:
    config = yaml.safe_load(f)

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

def balance_reward(env,vz_slope=-2.5, wxy_slope=-2.0):
    vz = env.scene["ruff"].data.root_lin_vel_b[:, 2]
    wx = env.scene["ruff"].data.root_ang_vel_b[:, 0]
    wy = env.scene["ruff"].data.root_ang_vel_b[:, 1]
    cmd_x = mdp.generated_commands(env, command_name="velocity_command")[:, 0]
    cx = 1.0 / cmd_x.abs().clamp_min(1e-3)
    w_xy = torch.sqrt(wx ** 2 + wy ** 2)
    balance = (torch.exp(vz_slope * cx * vz ** 2) + torch.exp(wxy_slope * cx * w_xy ** 2))
    return balance

def twist(env):
    quat = env.scene["ruff"].data.root_quat_w               # already on GPU                # runs on GPU
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    cmd_x = mdp.generated_commands(env, command_name="velocity_command")[:, 0]
    cx = 1.0 / cmd_x.abs().clamp_min(1e-3)
    norm = torch.sqrt((roll ** 2 + pitch ** 2)) * cx
    #logto extras after mean
    # print(env.extras)
    env.extras["twist"] = norm
    return norm

def rhythm_rewards(env, c1=1.0, height_thresh=0.011, offset=0.007):
    sensors = [env.scene.sensors[n] for n in ["fl3_height", "fr3_height", "rl3_height", "rr3_height"]]
    # stack their world positions and hit points → shape [4, num_envs, num_rays, 3]
    pos_w = torch.stack([s.data.pos_w for s in sensors], dim=1)
    hits_w = torch.stack([s.data.ray_hits_w.mean(dim=1) for s in sensors], dim=1) 

    foot_ids = [env.scene["ruff"].data.body_names.index(name) for name in [
        "FL3_f", "FR3_f", "RL3_f", "RR3_f",
    ]]
    # compute heights in one go
    heights = pos_w[..., 2] - hits_w[..., 2] - offset  # [ num_envs, 4]
 
    phase = env.cmd["phase"]
    stance_mask = (phase < torch.pi).float()      # stance = first half of gait
    swing_mask  = (phase >= torch.pi).float() 

    #foot stance reward
    stance_term = (heights < height_thresh).float() * stance_mask
    foot_stance_reward = stance_term.sum(dim=1)
    #foot clear reward

    swing_term = (heights > height_thresh).float() * swing_mask
    foot_clear_reward = 0.7 * c1 * swing_term.sum(dim=1)

    cmd_x = mdp.generated_commands(env, command_name="velocity_command")[:, 0]
    cx = 1.0 / cmd_x.abs().clamp_min(1e-3)

    #get foot velocities
    foot_vel = env.scene["ruff"].data.body_lin_vel_w[:, foot_ids]

    #calculate foot slip penalty
    foot_vel_xy = foot_vel[..., :2]                               # [N,4,2]
    foot_speed_xy = torch.linalg.norm(foot_vel_xy, dim=-1) 
    foot_slip_reward = -0.7 * cx * (foot_speed_xy * stance_mask).sum(dim=1)

    #calculate foot z velocity penalty
    foot_vel_z = foot_vel[..., 2]                                 # [N,4]
    foot_zvel_reward = -0.03 * cx * (foot_vel_z.abs().sum(dim=1) ** 2) 

    #RG frequency penalty
    freq = env.cmd["frequency"].sum(dim=1)
    freq_penalty = -0.03 * cx * freq

    return foot_stance_reward + foot_clear_reward + foot_slip_reward + foot_zvel_reward + freq_penalty

def joint_constraints(env):
    joint_pos = env.scene["ruff"].data.joint_pos
    default_pos = env.scene["ruff"].data.default_joint_pos
    diff = joint_pos - default_pos
    norm_sq = torch.sum(diff ** 2, dim=1)
    cmd_x = mdp.generated_commands(env, command_name="velocity_command")[:, 0]
    cx = 1.0 / cmd_x.abs().clamp_min(1e-3)
    constraint_reward = cx * norm_sq
    return constraint_reward

def RG_phase(env):
    feet = ["fl_contact", "fr_contact", "rl_contact", "rr_contact"]
    contact_states = []

    phase = env.cmd["phase"]
    stance_mask = (phase < torch.pi).float()  
    
    for f in feet:
        data = env.scene.sensors[f].data
        forces = torch.linalg.norm(data.net_forces_w, dim=-1).squeeze(-1)  # [num_envs]
        contact_states.append((forces > 1.0).float())
    contact_tensor = torch.stack(contact_states, dim=1)  # [num_envs, 4]
    match = (contact_tensor == stance_mask).float()
    reward = match.sum(dim=1) 
    return reward

    

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    forward_vel = RewTerm(func=velocity_tracking, params={"key": "x", "slope": config["forward_velocity"]["slope"]}, weight=config["forward_velocity"]["weight"])
    lateral_vel = RewTerm(func=velocity_tracking, params={"key": "y", "slope": config["lateral_velocity"]["slope"]}, weight=config["lateral_velocity"]["weight"])
    angular_vel = RewTerm(func=velocity_tracking, params={"key": "z", "slope": config["angular_velocity"]["slope"]}, weight=config["angular_velocity"]["weight"])
    balance = RewTerm(func=balance_reward, params={"vz_slope": config["balance"]["vz_slope"], "wxy_slope": config["balance"]["wxy_slope"]}, weight=config["balance"]["weight"])
    twist = RewTerm(func=twist, weight=config["twist"]["weight"])
    foot_rhythm = RewTerm(func=rhythm_rewards, params={"c1": config["foot_rhythm"]["c1"], "height_thresh": config["foot_rhythm"]["height_thresh"], "offset": config["foot_rhythm"]["offset"]}, weight=config["foot_rhythm"]["weight"])
    joint_limits = RewTerm(func=joint_constraints, weight=config["joint_constraints"]["weight"])
    rg_phase = RewTerm(func=RG_phase, weight=config["rg_phase"]["weight"])

