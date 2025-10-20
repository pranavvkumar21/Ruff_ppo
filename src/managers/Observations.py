#! /usr/bin/env python3
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.managers import  (ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg)
import yaml
import torch
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "ruff_config.yaml", "r") as f:
    config = yaml.safe_load(f)

joint_names = config["scene"]["joint_names"]



def phase_obs(env, key="phase", use_trig=True):
    if not hasattr(env, "cmd") or key not in env.cmd:
        print("Warning: phase not initialized yet!")
        env.cmd = {}
        return torch.zeros((env.scene.num_envs, 8), device=env.device)
    ph = env.cmd[key]
    return torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1) if use_trig else ph
def freq_obs(env, key="frequency"):
    if not hasattr(env, "cmd") or key not in env.cmd:
        print("Warning: frequency not initialized yet!")
        env.cmd = {}
        return torch.zeros((env.scene.num_envs, 4), device=env.device)
    return env.cmd[key]
def target_obs(env):
    joint_ids = [env.scene["ruff"].data.joint_names.index(name) for name in joint_names]
    joints = env.scene["ruff"].data.joint_pos[:,joint_ids]
    target = env.scene["ruff"].data.joint_pos_target[:,joint_ids]
    return (target - joints)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "velocity_command"})  # must match CommandsCfg field name
        root_linear_velocity  = ObsTerm(func=mdp.base_lin_vel,  params={"asset_cfg": SceneEntityCfg("ruff")})
        root_angular_velocity = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("ruff")})
        root_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("ruff")})
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("ruff",joint_names=joint_names)})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("ruff",joint_names=joint_names)})
        # target_feat = ObsTerm(func=mdp.generated_commands, params={"command_name": "joint_init"})  # must match CommandsCfg field name
        position_error = ObsTerm(func=target_obs)
        phase = ObsTerm(func=phase_obs, )
        frequency = ObsTerm(func=freq_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

