#! /usr/bin/env python3
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.managers import  (ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg)

import torch

joint_names = [
            "FL1_", "FR1_", "RL1_", "RR1_",
            "FL2_", "FR2_", "RL2_", "RR2_",
            "FL3_", "FR3_", "RL3_", "RR3_",
        ]


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
def target_obs(env, key="target"):
    joints = env.scene["ruff"].data.joint_pos
    if not hasattr(env, "cmd") or key not in env.cmd:
        print("Warning: target not initialized yet!")
        env.cmd = {}
        return torch.zeros_like(joints)
    return env.cmd[key] - joints

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

