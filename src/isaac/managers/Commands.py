#!/usr/bin/env python3
from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.utils import configclass
from isaaclab.envs import mdp
import torch
import yaml
with open("../../config/ruff_config.yaml", "r") as f:
    config = yaml.safe_load(f)
joint_names = config["scene"]["joint_names"]

@configclass
class JointPosCommandCfg(CommandTermCfg):
    class_type: type = None
    asset_name: str = "ruff"
    joint_names = joint_names  # None means all joints


@configclass
class CommandsCfg:
    velocity_command = mdp.UniformVelocityCommandCfg(
        asset_name="ruff",
        heading_command=False,                 # use heading instead of direct ang vel
        heading_control_stiffness=1.0,        # convert heading error to ang vel
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,                 # all envs use heading mode
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 2),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),            # kept for completeness          # required when heading_command is True
        ),
        resampling_time_range=(3.0, 8.0),
    )