#!/usr/bin/env python3
from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.utils import configclass
from isaaclab.envs import mdp
import torch
joint_names = [
            "FL1_", "FR1_", "RL1_", "RR1_",
            "FL2_", "FR2_", "RL2_", "RR2_",
            "FL3_", "FR3_", "RL3_", "RR3_",
        ]

@configclass
class JointPosCommandCfg(CommandTermCfg):
    class_type: type = None
    asset_name: str = "ruff"
    joint_names = joint_names  # None means all joints

class JointPosCommand(CommandTerm):
    def __init__(self, cfg: JointPosCommandCfg, env):
        super().__init__(cfg, env)
        cfg.class_type = JointPosCommand
        self._env = env
        self._asset = env.scene[cfg.asset_name]
        names = self._asset.data.joint_names  # ordered
        if cfg.joint_names:
            idxs = [names.index(n) for n in cfg.joint_names]
        else:
            idxs = list(range(len(names)))
        self._jids = torch.tensor(idxs, device=env.device, dtype=torch.long)
        self._cmd = torch.zeros((env.num_envs, len(idxs)), device=env.device, dtype=self._asset.data.joint_pos.dtype)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        q = self._asset.data.joint_pos.index_select(1, self._jids)[env_ids]
        self._cmd[env_ids] = q
        return {"joint_pos_init": q}

    def compute(self, dt: float):
        return
    def _resample_command(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        q = self._asset.data.joint_pos.index_select(1, self._jids)[env_ids]
        self._cmd[env_ids] = q

    def _update_command(self, env_ids=None):
        return

    def _update_metrics(self, env_ids=None):
        return {}


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