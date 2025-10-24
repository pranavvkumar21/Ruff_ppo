#!/usr/bin/env python3
from isaaclab.managers import (
    ActionTermCfg,
    SceneEntityCfg,
    ActionTerm)
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import clip as il_clip
import isaaclab.envs.mdp as mdp
from pathlib import Path
import yaml
import os

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "ruff_config.yaml", "r") as f:
    config = yaml.safe_load(f)

N_PHASE = 4
N_JOINTS = len(config["scene"]["joint_names"])

class PhaseAction(ActionTerm):
    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self.asset_name = cfg.asset_name          # e.g., "ruff"
        self._raw  = None
        self._proc = None
        self.env = env
        self.joint_ids = [env.scene["ruff"].data.joint_names.index(name) for name in config["scene"]["joint_names"]]

        # ensure per-env buffer exists
        if not hasattr(env, "cmd"):
            self.env.cmd = {}
        if "phase" not in self.env.cmd:
            E, d = env.scene.num_envs, env.device
            self.env.cmd["phase"] = torch.zeros(E, N_PHASE, device=d)

    @property
    def action_dim(self) -> int:
        return N_PHASE #+ N_JOINTS
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw
    @property
    def processed_actions(self) -> torch.Tensor:
        # treat actions as frequencies ≥ 0
        return self._proc

    def process_actions(self, actions: torch.Tensor) -> None:

        self._raw = actions.clone()
        actions = actions.abs()
        self._proc = actions

    def apply_actions(self) -> None:
        self.env.cmd["frequency"] = self._proc
        delta = 2.0 * torch.pi * self.env.cmd["frequency"] * self.env.step_dt
        self.env.cmd["phase"] = torch.remainder(self.env.cmd["phase"] + delta, 2.0 * torch.pi)



@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="ruff",
        joint_names=config["scene"]["joint_names"],
        scale=6*torch.pi/180.0,  # 4 degrees
    )
    phase = ActionTermCfg(class_type=PhaseAction, asset_name="ruff", clip={".*":(-1.0, 1.0)})


