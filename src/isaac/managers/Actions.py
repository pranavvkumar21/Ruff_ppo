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
import json
with open("../../config/config.json", "r") as f:
    config = json.load(f)

N_PHASE = 4
N_JOINTS = len(config["joint_names"])

class PhaseAction(ActionTerm):
    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self.asset_name = cfg.asset_name          # e.g., "ruff"
        self._raw  = None
        self._proc = None
        self.env = env

        # ensure per-env buffer exists
        if not hasattr(env, "cmd"):
            self.env.cmd = {}
        if "phase" not in self.env.cmd:
            E, d = env.scene.num_envs, env.device
            self.env.cmd["phase"] = torch.zeros(E, N_PHASE, device=d)

    @property
    def action_dim(self) -> int:
        return N_PHASE+N_JOINTS
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw
    @property
    def processed_actions(self) -> torch.Tensor:
        # treat actions as frequencies ≥ 0
        return self._proc

    def process_actions(self, actions: torch.Tensor) -> None:
        #convert actions[12:] to absolute value
        self._raw = actions.clone()
        actions = il_clip(actions, (-1.0, 1.0))  
        actions[:, :N_JOINTS] = actions[:, :N_JOINTS]*18*torch.pi/180.0
        actions[:, N_JOINTS:] = actions[:, N_JOINTS:].abs()

        self._proc = actions

    def apply_actions(self) -> None:
        self.env.cmd["target"] += self._proc[:, :N_JOINTS]
        self.env.cmd["frequency"] = self._proc[:, N_JOINTS:]
        self.env.cmd["phase"] += 2.0 * torch.pi * self.env.cmd["frequency"] * self.env.step_dt
        self.env.scene[self.asset_name].set_joint_position_target(self.env.cmd["target"])


@configclass
class ActionsCfg:
    # joint_pos = mdp.RelativeJointPositionActionCfg(
    #     asset_name="ruff",
    #     joint_names=config["joint_names"],
    #     scale=18*torch.pi/180.0,  # 2 degrees
    # )
    phase = ActionTermCfg(class_type=PhaseAction, asset_name="ruff", clip={".*":(-1.0, 1.0)})


