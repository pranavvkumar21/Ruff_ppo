#! /usr/bin/env python3


import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from RuffEnv import RuffEnvCfg
import torch

def register_envs():
    gym.register(
        id="Ruff-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"cfg": RuffEnvCfg()},

    )
