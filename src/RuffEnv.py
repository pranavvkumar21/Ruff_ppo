#!/usr/bin/env python3
# run_env.py

if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True, livestream=2).app

import torch
import time
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.managers import (
    SceneEntityCfg,
)

from isaaclab.sim.schemas import define_rigid_body_properties
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from Scene import RuffSceneCfg
from managers.Observations import ObservationsCfg
from managers.Actions import ActionsCfg
from managers.Events import EventsCfg
from managers.Commands import CommandsCfg
from managers.Rewards import RewardsCfg
from managers.Terminations import TerminationsCfg
from managers.Curriculum import CurriculumCfg


@configclass
class RuffEnvCfg(ManagerBasedRLEnvCfg):
    scene = RuffSceneCfg(num_envs=4, env_spacing=5)
    actions = ActionsCfg()
    events = EventsCfg()

    observations = ObservationsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    curriculum = CurriculumCfg()

    def __post_init__(self):
        
        self.decimation = 10
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        self.max_episode_length = 1000
        self.episode_length_s = 10

def main():
    cfg = RuffEnvCfg()
    cfg.viewer.enable = False
    env = ManagerBasedRLEnv(cfg=cfg)
    n_steps = 1000

    # tiny warmup for JITs and GPU clocks
    for _ in range(1):
        with torch.inference_mode():
            env.step(torch.randn_like(env.action_manager.action))
            # print(env.scene["ruff"].data.body_names)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    offset = 0.007  # height offset from ray hit to foot bottom
    feet = ["fl_contact", "fr_contact", "rl_contact", "rr_contact"]
    for step in range(n_steps):
        contacts = []
        with torch.inference_mode():
            # env.step(torch.randn_like(env.action_manager.action))
            obs, rewards, dones, trunc, info = env.step(torch.zeros_like(env.action_manager.action))
            for f in feet:
                    data = env.scene.sensors[f].data
                    # IsaacLab ContactSensorData fields:
                    #   net_forces_w: [num_envs, 3]
                    #   net_moments_w: [num_envs, 3]
                    #   num_prim_contacts: [num_envs]
                    #   force_matrix_w: [num_envs, N, 3]
                    print(f"net forces shape: {data.net_forces_w.shape}")
                    forces = torch.linalg.norm(data.net_forces_w, dim=-1).squeeze(-1)  # [num_envs]
                    contacts.append(forces > 1.0) 
            print(step)
            print(torch.stack(contacts, dim=1).float().shape)


    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    dt = t1 - t0
    action_fps = n_steps / dt
    sim_fps = action_fps * cfg.decimation
    frames_total = n_steps * cfg.scene.num_envs
    #total time for 98 million frames on 1 GPU
    total_time_hours = (98e6/frames_total)*dt/3600

    print(f"t {dt:.3f}s for {n_steps} steps")
    print(f"action_fps {action_fps:.1f}")
    print(f"sim_fps {sim_fps:.1f}")
    print(f"frames_processed {frames_total}")
    print(f"total_time_hours {total_time_hours:.1f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
