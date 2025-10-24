#! /usr/bin/env python3
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.envs import ManagerBasedRLEnv
from typing import Sequence
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config" / "ruff_curriculum_config.yaml", "r") as f:
    config = yaml.safe_load(f)


class RewardCurriculum(ManagerTermBase):
    def __init__(self, cfg: CurrTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
                # obtain term configuration
        self._term_cfg = {}
        self._weight = {}
        start_frac = cfg.params["start_frac"]
        term_names = cfg.params["term_names"]
        for term_name in term_names:

            self._term_cfg[term_name] = env.reward_manager.get_term_cfg(term_name)
            self._weight[term_name] = self._term_cfg[term_name].weight
            self._term_cfg[term_name].weight = self._weight[term_name] * start_frac  
            # print(f"Initialized curriculum for term '{term_name}' with initial weight {self._weight[term_name]}")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_names: Sequence[str],
        num_steps: int,
        start_frac: float = 0.0,
    ) -> float:
        # update term settings
        if env.common_step_counter <= num_steps:
            for term_name in term_names:
                # weight = start_frac x current weight + (1 - start_frac) x current weight x (current step / total steps)
                weight = ((self._weight[term_name] * start_frac) + (((1 - start_frac) * self._weight[term_name]) * (env.common_step_counter / num_steps)))
                self._term_cfg[term_name].weight = weight
                env.reward_manager.set_term_cfg(term_name, self._term_cfg[term_name])
            if env.common_step_counter % 100 == 0:
                print(f"curriculum percentage applied: {env.common_step_counter/num_steps*100:.2f}%")
        return None

# def rhythm_curriculum(env, env_ids, term_name, start=0.0, end=1, steps=1000):
#     weight = ((end - start) * (env.common_step_counter / steps) )+ start
#     self._term_cfg = env.reward_manager.get_term_cfg(term_name)
#     self._term_cfg.weight = weight
#     env.reward_manager.set_term_cfg(term_name, self._term_cfg)
#     return weight


@configclass
class CurriculumCfg:
    reward_curriculum = CurrTerm(
        func=RewardCurriculum,
        params={"term_names": config["reward_curriculum"]["reward_terms"], 
            "num_steps": config["reward_curriculum"]["num_steps"],
            "start_frac": config["reward_curriculum"]["start_frac"]}
    )