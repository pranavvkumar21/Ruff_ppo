#! /usr/bin/env python3
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.envs import ManagerBasedRLEnv
from typing import Sequence


class RewardCurriculum(ManagerTermBase):
    def __init__(self, cfg: CurrTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
                # obtain term configuration
        self._term_cfg = {}
        self._weight = {}
        term_names = cfg.params["term_names"]
        for term_name in term_names:

            self._term_cfg[term_name] = env.reward_manager.get_term_cfg(term_name)
            self._weight[term_name] = self._term_cfg[term_name].weight
            self._term_cfg[term_name].weight = 0.0  # start from zero weight
            # print(f"Initialized curriculum for term '{term_name}' with initial weight {self._weight[term_name]}")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_names: Sequence[str],
        num_steps: int,
    ) -> float:
        # update term settings
        if env.common_step_counter <= num_steps:
            for term_name in term_names:
                weight = ((self._weight[term_name]) * (env.common_step_counter / num_steps))
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
        params={"term_names": ["twist", "foot_rhythm", "joint_limits","rg_phase"], "num_steps": 8000},
    )