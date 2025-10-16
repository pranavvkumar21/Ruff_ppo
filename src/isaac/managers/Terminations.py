#!/usr/bin/env python3
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp



def body_touch(env):
    data = env.scene.sensors["body_contact"].data
    F = data.net_forces_w          # [env, bodies, 3]
    in_contact = (F.norm(dim=-1) > 1.0).any(dim=-1)
    return in_contact

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Body contact with ground
    body_contact = DoneTerm(func=body_touch)
    