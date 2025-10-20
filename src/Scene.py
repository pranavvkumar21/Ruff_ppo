#!/usr/bin/env python3
# ruff_env.py
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains import HfRandomUniformTerrainCfg, HfDiscreteObstaclesTerrainCfg, HfPyramidSlopedTerrainCfg
from isaaclab.terrains import HfPyramidStairsTerrainCfg, HfWaveTerrainCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from math import ceil, sqrt
import yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "config" / "ruff_config.yaml", "r") as f:
    config = yaml.safe_load(f)

rows = config["scene"]["rows"]
cols = config["scene"]["cols"]
size = (config["scene"]["env_spacing"], config["scene"]["env_spacing"])

terrain_gen = TerrainGeneratorCfg(
    size=size,
    num_rows=rows,
    num_cols=cols,
    seed=42,
    sub_terrains={
        "random": HfRandomUniformTerrainCfg(
            noise_range=(0.0, 0.02),
            noise_step=0.01,
            border_width=0.1,

        ),
        # "pyramid": HfPyramidSlopedTerrainCfg(
        #     slope_range=(0.15, 0.3),
        #     platform_width=0.3,
        #     border_width=0.1,
        #     inverted=True,
        # ),
        # "stairs": HfPyramidStairsTerrainCfg(
        #     step_height_range=(0.05, 0.1),
        #     step_width=0.2,
        #     border_width=0.1,
        #     inverted=True,
        # ),
        "waves": HfWaveTerrainCfg(
            amplitude_range=(0.02, 0.05),
            num_waves=3,
            border_width=0.1,
        ),

    }
)

class RuffSceneCfg(InteractiveSceneCfg):
    replicate_physics = False
    ruff = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ruff",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ROOT}/urdf/ruff_usd/ruff.usd", activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.45)),
        actuators={"joint-acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=50, stiffness=1500,effort_limit=18)},
    )
    # ground = AssetBaseCfg(prim_path="/World/terrain", spawn=sim_utils.GroundPlaneCfg())
    terrain_importer = TerrainImporterCfg(
    prim_path="/World/terrain",
    terrain_type="generator",
    terrain_generator=terrain_gen,

    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    #define leg contact sensors
    fl_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ruff/FL3",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/terrain"],
        force_threshold=1.0,          
        debug_vis=False,
    )
    fr_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ruff/FR3",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/terrain"],
        force_threshold=1.0,         
        debug_vis=False,
    )
    rl_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ruff/RL3",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/terrain"],
        force_threshold=1.0,          
        debug_vis=False,
    )
    rr_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ruff/RR3",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/terrain"],
        force_threshold=1.0,      
        debug_vis=False,
    )
    #define body contact sensor group
    body_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ruff/(base_link|FL1|FR1|RL1|RR1|FL2|FR2|RL2|RR2)$",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["/World/terrain"],
        force_threshold=1.0,      
        debug_vis=False,
    )
    #define foot height raycasters
    fl3_height = RayCasterCfg(
        mesh_prim_paths=["/World/terrain"],
        prim_path="{ENV_REGEX_NS}/ruff/FL3_f",
        ray_alignment="world",
        update_period=0.0,
        history_length=1,
        pattern_cfg=GridPatternCfg(
            resolution=0.02,
            size=(0.01, 0.01),
            direction=(0.0, 0.0, -1.0),
        )
    )
    fr3_height = RayCasterCfg(
        mesh_prim_paths=["/World/terrain"],
        prim_path="{ENV_REGEX_NS}/ruff/FR3_f",
        ray_alignment="world",
        update_period=0.0,
        history_length=1,
        pattern_cfg=GridPatternCfg(
            resolution=0.02,
            size=(0.01, 0.01),
            direction=(0.0, 0.0, -1.0),
        )
    )
    rl3_height = RayCasterCfg(
        mesh_prim_paths=["/World/terrain"],
        prim_path="{ENV_REGEX_NS}/ruff/RL3_f",
        ray_alignment="world",
        update_period=0.0,
        history_length=1,
        pattern_cfg=GridPatternCfg(
            resolution=0.02,
            size=(0.01, 0.01),
            direction=(0.0, 0.0, -1.0),
        )
    )
    rr3_height = RayCasterCfg(
        mesh_prim_paths=["/World/terrain"],
        prim_path="{ENV_REGEX_NS}/ruff/RR3_f",
        update_period=0.0,
        history_length=1,
        ray_alignment="world",
        pattern_cfg=GridPatternCfg(
            resolution=0.02,
            size=(0.01, 0.01),
            direction=(0.0, 0.0, -1.0),
        )
    )
