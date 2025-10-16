#!/usr/bin/env python3
from isaaclab.app import AppLauncher
import carb.settings as cs
import time
import numpy as np
import torch
# Launch with livestream enabled before Kit boots

simulation_app = AppLauncher( headless=False, livestream=2).app

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains.height_field import hf_terrains, hf_terrains_cfg as hf_cfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from pxr import Usd, UsdGeom, UsdPhysics
import omni.physics.tensors as phys
from isaaclab.sensors import ContactSensorCfg

class robot_scene(InteractiveSceneCfg):

        # Add any additional setup here if needed
    ROBOT = ArticulationCfg( spawn=sim_utils.UsdFileCfg(
            usd_path="../../urdf/ruff_usd/ruff.usd",activate_contact_sensors=True,),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.9),
            ),
            prim_path="{ENV_REGEX_NS}/ruff_new",
            actuators={"joint-acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=50, stiffness=1500)},)
    all_contacts=ContactSensorCfg(
        prim_path=fr"{{ENV_REGEX_NS}}/ruff_new/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["/World/defaultGroundPlane"]  # optional
    )
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg())
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=TerrainGeneratorCfg(
    #         num_rows=6, num_cols=6,
    #         size=(8.0, 8.0),
    #         horizontal_scale=0.005,
    #         vertical_scale=0.001,
    #         sub_terrains={
    #             "rough": hf_cfg.HfRandomUniformTerrainCfg(
    #                 noise_range=(-0.05, 0.05),
    #                 noise_step=0.01
    #             )
    #         },
    #     ),
    # )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))

sim_cfg = sim_utils.SimulationCfg(dt=0.01)
sim = sim_utils.SimulationContext(sim_cfg)

num_envs = 32
scene_cfg = robot_scene(num_envs, env_spacing=4.0)
scene = InteractiveScene(scene_cfg)


print("Simulation created")

sim.reset()

sim_dt = sim.get_physics_dt()
sim_time = 0.0



import omni.usd
stage = omni.usd.get_context().get_stage()




stage = omni.usd.get_context().get_stage()
root="/World/envs/env_0/ruff_new"
print("robot_root_exists=", stage.GetPrimAtPath(root).IsValid())

robot = scene["ROBOT"]
print(robot.joint_names)

print("joint position", robot.data.joint_pos[0].cpu().numpy())
print("joint velocity", robot.data.joint_vel[0].cpu().numpy())
nj = robot.num_joints
# robot.write_joint_state_to_sim(joint_pos=np.zeros(nj), joint_vel=np.zeros(nj))
print("After reset")
print("joint position", robot.data.joint_pos[0].cpu().numpy())
base_p   = robot.data.root_pos_w[0].cpu().numpy()
base_qxyzw = robot.data.root_quat_w[0].cpu().numpy()
print("base pos", base_p)
print("base quat", base_qxyzw)
base_z=float(robot.data.root_pos_w[0,2])




print("Starting simulation")
count=0
while simulation_app.is_running():
    wave_action = scene["ROBOT"].data.default_joint_pos
    wave_action[:, 0:4] = np.sin(2 * np.pi * 0.5 * sim_time)
    # wave_action[:, 3] = np.sin(2 * np.pi * 0.5 * sim_time)
    # wave_action[:, 6] = np.sin(2 * np.pi * 0.5 * sim_time)
    # wave_action[:, 9] = np.sin(2 * np.pi * 0.5 * sim_time)

    scene["ROBOT"].set_joint_position_target(wave_action)
    scene.write_data_to_sim()
    sim.step()
    sim_time += sim_dt
    scene.update(sim_dt)
    if count%20==0 and count<4000:
        base_p   = robot.data.root_pos_w[0].cpu().numpy()
        print("base pos", base_p[2])
        print(robot.joint_names)
        print("action", wave_action[0,0])
    # print("joint position", robot.data.joint_pos[0].cpu().numpy())
    count += 1
simulation_app.close()
