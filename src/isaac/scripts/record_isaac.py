#!/usr/bin/env python3
import os, sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg



USD_IN = os.path.abspath("../urdf/ruff_usd/ruff.usd")
if not os.path.exists(USD_IN):
    print("usd not found", USD_IN); sys.exit(2)

class SceneCfg(InteractiveSceneCfg):
    ROBOT = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ruff_new",
        spawn=UsdFileCfg(usd_path=USD_IN),
        actuators={"acts": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
    )

sim = SimulationContext(SimulationCfg(dt=0.01))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0))

sim.reset(); scene.reset(); sim.step()

import omni.usd
from pxr import Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
root = "/World/envs/env_0/ruff_new"

# include instance proxies so payload geometry is visible
pred = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
mesh_paths = [p.GetPath().pathString for p in stage.Traverse(pred) if p.IsA(UsdGeom.Mesh) and p.GetPath().pathString.startswith(root)]
prim_paths = [p.GetPath().pathString for p in stage.Traverse(pred) if p.GetPath().pathString.startswith(root)]

print("robot_root_exists=", stage.GetPrimAtPath(root).IsValid())
print("mesh_count_after_spawn=", len(mesh_paths))
for p in mesh_paths[:20]:
    print("mesh", p)
print("prim_samples", prim_paths[:10])

app.close()
