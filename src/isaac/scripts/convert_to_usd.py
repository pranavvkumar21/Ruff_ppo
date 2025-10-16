#!/usr/bin/env python3
import os, sys, xml.etree.ElementTree as ET
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pxr import Usd, UsdGeom

URDF_IN  = os.path.abspath("../urdf/ruff_2.urdf")
OUT_DIR  = os.path.abspath("../urdf/ruff_usd")
USD_OUT  = os.path.join(OUT_DIR, "ruff.usd")
os.makedirs(OUT_DIR, exist_ok=True)

def resolve_pkg(path):
    if not path.startswith("package://"): return None
    try:
        from ament_index_python.packages import get_package_share_directory as share_dir
        pkg = path.split("package://",1)[1].split("/",1)[0]
        tail = path.split(pkg+"/",1)[1] if "/" in path.split("package://",1)[1] else ""
        return os.path.join(share_dir(pkg), tail)
    except Exception:
        return None

def preflight_meshes(urdf_path):
    base = os.path.dirname(urdf_path)
    miss, hits = [], 0
    tree = ET.parse(urdf_path)
    for i, m in enumerate(tree.iterfind(".//mesh"), 1):
        fn = m.get("filename") or ""
        cand = resolve_pkg(fn) if fn.startswith("package://") else None
        if cand is None:
            cand = fn if os.path.isabs(fn) else os.path.normpath(os.path.join(base, fn))
        ok = os.path.exists(cand)
        print(f"[{i:03d}] "+(("OK  " if ok else "MISS")+"  "+fn+"  ->  "+cand))
        hits += int(ok)
        if not ok: miss.append((fn, cand))
    print(f"preflight_summary hits {hits} miss {len(miss)}")
    return miss

def scan_stage(usd_path, label):
    stage = Usd.Stage.Open(usd_path)
    stage.Load()  # load payloads
    kinds = {"Mesh": UsdGeom.Mesh, "Sphere": UsdGeom.Sphere, "Cube": UsdGeom.Cube,
             "Capsule": UsdGeom.Capsule, "Cylinder": UsdGeom.Cylinder, "Cone": UsdGeom.Cone, "Points": UsdGeom.Points}
    counts, empties = {}, []
    for name, T in kinds.items():
        items = [p for p in stage.Traverse() if p.IsA(T)]
        counts[name] = len(items)
        print(f"{label}: {name} count {len(items)}")
        if name == "Mesh":
            for p in items[:3]:
                pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
                if not pts: empties.append(p.GetPath().pathString)
    print(f"{label}: empty_meshes {len(empties)}")
    any_geom = sum(counts.values()) > 0
    print(f"{label}: any_geometry {any_geom}")
    return any_geom, counts, empties

if not os.path.exists(URDF_IN):
    print("urdf not found", URDF_IN); sys.exit(2)
if preflight_meshes(URDF_IN):
    print("fix missing meshes before convert"); app.close(); sys.exit(4)

cfg = UrdfConverterCfg(
    asset_path=URDF_IN,
    usd_dir=OUT_DIR,
    usd_file_name="ruff.usd",
    fix_base=False,
    merge_fixed_joints=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        target_type="none",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    ),
    # optional: uncomment to bake geometry into main file instead of payloads
    # make_instanceable=False,
)
conv = UrdfConverter(cfg)
usd_main = conv.usd_path
usd_meshes_rel = conv.usd_instanceable_meshes_path  # file that holds the meshes
usd_meshes = os.path.join(conv.usd_dir, usd_meshes_rel) if usd_meshes_rel else None
print("usd_main", os.path.exists(usd_main), usd_main)
print("usd_meshes", bool(usd_meshes) and os.path.exists(usd_meshes), usd_meshes)

ok_main, c_main, e_main = scan_stage(usd_main, "main")
ok_inst, c_inst, e_inst = (False, {}, [])
if usd_meshes and os.path.exists(usd_meshes):
    ok_inst, c_inst, e_inst = scan_stage(usd_meshes, "inst")

print("conversion_ok=", ok_main or ok_inst)
print("conversion_counts=", {"main": c_main, "inst": c_inst})
print("conversion_empties=", {"main": len(e_main), "inst": len(e_inst)})
if not (ok_main or ok_inst):
    print("no geometry found in either file"); app.close(); sys.exit(5)
app.close()
