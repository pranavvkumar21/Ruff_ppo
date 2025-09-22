#!/usr/bin/env python3
import os, glob, time
import numpy as np
import cv2
from tqdm import tqdm
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from ruff_trainv2 import Ruff_env

MODELS_ROOT = os.path.join("..", "models")
OUTPUT_DIR  = os.path.join("..", "outputs")
VIDEO_STEPS = 2000
FPS, W, H   = 30, 640, 480
FORWARD_CMD = [[0.3, 0.0, 0.0]]
USE_GUI     = True  # set False to render headless

os.makedirs(OUTPUT_DIR, exist_ok=True)

def latest_run_dir(root):
    dirs = [d.path for d in os.scandir(root) if d.is_dir()]
    if not dirs: raise SystemExit(f"No run folders in {root}")
    return max(dirs, key=lambda d: os.stat(d).st_mtime)

def list_checkpoints(run_dir):
    zips = sorted(glob.glob(os.path.join(run_dir, "*.zip")), key=os.path.getmtime)
    if not zips:
        print("no checkpoints found") 
        return []
    return zips

def _cam_mats(target, dist=3.0, yaw=50, pitch=-35, w=W, h=H):
    view = p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=60.0, aspect=w / h, nearVal=0.01, farVal=10.0)
    return view, proj

def record_video(model_path, use_gui=USE_GUI, output_dir=OUTPUT_DIR):
    base = os.path.splitext(os.path.basename(model_path))[0]
    out_mp4 = os.path.join(output_dir, f"{base}.mp4")
    print(f"[INFO] Recording {base} → {out_mp4}")

    env = Ruff_env(render_type="gui" if use_gui else "DIRECT")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    if use_gui:
        p.resetDebugVisualizerCamera(3.0, 50, -35, [0, 0, 0.5])

    model = PPO.load(model_path, env=env)

    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened(): raise RuntimeError("VideoWriter failed to open")

    obs, _ = env.reset(commands=FORWARD_CMD)

    for _ in range(10):
        p.stepSimulation()

    steps_per_sec = int(1.0 / env.timestep / env.sim_steps_per_control_step)  # ≈100
    stride = max(1, steps_per_sec // FPS)                                     # ≈3

    for t in tqdm(range(VIDEO_STEPS), desc=f"recording {base}"):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, d, tr, _ = env.step(act)

        pos, _orn = p.getBasePositionAndOrientation(env.Id)
        target = [pos[0], pos[1], pos[2] + 0.35]
        view, proj = _cam_mats(target, w=W, h=H)

        if t % stride == 0:
            _, _, rgba, _, _ = p.getCameraImage(
                W, H, view, proj,
                renderer=p.ER_BULLET_HARDWARE_OPENGL if use_gui else p.ER_TINY_RENDERER
            )
            
            arr = np.asarray(rgba)
            if arr.dtype.kind == "f":
                arr = (arr * 255.0).clip(0,255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

            frame = arr.reshape(H, W, 4)[..., :3]
            frame = np.ascontiguousarray(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        if d or tr:
            obs, _ = env.reset(commands=FORWARD_CMD)

        if use_gui:
            time.sleep(1 / 240)

    writer.release()
    env.close()
    print(f"[INFO] Saved {out_mp4}")

def main():
    run_dir = latest_run_dir(MODELS_ROOT)
    output_dir = os.path.join(OUTPUT_DIR, os.path.basename(run_dir))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Using run folder {run_dir}")
    completed_checkpoints = []
    while True:
        try:
            for ck in list_checkpoints(run_dir):
                if ck not in completed_checkpoints:
                    record_video(ck, use_gui=USE_GUI, output_dir=output_dir)
                    completed_checkpoints.append(ck)
                    print(f"checkpoint: {ck} saved")
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user, exiting...")
            break
        time.sleep(30*60)
    print(f"[INFO] All videos saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()