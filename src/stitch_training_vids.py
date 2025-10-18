#!/usr/bin/env python3
import os
import re
import cv2
import argparse
import subprocess
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = os.path.join("..", "outputs")


def latest_run_videos(root=OUTPUT_DIR):
    runs = [p for p in Path(root).iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit(f"No run folders in {root}")
    run = max(runs, key=lambda d: d.stat().st_mtime)

    vids = [p for p in run.glob("*.mp4")]
    if not vids:
        raise SystemExit(f"No mp4s in {run}")

    def stepnum(p):
        m = re.search(r"(\d+)_steps", p.stem)
        return int(m.group(1)) if m else -1

    vids.sort(key=lambda p: (stepnum(p), p.stat().st_mtime))
    return run, vids


def extract_steps_millions(path):
    m = re.search(r"(\d+)_steps", Path(path).stem)
    steps = int(m.group(1)) if m else 0
    return steps, steps / 1_000_000.0


def draw_text_right(frame, text, margin=24):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    x = max(margin, w - tw - margin)
    y = margin + th
    # Black outline
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),
                4, cv2.LINE_AA)
    # White text
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                2, cv2.LINE_AA)


def try_fix_with_ffmpeg(src_path):
    if shutil.which("ffmpeg") is None:
        return None
    dst_path = src_path.with_suffix(".fixed.mp4")
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(src_path),
        "-c", "copy", "-movflags", "+faststart", str(dst_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return dst_path if dst_path.exists() and dst_path.stat().st_size > 0 else None


def safe_open_video(p):
    cap = cv2.VideoCapture(str(p))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if ok:
        return str(p)

    fixed = try_fix_with_ffmpeg(p)
    if fixed:
        cap2 = cv2.VideoCapture(str(fixed))
        ok2 = cap2.isOpened() and cap2.read()[0]
        cap2.release()
        if ok2:
            return str(fixed)

    print(f"Skipping {p}")
    return None


def stitch(out_path, fps_override=None):
    _, vids = latest_run_videos()

    # find first valid video
    first_ok = None
    for v in vids:
        cand = safe_open_video(v)
        if cand:
            first_ok = cand
            break
    if not first_ok:
        raise SystemExit("No readable videos")

    cap0 = cv2.VideoCapture(first_ok)
    fps = fps_override or cap0.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    out = cv2.VideoWriter(out_path,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))
    if not out.isOpened():
        raise SystemExit("Failed to open VideoWriter")

    for vid in tqdm(vids, desc="Processing videos", unit="video"):
        readable = safe_open_video(vid)
        if not readable:
            continue

        steps, msteps = extract_steps_millions(readable)
        label = f"{msteps:.2f}M steps"

        cap = cv2.VideoCapture(readable)
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h),
                                   interpolation=cv2.INTER_AREA)

            draw_text_right(frame, label)
            out.write(frame)

        cap.release()

    out.release()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=os.path.join(OUTPUT_DIR, "stitched.mp4"))
    ap.add_argument("--fps", type=int, default=0)
    args = ap.parse_args()

    stitch(args.out, fps_override=(args.fps or None))
