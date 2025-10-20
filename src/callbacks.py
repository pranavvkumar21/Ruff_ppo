#!/usr/bin/env python3
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
import os, time
from pathlib import Path
from natsort import natsorted
ROOT = Path(__file__).resolve().parent.parent

def get_latest_model_path(folder_path, prefix):
    checkpoint_folders = os.listdir(folder_path)
    if not checkpoint_folders:
        return None
    # print(f"Checkpoint folders found: {checkpoint_folders}")

    checkpoint_folders  = natsorted(checkpoint_folders)
    latest_folder = os.path.join(folder_path, checkpoint_folders[-1])
    # print(f"Latest checkpoint folder: {latest_folder}")

    checkpoint_files = [f for f in os.listdir(latest_folder) if f.endswith(".zip")]
    latest = natsorted(checkpoint_files)[-1] if checkpoint_files else None
    # print(f"Checkpoints in latest folder: {checkpoint_files}")
    
    return os.path.join(latest_folder, latest) if latest else None

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_idx = 0

    def _on_rollout_end(self):
        self.rollout_idx += 1

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for info in infos:
            episode_data = info.get('episode', None)
            if not episode_data:
                continue
            for k, v in episode_data.items():
                if not isinstance(k, str):
                    continue
                if isinstance(v, (float, int)):
                    self.logger.record(f"{k}", v)
                elif hasattr(v, "item"):
                    self.logger.record(f"{k}", v.item())
                elif isinstance(v, (list, np.ndarray)):
                    self.logger.record(f"{k}_mean", np.mean(v))
        return True


def create_checkpoint_callback(load=False):
    if not load:
        run_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        root = Path(__file__).resolve().parent.parent
        save_dir = root / "models" / f"run_{run_time}"
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        latest_model_path = get_latest_model_path(str(ROOT / "models"), "ruff_ppo_model")
        if latest_model_path is None:
            print("No saved model found. run with --load False to train a new model")
            print("Exiting...")
            exit(0)
        save_dir = Path(latest_model_path).parent
    print(f"CheckpointCallback: Models will be saved to {save_dir}")
    return CheckpointCallback(
        save_freq=3000,
        save_path=str(save_dir),
        name_prefix="ruff_ppo_model"
    )


if __name__ == "__main__":
    print("This is a callback module and is not meant to be run directly.")
    print(f"{ROOT}")