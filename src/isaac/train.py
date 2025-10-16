
from isaaclab.app import AppLauncher
simulation_app = AppLauncher(headless=True, livestream=0, kit_args="--/log/level=error").app
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import gymnasium as gym
from Registration import register_envs
from RuffEnv import RuffEnvCfg

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
import os

import argparse
p = argparse.ArgumentParser()
# define command line arguments
p.add_argument("--mode", choices=["train", "eval"], default="eval", help="Mode: train or eval")
args = p.parse_args()

model_path = '../../models/isaac_models/'
prefix = 'ruu_ppo_model'
testing_mode = False  # Set to True to load the second latest model instead of the latest
def get_latest_model_path(folder_path, prefix):
    folders = os.listdir(folder_path)
    folders.sort(reverse=True)
    print(folders)
    if not testing_mode:
        latest_folder = folders[0]
    else:
        latest_folder = folders[0]
    print(latest_folder)
    folder_path = os.path.join(folder_path,latest_folder)
    print("-"*30)
    print(latest_folder)

def main():
    register_envs()
    cfg = RuffEnvCfg()
    cfg.scene.num_envs = 4096
    env = gym.make("Ruff-v0", cfg=cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=True)
    # save_model_path,_ = get_latest_model_path(model_path, prefix)
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=128,
        batch_size=8192,
        learning_rate=3.5e-4,
        clip_range=0.2,
        gamma=0.992,
        ent_coef=0.0025,
        verbose=1,
        use_sde=False,
        policy_kwargs=dict(net_arch=[256, 256], log_std_init=0.0, full_std=True),  # Adjust the policy architecture if needed
        tensorboard_log="../logs/ppo_pybullet_tensorboard/",
    )
    if args.mode == "train":
        print("Training Mode")
        model.learn(total_timesteps=98e6)
    else:
        print("Evaluation Mode")
        obs = env.reset()
        print(obs.shape)

        for i in range(1):
            # action = model.predict(obs)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            print(info[0])
            if dones:
                obs = env.reset()

if __name__ == "__main__":
    main()