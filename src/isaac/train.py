
import argparse
p = argparse.ArgumentParser()
# define command line arguments
p.add_argument("--mode", choices=["train", "eval"], default="eval", help="Mode: train or eval")
args = p.parse_args()
import yaml
with open("../../config/ruff_config.yaml", "r") as f:
    config = yaml.safe_load(f)

if args.mode == "train":
    kit_args="--/log/level=error"
    livestream=0
else:
    livestream=2
    kit_args="--/log/level=warning"

from isaaclab.app import AppLauncher
simulation_app = AppLauncher(headless=True, livestream=livestream, kit_args=kit_args).app


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import gymnasium as gym
from Registration import register_envs
from RuffEnv import RuffEnvCfg

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
import os
import time



def get_latest_model_path(folder_path, prefix):
    files = os.listdir(folder_path)
    if len(files) == 0:
        return None
    files.sort(reverse=True)
    latest_file = files[0]
    file_path= os.path.join(folder_path,latest_file)
    return file_path

def config_env(cfg):
    if args.mode == "train":
        cfg.scene.num_envs = config["train"]["num_envs"]
        cfg.scene.terrain_importer.terrain_generator.rows = config["train"]["rows"]
        cfg.scene.terrain_importer.terrain_generator.cols = config["train"]["cols"]

    else:
        cfg.scene.num_envs = config["eval"]["num_envs"]
        cfg.scene.terrain_importer.terrain_generator.num_rows = config["eval"]["rows"]
        cfg.scene.terrain_importer.terrain_generator.num_cols = config["eval"]["cols"]
        cfg.commands.velocity_command.ranges.lin_vel_x = (0.3, 2.0)
        cfg.commands.velocity_command.ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands.velocity_command.ranges.ang_vel_z = (0.0, 0.0)

    cfg.scene.terrain_importer.terrain_generator.size = (config["scene"]["env_spacing"], config["scene"]["env_spacing"])
    return cfg


def main():
    register_envs()
    cfg = RuffEnvCfg()
    cfg = config_env(cfg)
    env = gym.make("Ruff-v0", cfg=cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=False)
    # save_model_path,_ = get_latest_model_path(model_path, prefix)
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=config["train"]["n_steps"],
        batch_size=config["train"]["batch_size"],
        learning_rate=config["train"]["learning_rate"],
        clip_range=config["train"]["clip_range"],
        gamma=config["train"]["gamma"],
        ent_coef=config["train"]["ent_coef"],
        verbose=1,
        use_sde=False,
        policy_kwargs=dict(net_arch=[256, 256], log_std_init=0.0, full_std=True),  # Adjust the policy architecture if needed
        tensorboard_log="../logs/ppo_pybullet_tensorboard/",
    )
    if args.mode == "train":
        print("Training Mode")
        model.learn(total_timesteps=98e6)
        save_path = config["save_config"]["save_path"] + config["save_config"]["save_prefix"] + time.strftime("%Y-%m-%d_%H-%M-%S")
        model.save(save_path)
    else:
        print("Evaluation Mode")
        latest_model_path = get_latest_model_path(config["save_config"]["save_path"], config["save_config"]["save_prefix"])
        if latest_model_path is None:
            print("No saved model found")
        else:
            print("Loading model from:", latest_model_path)
            model = PPO.load(latest_model_path, env=env)
        obs = env.reset()
        print(obs.shape)

        for i in range(10000):
            # action = model.predict(obs)
            action, _states = model.predict(obs,deterministic=True)
            obs, rewards, dones, info = env.step(action)

            # print(info)
            # if dones:
            #     obs = env.reset()

if __name__ == "__main__":
    main()