
import argparse
import time
p = argparse.ArgumentParser()
# define command line arguments
p.add_argument("--mode", choices=["train", "eval"], default="eval", help="Mode: train or eval")
p.add_argument("--load", action="store_true", help="Load model if flag is set")
args = p.parse_args()

print(f"Running in {args.mode} mode. Load flag is set to {args.load}")
time.sleep(2)
import os
from pathlib import Path
import yaml
from tabulate import tabulate
import pyfiglet

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
from callbacks import TensorboardCallback
from callbacks import create_checkpoint_callback, get_latest_model_path
import yaml
from pathlib import Path
import os

from natsort import natsorted

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config" / "ruff_config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = ROOT / "models"



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
    cfg.scene.env_spacing = config["scene"]["env_spacing"]
    cfg.seed = config["env_config"]["seed"]
    return cfg

def config_train(model):
    model.learning_rate = config["train"]["learning_rate"]
    model.batch_size = config["train"]["batch_size"]
    model.n_epochs = config["train"]["n_epochs"]
    model.gamma = config["train"]["gamma"]
    model.ent_coef = config["train"]["ent_coef"]
    model.clip_range = lambda _: config["train"]["clip_range"] 
    model.target_kl = config["train"]["target_kl"]
    model.n_steps = config["train"]["n_steps"]
    return model

def print_model_info(model):
    data = [
        ["Algorithm", type(model).__name__],
        ["Policy", model.policy.__class__.__name__],
        ["Num timesteps", model.num_timesteps],
        ["Learning rate", model.learning_rate],
        ["Gamma", model.gamma],
        ["Clip range", getattr(model, "clip_range", "N/A")],
        ["Ent coef", getattr(model, "ent_coef", "N/A")],
        ["VF coef", getattr(model, "vf_coef", "N/A")],
        ["Batch size", getattr(model, "batch_size", "N/A")],
        ["N steps", getattr(model, "n_steps", "N/A")],
    ]
    print(tabulate(data, headers=["Parameter", "Value"], tablefmt="fancy_grid"))

def main():
    register_envs()
    cfg = RuffEnvCfg()
    cfg = config_env(cfg)
    env = gym.make("Ruff-v0", cfg=cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=False)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # save_model_path,_ = get_latest_model_path(model_path, prefix)
    if not args.load :
        model = PPO(
            "MlpPolicy", 
            env, 
            n_steps=config["train"]["n_steps"],
            batch_size=config["train"]["batch_size"],
            learning_rate=config["train"]["learning_rate"],
            clip_range=config["train"]["clip_range"],
            gamma=config["train"]["gamma"],
            ent_coef=config["train"]["ent_coef"],
            n_epochs=config["train"]["n_epochs"],
            verbose=1,
            target_kl=config["train"]["target_kl"],
            use_sde=False,
            normalize_advantage=True,
            policy_kwargs=dict(net_arch=[256, 256], log_std_init=0.0, full_std=True),  # Adjust the policy architecture if needed
            tensorboard_log=str(ROOT / "logs/ppo_pybullet_tensorboard/"),
        )
    else:
        latest_model_path = get_latest_model_path(str(MODEL_PATH), "ruff_ppo_model")
        if latest_model_path is None:
            print("No saved model found. run with --load False to train a new model")
            print("Exiting...")
            exit(0)
        else:
            print("Loading model from:", latest_model_path)
            model = PPO.load(latest_model_path, env=env)
            model = config_train(model)
    print_model_info(model)
    time.sleep(2)
    if args.mode == "train":
        print(pyfiglet.figlet_format("---Training Mode---", font="slant"))
        time.sleep(2)
        callbacks = CallbackList([TensorboardCallback(), create_checkpoint_callback(args.load)])
        total_timesteps = config["train"]["num_iterations"] * config["train"]["n_steps"] * config["train"]["num_envs"]
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=not args.load, callback=callbacks)
        save_path = config["save_config"]["save_path"] + config["save_config"]["save_prefix"] + time.strftime("%Y-%m-%d_%H-%M-%S")
        model.save(save_path)
    else:
        print(pyfiglet.figlet_format("---Evaluation Mode---", font="slant"))
        time.sleep(2)
        obs = env.reset()
        print(obs.shape)

        for i in range(1000):
            # action = model.predict(obs)
            action, _states = model.predict(obs,deterministic=True)
            obs, rewards, dones, info = env.step(action)
            # if info[0]["log"]!=None:
            print(info[0])
            # print(info)
            # if dones:
            #     obs = env.reset()

if __name__ == "__main__":
    main()