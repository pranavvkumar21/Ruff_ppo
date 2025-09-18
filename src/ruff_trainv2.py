#!/usr/bin/env python3
import logging
import os
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel(logging.ERROR)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from collections import UserDict
import pybullet_data
from ruff import *
import random, math
from datetime import datetime
import json
import glob
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
import torch
import psutil
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)



TOTAL_TIMESTEPS = 98_304_000
ELAPSED_TIMESTEPS = 0
kc = 1
kd = 0.996
kc = {
    "forward": 1,
    "lateral": 1,
    "angular": 1,
    "balance_twist": 0.1,
    "rhythm": 0.01,
    "efficiency": 0.01,
}
kd = {
    "forward": 1,
    "lateral": 0.999_994,
    "angular": 0.999_994,
    "balance_twist": 0.999_994,
    "rhythm": 0.999_994,
    "efficiency": 0.999_994
}
LOAD = False
testing_mode = False
checkpoint = 50_000

now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
model_path = '../models/'
save_model_path = model_path+"model_"+str(formatted_time)
save_config_path = "../config/config.txt"
log_dir = "../logs"
video_dir = "../videos"
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# Remove all files in the directory


prefix = 'ruu_ppo_model'
if testing_mode:
    NUM_ENV = 1
    render_type = "gui"
    LOAD = False
else:
    NUM_ENV = 32
    render_type = "DIRECT"

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.env_steps = {}
        self.step_counts = [] 

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, done in enumerate(dones):
            self.env_steps[i] = self.env_steps.get(i, 0) + 1
            if done:
                self.step_counts.append(self.env_steps[i])
                self.env_steps[i] = 0
        return True

    def _on_rollout_end(self) -> None:
        # kc update
        # kc = self.training_env.get_attr("kc")[0]
        # kd = self.training_env.get_attr("kd")[0]
        
        # if isinstance(kc, dict):
        #     kc_new = {key: min(1.0, value ** kd[key]) for key, value in kc.items()}
        # else:
        #     kc_new = min(1.0, kc ** kd)
        
        # self.training_env.set_attr("kc", kc_new)
        # print(f"[Curriculum] kc updated to: {kc_new}")
        #print average kc values across all envs
        avg_kc = {key: np.mean([env_kc[key] for env_kc in self.training_env.get_attr("kc")]) for key in kc.keys()}
        print(f"average kc update to {avg_kc}")
        # avg episode length
        if self.step_counts:
            avg_steps = sum(self.step_counts) / len(self.step_counts)
            print(f"[Rollout] Avg episode length: {avg_steps:.1f} steps")
            self.step_counts.clear()


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []  # Store total rewards for episodes
        self.episode_sub_rewards = {}  # Store sub-rewards for episodes
        self.rollout_idx=0

    def _on_rollout_end(self):
        self.rollout_idx += 1
        self.logger.set_step(self.rollout_idx)

    def _on_step(self) -> bool:
        # Collect rewards from all environments
        rewards = np.array(self.locals['rewards'])
        self.episode_rewards.append(rewards)

        # Assuming `info` dictionary contains sub-rewards as 'rewards'
        infos = self.locals['infos']
        
        for info in infos:
            if 'rewards' in info:
                for key, value in info['rewards'].items():
                    if key not in self.episode_sub_rewards:
                        self.episode_sub_rewards[key] = []
                    self.episode_sub_rewards[key].append(value)

        # Check if the episode ended
        if self.locals['dones'].any():
            # Log the mean reward for the episode
            mean_reward = np.mean([np.sum(rew) for rew in self.episode_rewards])
            self.logger.record('rollout/ep_rew_mean', mean_reward)
            # Log sub-rewards (mean of each sub-reward over the episode)
            for key, sub_reward_list in self.episode_sub_rewards.items():
                mean_sub_reward = np.mean(sub_reward_list)
                #print(f"len of infos: {mean_sub_reward}")
                self.logger.record(f'rollout/sub_reward_{key}_mean', mean_sub_reward)

            # Reset episode rewards for next episode
            self.episode_rewards = []
            self.episode_sub_rewards = {}

        return True
    
class CustomCheckpointCallback(CheckpointCallback):
    """
    Custom checkpoint callback that saves an extra variable.
    """
    def __init__(self, save_freq, save_path, name_prefix, verbose=1):
        super(CustomCheckpointCallback, self).__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, verbose=verbose)
        self.extra_variable = {"elapsed_timesteps":0, "n_calls":0}

    def _on_step(self) -> bool:
        # Call parent class's _on_step method to handle checkpointing logic
        result = super(CustomCheckpointCallback, self)._on_step()

        # Additional logic to save extra variable along with the model
        if self.n_calls % self.save_freq == 0:
            # Save extra variable to a file (you can change this based on your needs)
            self.extra_variable = {
                "elapsed_timesteps": self.num_timesteps,
                "n_calls": self.n_calls,
                "kc": self.training_env.get_attr("kc")[0]  # Save kc
            }
            with open(save_config_path, 'w') as f:
                json.dump(self.extra_variable, f, indent=4) # Save the extra variable with numpy

            if self.verbose > 0:
                print(f"Saving checkpoint data at {self.num_timesteps} timesteps to {save_config_path}")

        return result

class VideoCheckpointCallback(BaseCallback):
    def __init__(self, video_root, run_id,
                 save_freq=50_000, video_len=5_000,
                 name_prefix="ruu_ppo_model", verbose=0):
        super().__init__(verbose)
        self.save_freq, self.video_len = save_freq, video_len
        self.video_dir = os.path.join(video_root, f"videos_{run_id}")
        os.makedirs(self.video_dir, exist_ok=True)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self._record_video()
        return True

    def _record_video(self):
        import pybullet as p, os
        from ruff_trainv2 import Ruff_env         # same file
        path = os.path.join(self.video_dir,
                            f"{self.name_prefix}_{self.num_timesteps}.mp4")

        env = Ruff_env(render_type="DIRECT")
        obs, _ = env.reset(commands=[[0.3, 0, 0]])
        log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, path)

        for _ in range(self.video_len):
            act, _ = self.model.predict(obs, deterministic=True)
            obs, _, d, t, _ = env.step(act)
            if d or t:
                obs, _ = env.reset(commands=[[0.3, 0, 0]])

        p.stopStateLogging(log)
        env.close()

def load_checkpoint_data(json_path: str):
    try:

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract the values of interest (elapsed_timesteps and kc)
        elapsed_timesteps = data.get('elapsed_timesteps', None)
        n_calls = data.get('n_calls', None)
        global kc
        kc = data.get('kc')
        return elapsed_timesteps, kc
    except:
        print("config file not found")
    
        return 0, kc



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

    files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith('.zip')]
    if not files:
        raise FileNotFoundError(f"No checkpoint files found with prefix '{prefix}' in '{folder_path}'")
    
    # Sort files by modification time
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    return folder_path, os.path.join(folder_path, files[0])



class Ruff_env(gym.Env):
    def __init__(self,rank=1, render_type="gui",command=[0.3,1e-9,1e-9],curriculum = False, kc=0,kd=1):
        super(Ruff_env, self).__init__()

                # Define the action and observation space
        self.env_rank = rank
        self.timestep = 1.0/2000.0
        self.sim_steps_per_control_step = 20
        self.action_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float64)
        self.command = command
        # Initialize PyBullet
        if render_type == "gui":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)  # Use p.GUI for graphical version
        p.setTimeStep(self.timestep)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(numSolverIterations=10, enableFileCaching=0)
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, lateralFriction=0.75)
        self.Initial_position = [0,0.0,0.45]
        self.Initial_orientation = p.getQuaternionFromEuler([0,0,math.pi/2])
        self.Id = p.loadURDF("../urdf/ruff.urdf",self.Initial_position, self.Initial_orientation)
        p.resetBasePositionAndOrientation(self.Id, self.Initial_position,  self.Initial_orientation)
        self.ru = ruff(self.Id,self.command)
        print("created doggo with id: " +str(self.env_rank))
        self.og_state = p.saveState()
        self.state = self.ru.get_state()
        self.step_count = 0
        self.curriculum = curriculum
        if self.curriculum:
            self.kc = kc
            self.kd = kd
        else:
            self.kc = {
                "forward": 1,
                "lateral": 1,
                "angular": 1,
                "balance_twist": 1,
                "rhythm": 1,
                "efficiency": 1,
            }


    def set_curriculum(self):
        self.kc = {key: min(1.0, value ** self.kd[key]) for key, value in self.kc.items()}

    def reset(self, seed=None, options=None,commands=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        p.restoreState(stateId=self.og_state)
        
        # commands = [[random.uniform(0.3,2),1e-9,1e-9],[random.uniform(0.3,2),random.uniform(-1,1),1e-9],[random.uniform(0.3,2),1e-9,random.uniform(-1,1)]]
        if commands==None:
        #   commands = [[random.uniform(0.3,2),random.uniform(-1,1),random.uniform(-1,1)]]
        #   commands = [[random.uniform(0.3,2),1e-9,1e-9],[random.uniform(0.3,2),random.uniform(-1,1),1e-9],[random.uniform(0.3,2),1e-9,random.uniform(-1,1)]]
            fwd = round(random.uniform(0.3, 2.0), 1)   # e.g. 0.3, 0.4 … 2.0
            lat = round(random.uniform(-1.0, 1.0), 1)  # e.g. -1.0, -0.9 … 1.0
            yaw = round(random.uniform(-1.0, 1.0), 1)  # same for angular
            # choose among forward only, forward+lateral, forward+yaw
            commands = [
                [fwd, 0.0, 0.0],
                [fwd, lat, 0.0],
                [fwd, 0.0, yaw],
            ]
        self.command = random.choice(commands)
        self.ru = ruff(self.Id,self.command)
        self.state = self.ru.get_state().flatten()
        # print("resetting.. doggo no: "+str(self.env_rank))
        # print("new command: "+ str(self.command)+"\n\n")
        # print("kc updated to: "+str(self.kc))
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        if self.curriculum:
            #update kc
            self.set_curriculum()
            #print(self.kc)
        freq = np.abs(action[12:]).tolist()
        pos_update = action[0:12]
        pos_update = [math.radians(deg*6) for deg in pos_update]
        self.ru.set_frequency(freq)
        self.ru.phase_modulator()
        self.ru.update_policy(action)
        tp = self.ru.update_target_pos(pos_update)
        #print(tp)
        self.ru.move()
        for _ in range(self.sim_steps_per_control_step):
            p.stepSimulation()
        #print("moved")
        self.step_count+=1
        if self.ru.is_end():
            done = True
            # print("doggo no:"+str(self.env_rank)+" fell")
        else:
            done = False
        if self.step_count>2000:
            truncated = True
            # print("doggo no:"+sstr(self.env_rank)+"completed the episode")
        else:
            truncated = False
        self.new_state = self.ru.get_state().flatten()
        reward,infos = self.ru.get_reward(self.kc)
        return self.new_state, reward, done, truncated, infos
    
    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def close(self):
        # Close the PyBullet simulation
        p.disconnect(self.physics_client)



# Train PPO Model

def make_env(rank, seed=0, render_type="direct"):
    def _init():
        #print(render_type)
        #print(rank)
        pr = psutil.Process()
        try:
            pr.cpu_affinity([rank % os.cpu_count()])
        except Exception:
            pass
        env = Ruff_env(rank, render_type,curriculum=True,kc = kc, kd=kd)
        #env.seed(seed + rank)
        return env
    return _init


if __name__ == "__main__":
    print("--"*50)
    print("initializing ruff training")
    print(f"testing mode: {testing_mode}")
    print(f"load set to: {LOAD}")
    print(f"number of env: {NUM_ENV}")
    print("\n\n")

    if not LOAD:
        try:
            shutil.rmtree("../logs/ppo_pybullet_tensorboard/")
            print("removed tensorboard los")
        except:
            print("no tensorboard logs found in directory")
        try:
            os.remove("../config/config.txt")
            print("checkpoint data removed")
        except:
            print("no config file found to remove")

    if not testing_mode:
        if not LOAD:
            os.mkdir(save_model_path)
        else:
            save_model_path,_ = get_latest_model_path(model_path, prefix)

        custom_checkpoint_callback = CustomCheckpointCallback(save_freq=checkpoint,
                                                              save_path=save_model_path,
                                                              name_prefix=prefix )
        video_cb = VideoCheckpointCallback(
            video_root="../videos",
            run_id=formatted_time,
            save_freq=checkpoint,
            name_prefix=prefix)
        callback = CallbackList([custom_checkpoint_callback,
                                 TensorboardCallback(),
                                 CurriculumCallback(),
                                 video_cb])
        print("created new save path")

        if LOAD:
            ELAPSED_TIMESTEPS,kc = load_checkpoint_data(save_config_path)
        env = SubprocVecEnv([make_env(i,render_type=render_type) for i in range(NUM_ENV)],start_method="spawn")
    else:
        env = Ruff_env(render_type=render_type)
    print("env init done")
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=1024,
        batch_size=8192,
        learning_rate=3.5e-4,
        clip_range=0.2,
        gamma=0.992,
        ent_coef=0.0025,
        target_kl=0.015,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),  # Adjust the policy architecture if needed
        tensorboard_log="../logs/ppo_pybullet_tensorboard/"
    )
    try:
        if LOAD:
            _,latest_model_path = get_latest_model_path(model_path, prefix)
            model = PPO.load(latest_model_path,env=env)
            print("loaded model: "+latest_model_path)
            print("-"*120)
            ELAPSED_TIMESTEPS,kc = load_checkpoint_data(save_config_path)
        else:
            print("load set to off")
    except Exception as e:
        print("could not load model")
        print("error: "+str(e))
    
    if not testing_mode:
        model.learn(total_timesteps=(TOTAL_TIMESTEPS - ELAPSED_TIMESTEPS), callback=callback, reset_num_timesteps=(not LOAD), tb_log_name="ruff_ppo")
        model.save("ppo_custom_pybullet_env")
        print("saved file. training completeee")


    else:
        obs, info = env.reset()
        count = 0
        for i in range(10000):

            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, trunc, info = env.step(action)
            kc = kc**kd

            if done or trunc:
                env.reset()
                count+=1
                print("episode "+str(count)+ " has been completed")
        # Check if the environment follows the Gym API
        #check_env(env)

