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



TOTAL_TIMESTEPS = 98_304_000
ELAPSED_TIMESTEPS = 0
kc = 0.0001
kd = 0.999997
LOAD = True
testing_mode = False
checkpoint = 50_000

now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
model_path = '../model/v2_models/'
save_model_path = model_path+"model_"+str(formatted_time)
save_config_path = "../config/config.txt"
log_dir = "../logs"

# Remove all files in the directory


prefix = 'ruu_ppo_model'
if testing_mode:
    NUM_ENV = 1
    render_type = "gui"
    LOAD = False
else:
    NUM_ENV = 32
    render_type = "DIRECT"



class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []  # Store total rewards for episodes
        self.episode_sub_rewards = {}  # Store sub-rewards for episodes

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
            self.extra_variable = {"elapsed_timesteps":self.num_timesteps, "n_calls":self.n_calls}
            with open(save_config_path, 'w') as f:
                json.dump(self.extra_variable, f, indent=4) # Save the extra variable with numpy

            if self.verbose > 0:
                print(f"Saving checkpoint data at {self.num_timesteps} timesteps to {save_config_path}")

        return result

def load_checkpoint_data(json_path: str):
    try:

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract the values of interest (elapsed_timesteps and kc)
        elapsed_timesteps = data.get('elapsed_timesteps', None)
        n_calls = data.get('n_calls', None)
        for i in range(n_calls):
            global kc,kd
            kc = kc**kd
        #kc = data.get('kc', None)

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
        #print("-"*20)
        self.og_state = p.saveState()
        self.state = self.ru.get_state()
        self.timestep = 0
        self.curriculum = curriculum
        if self.curriculum:
            self.kc = kc
            self.kd = kd
        else:
            self.kc = 0


    def set_curriculum(self):
        self.kc = self.kc ** self.kd

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        p.restoreState(stateId=self.og_state)
        
        #commands = [[random.uniform(0.3,2),1e-9,1e-9],[random.uniform(0.3,2),random.uniform(-self.kc,self.kc),1e-9],[random.uniform(0.3,2),1e-9,random.uniform(-self.kc,self.kc)]]
        commands = [[random.uniform(0.3,2),1e-9,1e-9]]
        self.command = random.choice(commands)
        self.ru = ruff(self.Id,self.command)
        self.state = self.ru.get_state().flatten()
        print("resetting.. doggo no: "+str(self.env_rank))
        print("new command: "+ str(self.command)+"\n\n")
        print("kc updated to: "+str(self.kc))
        self.timestep = 0
        return self.state, {}

    def step(self, action):
        if self.curriculum:
            self.set_curriculum()
            #print(self.kc)
        freq = np.abs(action[12:]).tolist()
        pos_update = action[0:12]
        pos_update = [math.radians(deg) for deg in pos_update]
        self.ru.set_frequency(freq)
        self.ru.phase_modulator()
        self.ru.update_policy(action)
        tp = self.ru.update_target_pos(pos_update)
        #print(tp)
        self.ru.move()
        for _ in range(self.sim_steps_per_control_step):
            p.stepSimulation()
        #print("moved")
        self.timestep+=1
        if self.ru.is_end():
            done = True
            print("doggo no:"+str(self.env_rank)+" fell")
        else:
            done = False
        if self.timestep>2000:
            truncated = True
            print("doggo no:"+str(self.env_rank)+"completed the episode")
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
        checkpoint_callback = CheckpointCallback(save_freq=checkpoint, save_path=save_model_path,
                                         name_prefix=prefix)
        custom_checkpoint_callback = CustomCheckpointCallback(save_freq=checkpoint,save_path=save_model_path,name_prefix=prefix )
        callback = CallbackList([custom_checkpoint_callback, TensorboardCallback()])
        print("created new save path")

        if LOAD:
            ELAPSED_TIMESTEPS,kc = load_checkpoint_data(save_config_path)
        env = SubprocVecEnv([make_env(i,render_type=render_type) for i in range(NUM_ENV)])
    else:
        env = Ruff_env(render_type=render_type)
    print("env init done")
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=1024,
        learning_rate=3.5e-4,
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

