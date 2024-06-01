#!/usr/bin/env python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from collections import UserDict
import pybullet_data
from ruff import *
import random, math
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel(logging.ERROR)

NUM_EPISODES = 100000
kc = 2e-10
#kc = 0.99999
kd = 0.999994
LOAD = True
testing_mode = False
checkpoint = 50000

now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
model_path = '../model/v2_models/'
save_model_path = model_path+"model_"+str(formatted_time)

prefix = 'ruu_ppo_model'
if testing_mode:
    NUM_ENV = 1
    render_type = "gui"
    LOAD = True
else:
    NUM_ENV = 24
    render_type = "DIRECT"

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Collect rewards from all environments
        rewards = np.array(self.locals['rewards'])
        self.episode_rewards.append(rewards)
        
        if self.locals['dones'].any():
            mean_reward = np.mean([np.sum(rew) for rew in self.episode_rewards])
            self.logger.record('rollout/ep_rew_mean', mean_reward)
            self.episode_rewards = []  # Reset for next episode
        
        return True




def get_latest_model_path(folder_path, prefix):
    folders = os.listdir(folder_path)
    folders.sort(reverse=True)
    print(folders)
    latest_folder = folders[1]
    print(latest_folder)
    folder_path = os.path.join(folder_path,latest_folder)
    print("-"*30)
    print(latest_folder)

    files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith('.zip')]
    if not files:
        raise FileNotFoundError(f"No checkpoint files found with prefix '{prefix}' in '{folder_path}'")
    
    # Sort files by modification time
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    return os.path.join(folder_path, files[0])



class Ruff_env(gym.Env):
    def __init__(self,render_type="gui",command=[0.3,1e-9,1e-9],curriculum = False, kc=0,kd=1):
        super(Ruff_env, self).__init__()
        # Define the action and observation space
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
        print(self.ru.id)
        print("-"*20)
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
        command = [random.uniform(0.3,2),1e-9,1e-9]
        self.ru = ruff(self.Id,self.command)
        self.state = self.ru.get_state().flatten()
        print("resetting..")
        print("new command: "+ str(command))
        print("kc: "+str(self.kc))
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
            print("ruff fell")
        else:
            done = False
        if self.timestep>2000:
            truncated = True
            print("episode end")
        else:
            truncated = False
        self.new_state = self.ru.get_state().flatten()
        reward,rewards = self.ru.get_reward(self.kc)
        return self.new_state, reward, done, truncated, {"rewards":rewards}
    
    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def close(self):
        # Close the PyBullet simulation
        p.disconnect(self.physics_client)



# Train PPO Model

def make_env(rank, seed=0, render_type="direct"):
    def _init():
        print(render_type)
        env = Ruff_env(render_type,curriculum=True,kc = kc, kd=kd)
        #env.seed(seed + rank)
        return env
    return _init


if __name__ == "__main__":

    if not testing_mode:
        os.mkdir(save_model_path)
        checkpoint_callback = CheckpointCallback(save_freq=checkpoint, save_path=save_model_path,
                                         name_prefix=prefix)

        callback = CallbackList([checkpoint_callback, TensorboardCallback()])
        print("created new save path")
        env = SubprocVecEnv([make_env(i,render_type=render_type) for i in range(NUM_ENV)])
    else:
        env = Ruff_env(render_type=render_type)
    print("env init done")
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=1024,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),  # Adjust the policy architecture if needed
        tensorboard_log="./ppo_pybullet_tensorboard/"
    )
    try:
        if LOAD:
            latest_model_path = get_latest_model_path(model_path, prefix)

            model = PPO.load(latest_model_path,env=env)
            print("loaded model: "+latest_model_path)
            print("-"*120)
        else:
            print("load set to off")
    except Exception as e:
        print("could not load model")
        print("error: "+str(e))
    
    if not testing_mode:
        model.learn(total_timesteps=98304000, callback=callback)
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

