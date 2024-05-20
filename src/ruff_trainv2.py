#!/usr/bin/env python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from collections import UserDict
import pybullet_data
from ruff import *
import random, math

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

NUM_EPISODES = 100000
kc = 2e-10
kd = 0.999994
LOAD = False

class Ruff_env(gym.Env):
    def __init__(self,render_type="gui",command=[1.0,1e-9,1e-9]):
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
        self.timestep = 0
        return self.state, {}

    def step(self, action,kc=0):

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
        else:
            done = False
        if self.timestep>2000:
            truncated = True
        else:
            truncated = False
        self.new_state = self.ru.get_state().flatten()
        reward,rewards = self.ru.get_reward(kc)
        return self.new_state, reward, done, truncated, {"rewards":rewards}
    
    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def close(self):
        # Close the PyBullet simulation
        p.disconnect(self.physics_client)



# Train PPO Model
env = Ruff_env("direct")
print("env init done")
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    policy_kwargs=dict(net_arch=[256, 256]),  # Adjust the policy architecture if needed
    tensorboard_log="./ppo_pybullet_tensorboard/"
)
try:
    if LOAD:
        model = PPO.load("ppo_custom_pybullet_env",env=env)
        print("loaded model")
    else:
        print("load set to off")
except:
    print("could not load model")
model.learn(total_timesteps=10000000)
model.save("ppo_custom_pybullet_env")



obs, info = env.reset()
count = 0
for i in range(10000000):
    action, _states = model.predict(obs)
    obs, rewards, done, trunc, info = env.step(action,kc)
    kc = kc**kd
    if done or trunc:
        env.reset()
        count+=1
        print("episode "+str(count)+ " has been completed")
    # Check if the environment follows the Gym API
    check_env(env)

