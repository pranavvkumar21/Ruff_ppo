#!/usr/bin/env python3

import gym
import numpy as np

env = gym.make("MountainCar-v0")
episodes = 3
env.reset()
for i in range(episodes):
    env.reset()
    done=False
    count = 0
    while not done:
        _,_,done,_ = env.step(np.random.randint(3))
        count+=1
        print(count)
