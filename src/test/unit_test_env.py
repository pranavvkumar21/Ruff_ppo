#!/usr/bin/env python3
import numpy as np
import sys, os
sys.path.append("..")
from model_ppo import *
from ruff_trainv2 import *
import random
import time
os.chdir("..")


def get_keyboard_input(current_joint_index):
    keys = p.getKeyboardEvents()
    joint_delta = 0.0

    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        joint_delta += 0.001745
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        joint_delta -= 0.001745

    for i in range(10):
        if ord(str(i)) in keys and keys[ord(str(i))] & p.KEY_WAS_TRIGGERED:
            current_joint_index = i
            print(str(i)+" was pressed")
    
    return joint_delta, current_joint_index

    

def test_reset(env):
    for i in range(3):

        for i in range(1000):
            action = [random.uniform(-0.0174533, 0.0174533) for _ in range(16)]
            print(i)
            #print(action)
            state,reward ,d,t,_ = env.step(action)
            #print((state.shape))
        env.reset()
        time.sleep(1)

def test_control(env):
    action = [0.0]*16
    current_joint_index = 0
    count = 0
    try:
        while True:
            joint_delta, current_joint_index = get_keyboard_input(current_joint_index)
            print(joint_delta)
            action[current_joint_index] += joint_delta
            state,reward ,d,t,_ = env.step(action)
            action = [0.0]*16
            count+=1
            print(count)
            if d or t:
                exit()
    except KeyboardInterrupt:
        print("Simulation stopped by user.")

    env.reset()

def test_reward(env):
    current_joint_index = 0
    try:
        while True:
            joint_delta, current_joint_index = get_keyboard_input(current_joint_index)
            if joint_delta != 0:
                p.applyExternalForce(objectUniqueId=1,linkIndex=0,forceObj=[0,50,0],posObj=[0, 0, 0],flags=p.WORLD_FRAME)
            else:
                p.applyExternalForce(objectUniqueId=1,linkIndex=0,forceObj=[0,0,0],posObj=[0, 0, 0],flags=p.WORLD_FRAME)
            action = [0.0]*16
            state,reward ,d,t,re = env.step(action,0)
            rewards = re["rewards"]
            print(rewards[0:3])
            
    except KeyboardInterrupt:
        print("Simulation stopped by user.")



if __name__=="__main__":
    env = Ruff_env()
    test_control(env)