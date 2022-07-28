#!/usr/bin/env python3
import pybullet as p
import time
import pybullet_data
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense
import tensorflow_probability as tfp
from model import *
import math
from os.path import exists
import os
import csv
from datetime import datetime
import random
tfd = tfp.distributions

NUM_EPISODES = 50_000
STEPS_PER_EPISODE = 1_000
timestep =1.0/240.0
num_inputs = (60,)
gamma= 0.992
lmbda = 0.95
critic_discount = 0.5
clip_range = 0.2
entropy = 0.0025
curDT = datetime.now()
filename = "ruff_logfile"
reward_log = 'reward_logfile.csv'
discounted_sum = 0
kc = 0
kd = 1


def setup_world():
    physicsClient = p.connect(p.DIRECT)##or p.DIRECT for no    n-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0,0,0.4]
    startOrientation = p.getQuaternionFromEuler([0,0,math.pi/2])
    boxId = p.loadURDF("../urdf/ruff.urdf",startPos, startOrientation)
    p.resetBasePositionAndOrientation(boxId, startPos,  startOrientation)
    return boxId

def close_world():
    p.disconnect()

def train(ru,actor,critic,discounted_sum,episode,step):
    with tf.GradientTape(persistent = True) as tape:
        state_curr = ru.get_state()
        mu,sigma = actor(state_curr)
        critic_value = critic(state_curr)
        dist = tfd.Normal(loc=mu, scale=sigma)
        actions = dist.sample(1)
        actions = actions.numpy().tolist()[0][0]

        if np.isnan(actions).any():
            print("exiting")
            exit()

        pos_inc = actions[0:12]
        pos_inc = [i*math.pi/90 for i in pos_inc]
        freq = actions[12:]

        ru.set_frequency(freq)
        ru.phase_modulator()
        ru.update_policy(actions)
        log_probs = dist.log_prob(actions)
        #print(log_probs.shape)
        ru.update_target_pos(pos_inc)
        ru.move()
        p.stepSimulation()
        #time.sleep(1./240.)
        new_state = ru.get_state()

        reward = ru.get_reward(episode,step)

    return discounted_sum

def check_log(filename):
    files = os.listdir("./logs/")
    filecode = len(files)+1
    filename = '../logs/ '+filename + "_"+str(filecode)+ ".csv"
    return filename

def run_episode():
    state = []
    rewards = []
    actions = []
    logprobs = []

    if exists("./test.bullet"):
        p.restoreState(fileName="test.bullet")
    kc = kc**kd
    ru = ruff(id,kc)
    for i in range(STEPS_PER_EPISODE):
        if j==0 and i==1:
            p.saveBullet("test.bullet")

        episode_reward += ru.reward
        counter +=1
        if ru.is_end() :
            break

if __name__=="__main__":
    filename =check_log(filename)
    id = setup_world()
    ru = ruff(id,kc)
    is_nan = 0
    actor = actor_Model(num_inputs, 16)
    critic = critic_Model(num_inputs, 1)
    try:
        actor.load_weights("actor.h5")
        critic.load_weights("critic.h5")

    except:
        pass



    for j in range(NUM_EPISODES ):

        run_episode(actor,critic)
        
        data = [[j, ru.reward]]
        print("episode: "+ str(j)+"  step: "+ str(i)+"  reward: "+ "{:.2f}".format(episode_reward)+" discounted sum: "+ "{:.2f}".format(discounted_sum))
        act_json = actor.to_json()
        cri_json = critic.to_json()
        with open("../model/actor.json","w") as json_file:
            json_file.write(act_json)
        with open("../model/critic.json","w") as json_file:
            json_file.write(cri_json)
        actor.save_weights("../model/actor.h5")
        critic.save_weights("../model/critic.h5")
        print("model saved")
        #exit()
        with open(filename, 'a', newline="") as file:
            csvwriter = csv.writer(file) # 2. create a csvwriter object
            csvwriter.writerows(data) # 5. write the rest of the data

    close_world()
