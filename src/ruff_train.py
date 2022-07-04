#!/usr/bin/env python3
import pybullet as p
import time
import pybullet_data
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense
from model import *

NUM_EPISODES = 200
STEPS_PER_EPISODE = 100


def setup_world():
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    startPos = [0,0,1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    boxId = p.loadURDF("../urdf/ruff.urdf",startPos, startOrientation)
    #set the center of mass frame (loadURDF sets base link frame)
    p.resetBasePositionAndOrientation(boxId, startPos,  startOrientation)
    return boxId
class ruff:
    def __init__(self, id):
        self.id = id
        self.command = [0, 0, 0] #3 commands for motion
        self.num_joints = p.getNumJoints(self.id)
        self.n_joints = [i for i in range(self.num_joints) ]
        self.getjointinfo()    #12 joint position and 12 joint velocity
        self.getvelocity()     #6 base velocity velocity
        self.get_base_info()   # 3 base orientation
        self.target_pos = self.joint_position
        print(self.target_pos)
        self.pos_error = [i-j for i,j in zip(self.target_pos,self.joint_position)]  #12 positional error
        self.rg_freq = None         #4 rg frequency 1 for each limb
        self.rg_phase = None        #8 rg phase 2 for each limb
    def getvelocity(self):
        self.base_linear_velocity,self.base_angular_velocity = p.getBaseVelocity(self.id)
    def getjointinfo(self):
        self.joint_position = []
        self.joint_velocity = []
        self.joint_force = []
        self.joint_torque = []
        n_joints = [i for i in range(self.num_joints) ]
        self.joint_state = p.getJointStates(id,n_joints)
        for i in self.joint_state:
            self.joint_position.append(i[0])
            self.joint_velocity.append(i[1])
            self.joint_force.append(i[2])
            self.joint_torque.append(i[3])
    def get_base_info(self):
        self.base_orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.id)[1])
    def get_state(self):
        self.getvelocity()
        self.getjointinfo()
        self.get_base_info()
    def update_target_pos(self,pos_inc):
        for i in target_pos:
            self.target_pos[i]+=pos_inc[i]
    def update
    def move(self):
        max_force = [50]*12
        p.setJointMotorControlArray(self.id,self.n_joints,controlMode=p.POSITION_CONTROL,
                                    targetPositions = self.target_pos,forces=max_force)
    def get_reward():


def close_world():
    p.disconnect()

if __name__=="__main__":
    id = setup_world()

    ru = ruff(id)

    actor = actor_Model(num_inputs, 16, layer1, layer2, layer3)
    critic = critic_Model(num_inputs, 1, layer1, layer2, layer3 )
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    count = 0

    for i in range(STEPS_PER_EPISODE):
        p.stepSimulation()
        time.sleep(1./240.)
        with tf.GradientTape(persistent = True) as tape:
            state_curr = ru.get_state()
            mu,sigma = actor(state_curr)
            critic_value = critic(state_curr)

            dist = tfd.Normal(loc=mu, scale=sigma)
            actions = dist.sample(1)
            log_probs = dist.log_prob(actions)
            #print(log_probs)

            new_state = ru.get_state(actions)

            critic_value_ = critic(new_state)

            reward = ru.get_reward()

            delta = reward + gamma*critic_value_ - critic_value
            actor_loss = -log_probs * delta
            critic_loss = delta**2

            #print("actor loss ")


        actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, critic.trainable_variables)

        optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
        optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    count = count+1
    print("time taken for 10 steps in episode "+str(count)+":  ")
    print(time.time()-a)
    close_world()
