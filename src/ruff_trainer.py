#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense
from model import *
import tensorflow_probability as tfp
import time
import rospy
from insectbot.msg import state,key
from rospy.numpy_msg import numpy_msg
import numpy as np

gamma = 0.99
num_episodes = 200
num_inputs = (60,1)      # direction + angles + sensors
num_actions = 8       #no of motors
layer1 = 64
layer2 = 64
layer3 = 64
num_steps = 10

rospy.init_node('ruff_trainer', anonymous=False)

class ruff:
    def __init__(self):
        self.joint_position = None
        self.joint_velocity = None
        self.orientation = None
        self.base_linear_velocity = [0,0,0]
        self.base_angular_velocity = None
        self.key = None
        self.nsecs = None
        self.command = None
    def joint_state_callback(self,msg):
        self.joint_position = msg.position
        self.joint_velocity = msg.velocity
        self.joint_name = msg.name
        print(self.joint_position)
        #print(1)
    def imu_callback(self,msg):
        self.orientation = [msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w]
        self.base_angular_velocity = [msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z]
        print(self.orientation)
    def command_callback(self,msg):
        pass

    def __call__(self):

        #fullstate = np.concatenate((self.state,self.key))
        return self.key
    def get_state():
        rospy.Subscriber('/ruff/joint_states', JointState , states.joint_state_callback)
        rospy.Subscriber('/ruff/imu', Imu, states.imu_callback)


def get_reward():
    return np.random.randint(5,10)

state_obj = State()
def initialize(states):
    rospy.Subscriber('floats', numpy_msg(state), states.state_callback)
    rospy.Subscriber('key', numpy_msg(key), states.key_callback)
    return states()
def act(action,states):
    rospy.Subscriber('floats', numpy_msg(state), states.state_callback)
    rospy.Subscriber('key', numpy_msg(key), states.key_callback)
    return states()
