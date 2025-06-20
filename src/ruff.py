import pybullet as p
import time
import pybullet_data
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense
import tensorflow_probability as tfp
from model_ppo import *
import math
from os.path import exists
import os
import csv
from datetime import datetime
import random
import logging
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, message="A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel(logging.ERROR)
NUM_EPISODES = 100_000
STEPS_PER_EPISODE = 3_000
timestep = 1.0/2000.0
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
epsilon_min = 0.01

import os
urdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"urdf")
#print(os.listdir(os.path.dirname(os.path.abspath(__file__))))
tfd = tfp.distributions

urdf_constraint = [0,math.pi/6,math.pi/6]*4
def setup_world(n_actors,client_mode):
    physicsClient = p.connect(client_mode)##or p.DIRECT for no    n-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)
    planeId = p.loadURDF("plane.urdf")
    ids = []
    max_row = 3
    c=0
    d = 0
    for i in range(n_actors):
        print(i)
        if d>=max_row:
            c-=1
        startPos = [0,d*3,0.4]
        d+=1
        startOrientation = p.getQuaternionFromEuler([0,0,math.pi/2])
        boxId = p.loadURDF(urdf_path+"/ruff.urdf",startPos, startOrientation)
        p.resetBasePositionAndOrientation(boxId, startPos,  startOrientation)
        ids.append(boxId)
    return ids


def reset_world(filepath):
    if exists(filepath):
        p.restoreState(fileName=filepath)

def save_world(filepath):
    p.saveBullet(filepath)

def close_world():
    p.disconnect()


class ruff:
    def __init__(self, id,command):
        self.id = id
        #self.command = [1e-10,1.5,1e-10] #3 commands for motion
        self.command = command
        #self.command[np.random.randint(2)+1] = np.random.rand()
        #print(self.command)
        self.num_joints = p.getNumJoints(self.id)
        self.joint_names = {}
        for i in range(self.num_joints):
            self.joint_names[str(p.getJointInfo(self.id,i)[1])[2:-1]] = i

        self.n_joints = [i for i in range(self.num_joints) ]
        self.getjointinfo()    #12 joint position and 12 joint velocity
        self.getvelocity()     #6 base velocity velocity
        self.get_base_info()   # 3 base orientation
        self.get_contact()
        self.get_height()
        self.get_link_vel()
        self.policy = [0]*16
        self.prev_policy = self.policy.copy()
        self.target_pos = self.joint_position.copy()
        self.pos_error = [(i-j)/(2*math.pi) for i,j in zip(self.target_pos,self.joint_position)]  #12 positional error
        self.rg_freq = [0,0,0,0]         #4 rg frequency 1 for each limb
        self.rg_phase = [0,0,0,0]        #8 rg phase 2 for each limb
        self.binary_phase = [0,0,0,0]
        self.reward = 0
        self.actor_loss = 0
        self.critic_loss = 0
        self.reward_history = []

    def getvelocity(self):
        self.base_linear_velocity,self.base_angular_velocity = p.getBaseVelocity(self.id)
        self.base_linear_velocity = [i for i in self.base_linear_velocity]
        self.base_angular_velocity = [i for i in self.base_angular_velocity]

    def getpos_error(self):
        self.pos_error = [(i-j) for i,j in zip(self.target_pos,self.joint_position)]

    def getjointinfo(self):

        self.joint_position = []
        self.joint_velocity = []
        self.joint_force = []
        self.joint_torque = []
        n_joints = [i for i in range(self.num_joints) ]
        self.joint_state = p.getJointStates(self.id,n_joints)
        for i in self.joint_state:
            self.joint_position.append((i[0]))
            self.joint_velocity.append(i[1])
            self.joint_force.append(i[2])
            self.joint_torque.append(i[3]/50.0)

    def get_base_info(self):
        self.base_orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.id)[1])
        self.base_position = p.getBasePositionAndOrientation(self.id)[0]

    def get_link_vel(self):
        self.foot_zvel = [p.getLinkState(self.id,2,1)[6][2],p.getLinkState(self.id,5,1)[6][2],p.getLinkState(self.id,8,1)[6][2],p.getLinkState(self.id,11,1)[6][2]]
        self.foot_xvel = [p.getLinkState(self.id,2,1)[6][0],p.getLinkState(self.id,5,1)[6][0],p.getLinkState(self.id,8,1)[6][0],p.getLinkState(self.id,11,1)[6][0]]
        self.foot_yvel = [p.getLinkState(self.id,2,1)[6][1],p.getLinkState(self.id,5,1)[6][1],p.getLinkState(self.id,8,1)[6][1],p.getLinkState(self.id,11,1)[6][1]]

    def get_state(self):
        self.getvelocity()
        self.getjointinfo()
        self.get_base_info()
        self.get_contact()
        self.get_height()
        self.getpos_error()
        self.get_link_vel()
        freq_state = []
        for i in range(4):
            freq_state = freq_state + [math.sin(self.rg_phase[i]),math.cos(self.rg_phase[i])]
        state = list(self.command)
        state = state + list([i/10 for i in self.base_linear_velocity])+list([i/10 for i in self.base_angular_velocity])
        state = state + list(i/(2*math.pi) for i in self.joint_position)+list(i/10 for i in self.joint_velocity)

        state = state + list([i/(2*math.pi) for i in self.pos_error])

        state = state + list(i/(2*math.pi) for i in self.rg_freq)

        state = state + freq_state

        state = state + list([i/(2*math.pi) for i in self.base_orientation])

        state = np.array(state,dtype="float32")
        state = np.reshape(state, (1,-1))
        return state

    def phase_modulator(self):
        for i in range(len(self.rg_phase)):
            self.rg_phase[i] = (self.rg_phase[i]+2*math.pi*self.rg_freq[i]*timestep)%(2*math.pi)
            self.binary_phase[i] = True if self.rg_phase[i]<=math.pi else False #True if stance

    def update_target_pos(self,pos_inc):
        for i in range(len(self.target_pos)):
            self.target_pos[i]= self.joint_position[i] + pos_inc[i]

        return self.target_pos

    def get_contact(self):
        self.is_contact = [p.getContactPoints(self.id,0,linkIndexA=2)!=(),p.getContactPoints(self.id,0,linkIndexA=5)!=(),
        p.getContactPoints(self.id,0,linkIndexA=8)!=(),p.getContactPoints(self.id,0,linkIndexA=11)!=()]
    def get_height(self):
        self.foot_height = [p.getClosestPoints(self.id,0,50000.0,linkIndexA=2)[0][8],p.getClosestPoints(self.id,0,50000.0,linkIndexA=5)[0][8],p.getClosestPoints(self.id,0,50000.0,linkIndexA=8)[0][8],p.getClosestPoints(self.id,0,50000.0,linkIndexA=11)[0][8]]
    def move(self):
        self.getpos_error()
        Kp = 78
        Kd = 2
        self.joint_torque = []
        self.joint_velocity_error = []
        for i in range(len(self.n_joints)):
            vel_err = -self.joint_velocity[i]
            torque = Kp * self.pos_error[i] + Kd * vel_err
            self.joint_torque.append(torque)
            self.joint_velocity_error.append(vel_err)
            p.setJointMotorControl2(
                self.id, self.n_joints[i],
                controlMode=p.TORQUE_CONTROL,
                force=torque
            )
    def set_frequency(self,freq):
        self.rg_freq = freq

    def update_policy(self,actions):
        self.prev_policy = self.policy.copy()
        self.policy = actions
    def action_select(self,mu,sigma):
        dist = tfd.Normal(loc=mu, scale=sigma)
        actions = dist.sample(1)
        actions = actions.numpy().tolist()[0][0]
        pos_inc = actions[0:12]
        for i in range(len(pos_inc)):
            if pos_inc[i]<-1:
                pos_inc[i]= (-1*math.pi/180)*6
            elif pos_inc[i]>1:
                pos_inc[i]= (1*math.pi/180)*6
            else:
                pos_inc[i]= (pos_inc[i]*math.pi/180)*6

        #pos_inc = [i*math.pi/90 for i in pos_inc]
        freq = np.abs(actions[12:])
        for i in range(len(freq)):
            if freq[i]>1:
                freq[i]=1
        log_probs = dist.log_prob(actions)
        return pos_inc,freq, actions,log_probs

    def get_reward(self,kc=1):
        c1 = c2 = c3 = c4 =  1.2
        epsilon_min = 0.00001
        cx = 1.0/ max(abs(self.command[0]),epsilon_min)
        cy = 1.0/ max(abs(self.command[1]),epsilon_min)
        cw = 1.0/ max(abs(self.command[2]),epsilon_min)

        # transform forward and lateral velocity from base frame 
        yaw = self.base_orientation[2]-math.pi/2
        fwd_world_frame = np.array([np.cos(yaw), np.sin(yaw), 0])
        lat_world_frame = np.array([-np.sin(yaw), np.cos(yaw), 0])
        fwd_velocity = np.dot(self.base_linear_velocity, fwd_world_frame)
        lat_velocity = np.dot(self.base_linear_velocity, lat_world_frame)
        forward_velocity = 3*math.exp(-3 * cx * ((fwd_velocity-self.command[0])**2))
        lateral_velocity = 2*math.exp(-3 * cy * ((lat_velocity-self.command[1])**2))
        angular_velocity = 1.5*math.exp(-1.5 * cw *((self.base_angular_velocity[2]-self.command[2])**2))
        balance = 0.8*(math.exp(-2.5 * ((self.base_linear_velocity[2])**2)/max(abs(self.command[0]),epsilon_min)) + math.exp(-2* ((self.base_angular_velocity[0]**2+ self.base_angular_velocity[1]**2))/max(abs(self.command[0]),epsilon_min)))
        twist = -0.6 *((self.base_orientation[0]**2 + self.base_orientation[1]**2)**0.5) * cx

        if p.getContactPoints(self.id,0,linkIndexA=-1)!=() or p.getContactPoints(self.id,0,linkIndexA=0)!=() or p.getContactPoints(self.id,0,linkIndexA=1)!=() or p.getContactPoints(self.id,0,linkIndexA=3)!=() or p.getContactPoints(self.id,0,linkIndexA=4)!=() or p.getContactPoints(self.id,0,linkIndexA=6)!=() or p.getContactPoints(self.id,0,linkIndexA=7)!=() or p.getContactPoints(self.id,0,linkIndexA=9)!=() or p.getContactPoints(self.id,0,linkIndexA=10)!=():
            collision = -3
        else:
            collision = 0

        foot_slip = 0
        foot_stance = 0
        foot_clear = 0
        foot_zvel1 = 0
        policy_smooth = 0
        joint_constraints = 0
        joint_velocity_error = 0
        joint_torque_error = 0
        frequency_err = 0
        phase_err = 0
        for i in range(16):
            policy_smooth+=(self.policy[i] - self.prev_policy[i])**2
        for i in range(12):
            joint_constraints += self.pos_error[i]**2

        for i in range(4):
            foot_slip += (self.foot_xvel[i]**2 + self.foot_yvel[i]**2) if self.binary_phase[i] else 0
            foot_stance += 1 if self.foot_height[i]<0.01 and self.binary_phase[i] else 0
            foot_clear  += 0.7*c1 if self.foot_height[i]>0.01 and (not self.binary_phase[i]) else 0
            frequency_err += abs(self.rg_freq[i]) if self.binary_phase[i] else 0
            phase_err += 1 if self.is_contact[i] == self.binary_phase[i] else 0
            foot_zvel1 += abs(self.foot_zvel[i])
        policy_smooth = -0.016 * c4 * (policy_smooth**0.5)/max(abs(self.command[0]),epsilon_min)
        foot_zvel1 = (-0.03*foot_zvel1**2)/max(abs(self.command[0]),epsilon_min)
        foot_slip = -0.07*(foot_slip**0.5)/max(abs(self.command[0]),epsilon_min)
        frequency_err = -0.03*frequency_err
        joint_constraints = -0.8*(joint_constraints**0.5)/max(abs(self.command[0]),epsilon_min)
        torque_penalty = -0.0012 * c2 * cx * np.linalg.norm(self.joint_torque)
        velocity_penalty = -0.0008 * c3 * cx * np.linalg.norm(self.joint_velocity_error) ** 2

        basic_reward = forward_velocity + lateral_velocity + angular_velocity 
        freq_reward = (foot_stance + balance + foot_clear + foot_zvel1  + frequency_err + phase_err + foot_slip + policy_smooth+joint_constraints)
        efficiency_reward =  twist
        self.complete_reward = basic_reward + freq_reward + efficiency_reward
        rewards = {"forward_velocity":forward_velocity,
                   "lateral_velocity":lateral_velocity,
                   "angular_velocity":angular_velocity,
                   "Balance":balance,
                   "foot_stance":foot_stance, 
                   "foot_clear":foot_clear, 
                   "foot_zvel1":foot_zvel1, 
                   "frequency_err":frequency_err,
                   "phase_err":phase_err,
                   "joint_constraints":joint_constraints,
                   "foot_slip":foot_slip, 
                   "policy_smooth":policy_smooth,
                   "twist":twist,
                   "complete_reward":self.complete_reward, 
                   "torque_penalty":torque_penalty,
                   "velocity_penalty":velocity_penalty,}
        
        self.reward = kc["forward"]*forward_velocity + \
                        kc["lateral"]*lateral_velocity + \
                        kc["angular"]*angular_velocity + \
                        kc["balance_twist"]*balance + \
                        kc["balance_twist"]*twist + \
                        kc["rhythm"]*(foot_stance) + \
                        kc["rhythm"]*(foot_clear) + \
                        kc["rhythm"]*(foot_zvel1) + \
                        kc["rhythm"]*(frequency_err) + \
                        kc["rhythm"]*(phase_err) + \
                        kc["rhythm"]*(foot_slip) + \
                        kc["efficiency"]*(policy_smooth) + \
                        kc["efficiency"]*(joint_constraints) + \
                        kc["efficiency"]*(torque_penalty) + \
                        kc["efficiency"]*(velocity_penalty)

        #self.reward = forward_velocity + kc*(lateral_velocity + angular_velocity)
        infos = {"rewards":rewards}
        return self.reward,infos

    def is_end(self):
        if p.getContactPoints(self.id,0,linkIndexA=-1)!=(): #or p.getContactPoints(self.id,0,linkIndexA=0)!=() or p.getContactPoints(self.id,0,linkIndexA=1)!=() or p.getContactPoints(self.id,0,linkIndexA=3)!=() or p.getContactPoints(self.id,0,linkIndexA=4)!=() or p.getContactPoints(self.id,0,linkIndexA=6)!=() or p.getContactPoints(self.id,0,linkIndexA=7)!=() or p.getContactPoints(self.id,0,linkIndexA=9)!=() or p.getContactPoints(self.id,0,linkIndexA=10)!=() :
            return 1
        else:
            return 0
