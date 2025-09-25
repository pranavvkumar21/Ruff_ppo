import pybullet as p
import time
import pybullet_data
import numpy as np

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


NUM_EPISODES = 100_000
STEPS_PER_EPISODE = 3_000
timestep = 1.0/100.0
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



class ruff:
    def __init__(self, id, terrain_id, command,timestep=timestep):
        self.id = id
        self.terrain_id = terrain_id
        self.command = command
        self.num_joints = p.getNumJoints(self.id)
        self.joint_names = {}
        for i in range(self.num_joints):
            ji = p.getJointInfo(self.id, i)
            jname = ji[1].decode()
            self.joint_names[jname] = i

        # links named *_f
        self.foot_tips = [i for i in range(self.num_joints)
                        if p.getJointInfo(self.id, i)[12].decode().endswith("_f")]
        # print("length of foot tips:", len(self.foot_tips))

        # parent link index for each foot tip
        self.foot_links = [p.getJointInfo(self.id, i)[16] for i in self.foot_tips]
        # print("length of foot links:", len(self.foot_links))
        # movable joints
        self.movable_joints = [i for i in range(self.num_joints)
                       if p.getJointInfo(self.id, i)[2] == p.JOINT_REVOLUTE]
        # print("length of movable joints:", len(self.movable_joints))
        self.n_joints = [i for i in range(self.num_joints)]
        self.joint_lower_limits = np.asarray([p.getJointInfo(self.id,i)[8] for i in self.movable_joints], np.float32)
        self.joint_upper_limits = np.asarray([p.getJointInfo(self.id,i)[9] for i in self.movable_joints], np.float32)
        all_links = set(range(-1, self.num_joints))          # includes base -1
        self.is_end_links = list(all_links - set(self.foot_tips) - set(self.foot_links))
        self.getjointinfo()    #12 joint position and 12 joint velocity
        self.getvelocity()     #6 base velocity velocity
        self.get_base_info()   # 3 base orientation
        self.get_contact()
        self.get_link_vel()
        self.get_height()

        self.policy =  np.zeros(16, dtype=np.float32)  
        self.prev_policy = self.policy.copy()
        self.target_pos = self.joint_position.copy()
        self.og_joint_position = self.joint_position.copy()
        self.getpos_error()  #12 joint position error
        self.rg_freq        = np.zeros(4, dtype=np.float32)   # 4 limb frequencies
        self.rg_phase       = np.zeros(4, dtype=np.float32)   # 4 limb phases
        self.binary_phase   = np.zeros(4, dtype=bool)         # stance flags
        self.reward = 0
        self.timestep = timestep

    def getvelocity(self):
        lin_vel, ang_vel = p.getBaseVelocity(self.id)
        self.base_linear_velocity  = np.array(lin_vel, dtype=np.float32)
        self.base_angular_velocity = np.array(ang_vel, dtype=np.float32)

    def getpos_error(self):
        self.pos_error = self.joint_position - self.target_pos

    def getjointinfo(self):
        js = p.getJointStates(self.id, self.movable_joints)
        self.joint_position = np.array([s[0] for s in js], dtype=np.float32)
        self.joint_velocity = np.array([s[1] for s in js], dtype=np.float32)
        self.joint_force    = np.array([s[2] for s in js], dtype=np.float32)
        self.joint_torque   = np.array([s[3] for s in js], dtype=np.float32) / 50.0

    def get_base_info(self):
        pos, quat = p.getBasePositionAndOrientation(self.id)
        self.base_position    = np.array(pos, dtype=np.float32)
        self.base_orientation = np.array(p.getEulerFromQuaternion(quat), dtype=np.float32)  # roll, pitch, yaw
        R = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float32).reshape(3, 3)
        # world gravity (0,0,-1) expressed in base frame = R.T @ g_world
        self.base_gravity = -R[:, 2]   # third column negated

    def get_link_vel(self):
        self.ls  = p.getLinkStates(self.id, self.foot_tips, computeLinkVelocity=1)
        vel = np.array([s[6] for s in self.ls], dtype=np.float32)   # shape (4,3) -> x,y,z
        self.foot_xvel = vel[:, 0]
        self.foot_yvel = vel[:, 1]
        self.foot_zvel = vel[:, 2]
        print("foot xvel:", self.foot_xvel)
        print("foot yvel:", self.foot_yvel)
        print("foot zvel:", self.foot_zvel)

    def get_state(self):
        self.getvelocity()
        self.getjointinfo()
        self.get_base_info()
        self.get_contact()
        self.get_link_vel()
        self.get_height()
        self.getpos_error()

        freq_state = []
        #commands
        cmd   = np.asarray(self.command, np.float32)
        state = np.concatenate([
            cmd,
            self.base_linear_velocity/10, self.base_angular_velocity/10,
            self.base_gravity/(2*np.pi),
            self.joint_position/(2*np.pi), self.joint_velocity/10,
            self.pos_error/(2*np.pi),
            self.rg_freq,
            np.stack([np.sin(self.rg_phase), np.cos(self.rg_phase)], axis=1).ravel()
        ], dtype=np.float32)[None, :]
        return state

    def phase_modulator(self):
        self.rg_phase = (self.rg_phase + 2 * np.pi * self.rg_freq * self.timestep) % (2 * np.pi)
        self.binary_phase = self.rg_phase < np.pi


    def update_target_pos(self,pos_inc):
        self.target_pos = np.clip(self.target_pos + pos_inc,
                                self.joint_lower_limits,
                                self.joint_upper_limits)
        return self.target_pos

    def get_contact(self):
        cps = p.getContactPoints(self.id, self.terrain_id)          # all contacts once
        hit = {cp[3] for cp in cps}                                  # linkIndexA set
        self.is_contact = np.array([li in hit for li in self.foot_links], dtype=bool)
        # print("foot contact:", self.is_contact)

    def get_height(self):
        """Clearance for foot tips via batched rays (heightfield-safe)."""
        r = 0.005  # tip sphere radius

        # tip world positions
        tip_pos = [s[0] for s in self.ls]

        # build ray starts/ends
        starts = [[x, y, z + 0.30] for x, y, z in tip_pos]
        ends   = [[x, y, z - 1.00] for x, y, z in tip_pos]

        # one batch call
        hits = p.rayTestBatch(starts, ends)

        foot_height = []
        for (x, y, z), h in zip(tip_pos, hits):
            if h[0] == self.terrain_id:
                z_hit = h[3][2]
                foot_height.append(max(z - z_hit - r, 0.0))
            else:
                # second attempt: small offset along base +x
                base_R = p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.id)[1])
                x_axis = (base_R[0], base_R[3], base_R[6])
                xo, yo = x + 0.02 * x_axis[0], y + 0.02 * x_axis[1]
                h2 = p.rayTest([xo, yo, z + 0.30], [xo, yo, z - 1.00])[0]
                if h2[0] == self.terrain_id:
                    z_hit = h2[3][2]
                    foot_height.append(max(z - z_hit - r, 0.0))
                else:
                    # fallback cap
                    foot_height.append(0.10)
        # print("foot heights:", foot_height)

        self.foot_height = np.array(foot_height, dtype=np.float32)
    def move(self):

        max_force = 18
        p.setJointMotorControlArray(
            bodyUniqueId=self.id,
            jointIndices=self.movable_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_pos.tolist(),
            forces=[max_force] * len(self.movable_joints)
        )
                
    def set_frequency(self,freq):
        self.rg_freq = freq.copy()

    def update_policy(self,actions):
        self.prev_policy = self.policy.copy()
        self.policy = actions

    def get_reward(self,kc=1):
        c1 = c2 = c3 = c4 =  1.2
        epsilon_min = 0.01
        cx = 1.0/ max(abs(self.command[0]),epsilon_min)
        cy = 1.0/ max(abs(self.command[1]),epsilon_min)
        cw = 1.0/ max(abs(self.command[2]),epsilon_min)

        #intialize reward components
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

        # transform forward and lateral velocity from base frame 
        yaw = self.base_orientation[2]
        fwd_world_frame = np.array([np.cos(yaw), np.sin(yaw), 0])
        lat_world_frame = np.array([-np.sin(yaw), np.cos(yaw), 0])
        fwd_velocity = np.dot(self.base_linear_velocity, fwd_world_frame)
        lat_velocity = np.dot(self.base_linear_velocity, lat_world_frame)

        #compute basic rewards
        forward_velocity = 3*math.exp(-3 * cx * ((fwd_velocity-self.command[0])**2))
        lateral_velocity = 2*math.exp(-3 * cy * ((lat_velocity-self.command[1])**2))
        angular_velocity = 1.5*math.exp(-1.5 *cw*((self.base_angular_velocity[2]-self.command[2])**2))
        balance = 1.3*(math.exp(-4 * cx*((self.base_linear_velocity[2])**2)) + math.exp(-4*cx* ((self.base_angular_velocity[0]**2+ self.base_angular_velocity[1]**2))))
        twist = -0.6 *((self.base_orientation[0]**2 + self.base_orientation[1]**2)**0.5) * cx

        foot_slip     = np.sum((self.foot_xvel**2 + self.foot_yvel**2)[self.binary_phase])
        foot_stance   = np.sum((self.foot_height < 0.018) & self.binary_phase)
        foot_clear    = 0.7 * c1 * np.sum((self.foot_height > 0.023) & (~self.binary_phase))
        frequency_err = np.sum(np.abs(self.rg_freq)[self.binary_phase])
        phase_err     = np.sum(self.is_contact == self.binary_phase)
        foot_zvel1    = np.sum(self.foot_zvel**2)

        # scale
        foot_zvel1    = -0.03 * cx * float(foot_zvel1)
        foot_slip     = -0.07 * cx * np.sqrt(float(foot_slip))
        frequency_err = -0.03 * cx * float(frequency_err)

        #calculate joint constraints
        diff = np.asarray(self.joint_position[:12], np.float32) - np.asarray(self.og_joint_position[:12], np.float32)
        joint_constraints = -0.8 * float(np.dot(diff, diff)) * cx
        #calculate policy smoothness
        dp = np.asarray(self.policy, np.float32) - np.asarray(self.prev_policy, np.float32)
        policy_smooth = float(np.sum(dp * dp)) * -0.016 * c4 * cx

        #torque_penalty = -0.0012 * c2 * cx * np.linalg.norm(self.joint_torque)
        #velocity_penalty = -0.0008 * c3 * cx * np.linalg.norm(self.joint_velocity_error) ** 2

        basic_reward = forward_velocity + lateral_velocity + angular_velocity + balance
        freq_reward = (foot_stance  + foot_clear  + frequency_err + phase_err )
        efficiency_reward =  twist + joint_constraints +  foot_zvel1 + foot_slip + policy_smooth
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
                   "basic_reward":basic_reward,}
                #    "torque_penalty":torque_penalty,
                #    "velocity_penalty":velocity_penalty,}
        
        self.reward = kc["forward"]*forward_velocity + \
                        kc["lateral"]*lateral_velocity + \
                        kc["angular"]*angular_velocity + \
                        kc["balance_twist"]*balance + \
                        kc["balance_twist"]*twist + \
                        kc["rhythm"]*(foot_stance) + \
                        kc["rhythm"]*(foot_clear) + \
                        kc["rhythm"]*(frequency_err) + \
                        kc["rhythm"]*(phase_err) + \
                        kc["rhythm"]*(foot_slip) + \
                        kc["efficiency"]*(joint_constraints) 
                        #kc["efficiency"]*(policy_smooth) + \
                        #kc["efficiency"]*(torque_penalty) + \
                        #kc["efficiency"]*(velocity_penalty)
                        #kc["rhythm"]*(foot_zvel1) + \

        infos = {"rewards":rewards}
        return self.reward,infos

    def is_end(self):
        cps = p.getContactPoints(self.id, self.terrain_id)  # one call
        hit_links = {cp[3] for cp in cps}
        return int(any(li in hit_links for li in self.is_end_links))
