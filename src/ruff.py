class ruff:
    def __init__(self, id,kc):
        self.id = id
        self.command = [0.3 , 0.0000000001, 0.0000000001] #3 commands for motion
        self.kc = kc
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
        self.prev_policy = self.policy
        self.target_pos = [0.0]*12
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
        self.joint_state = p.getJointStates(id,n_joints)
        for i in self.joint_state:
            self.joint_position.append((i[0]/(2*math.pi)))
            self.joint_velocity.append(i[1]/50.0)
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
        state = state + list(self.joint_position)+list(self.joint_velocity)
        state = state + list([i/(2*math.pi) for i in self.pos_error])
        state = state + list(self.rg_freq)
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
            self.target_pos[i]+=pos_inc[i]

    def get_contact(self):
        self.is_contact = [p.getContactPoints(1,0,linkIndexA=2)!=(),p.getContactPoints(1,0,linkIndexA=5)!=(),
        p.getContactPoints(1,0,linkIndexA=8)!=(),p.getContactPoints(1,0,linkIndexA=11)!=()]
    def get_height(self):
        self.foot_height = [p.getClosestPoints(1,0,50000.0,linkIndexA=2)[0][8],p.getClosestPoints(1,0,50000.0,linkIndexA=5)[0][8],p.getClosestPoints(1,0,50000.0,linkIndexA=8)[0][8],p.getClosestPoints(1,0,50000.0,linkIndexA=11)[0][8]]
    def move(self):
        max_force = [50]*12
        p.setJointMotorControlArray(self.id,self.n_joints,controlMode=p.POSITION_CONTROL,
                                    targetPositions = self.target_pos,forces=max_force)
    def set_frequency(self,freq):
        self.rg_freq = freq

    def update_policy(self,actions):
        self.prev_policy = self.policy
        self.policy = actions

    def get_reward(self,episode,step):
        c1 = 1.2
        c4 = 7.5

        forward_velocity = 2*math.exp(-3 * ((self.base_linear_velocity[0]-self.command[0])**2)/abs(self.command[0]))
        lateral_velocity = 2*math.exp(-3 * ((self.base_linear_velocity[1]-self.command[1])**2)/abs(self.command[1]))
        angular_velocity = 1.5*math.exp(-1.5 * ((self.base_angular_velocity[2]-self.command[2])**2)/abs(self.command[2]))
        Balance = 1.3*(math.exp(-2.5 * ((self.base_linear_velocity[2])**2)/abs(self.command[0])) + math.exp(-2* ((self.base_angular_velocity[0]**2+ self.base_angular_velocity[1]**2))/abs(self.command[0])))
        twist = -0.6 *((self.base_orientation[0]**2 + self.base_orientation[1]**2)**0.5)/abs(self.command[0])

        if p.getContactPoints(1,0,linkIndexA=-1)!=() or p.getContactPoints(1,0,linkIndexA=0)!=() or p.getContactPoints(1,0,linkIndexA=1)!=() or p.getContactPoints(1,0,linkIndexA=3)!=() or p.getContactPoints(1,0,linkIndexA=4)!=() or p.getContactPoints(1,0,linkIndexA=6)!=() or p.getContactPoints(1,0,linkIndexA=7)!=() or p.getContactPoints(1,0,linkIndexA=9)!=() or p.getContactPoints(1,0,linkIndexA=10)!=():
            collision = -8
        else:
            collision = 0

        foot_slip = 0
        foot_stance = 0
        foot_clear = 0
        foot_zvel1 = 0
        policy_smooth = 0
        joint_constraints = 0
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
        policy_smooth = -0.016 * c4 * (policy_smooth**0.5)/abs(self.command[0])
        foot_zvel1 = (-0.03*foot_zvel1**2)/abs(self.command[0])
        foot_slip = -0.07*(foot_slip**0.5)/abs(self.command[0])
        frequency_err = -0.03*frequency_err
        joint_constraints = -0.8*(joint_constraints)/abs(self.command[0])

        curriculum_reward = self.kc* (Balance + twist + foot_stance + foot_clear + foot_zvel1 + joint_constraints  + frequency_err + phase_err + foot_slip + policy_smooth)
        self.reward = forward_velocity + lateral_velocity + angular_velocity + curriculum_reward
        return self.reward

    def is_end(self):
        if p.getContactPoints(1,0,linkIndexA=-1)!=() or p.getContactPoints(1,0,linkIndexA=0)!=() or p.getContactPoints(1,0,linkIndexA=1)!=() or p.getContactPoints(1,0,linkIndexA=3)!=() or p.getContactPoints(1,0,linkIndexA=4)!=() or p.getContactPoints(1,0,linkIndexA=6)!=() or p.getContactPoints(1,0,linkIndexA=7)!=() or p.getContactPoints(1,0,linkIndexA=9)!=() or p.getContactPoints(1,0,linkIndexA=10)!=() :
            return 1
        else:
            return 0
