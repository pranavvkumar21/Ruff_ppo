#!/usr/bin/env python3

from model_ppo import *
from ruff import *
import gc
import time
from sklearn.utils import shuffle

#----------pybullet configuration-------------------
client_mode = p.GUI
timestep =1.0/240.0
bullet_file = "../model/test_ppo.bullet"
n_actors = 1

#-----------model configuration----------------------
tfd = tfp.distributions
num_inputs = (60,)
n_actions = 16
gamma= 0.992
lmbda = 0.95
critic_discount = 0.5
clip_range = 0.2
entropy = 0.0025

#-----------log configuration------------------------
max_reward = float("-inf")
log_file = "../logs/ppo_ruff_logfile.csv"
reward_log = "../logs/ruff_reward_log.csv"
reward_list = ["forward_velocity","lateral_velocity","angular_velocity","Balance",
           "foot_stance", "foot_clear","foot_zvel", "frequency_err", "phase_err",
           "joint_constraints", "foot_slip", "policy_smooth","twist"]

#------------train configuration---------------------
LOAD = True
Train = False
NUM_EPISODES = 200_000
STEPS_PER_EPISODE = 1000
max_buffer = 9000
MINIBATCH_SIZE = 12000
ppo_epochs = 5
kc = 2e-10
kd = 0.999994

#kc override
kc = 0.999999999999



class buffer:
    def __init__(self,max_len,batch_size):
        self.states = np.empty((0,60))
        self.rewards = np.empty((0,1))
        self.actions = np.empty((0,16))
        self.logprobs = np.empty((0,16))
        self.values = np.empty((0,1))
        self.advantages = np.empty((0,1))
        self.returns = np.empty((0,1))
        self.max_len = max_len
        self.batch_size = batch_size
    def clear(self):
        self.states = np.empty((0,60))
        self.rewards = np.empty((0,1))
        self.actions = np.empty((0,16))
        self.logprobs = np.empty((0,16))
        self.values = np.empty((0,1))
        self.advantages = np.empty((0,1))
        self.returns = np.empty((0,1))
    def batch_gen(self):
        self.states,self.rewards,self.actions,self.logprobs,self.values,self.advantages = shuffle(self.states,self.rewards,self.actions,
                                                                                                  self.logprobs,self.values,self.advantages )
        n_batchs = (self.states.shape[0]//self.batch_size)+1
        idx = 0
        for i in range(n_batchs):
            state = self.states[idx:idx+self.batch_size,:]
            action = self.actions[idx:idx+self.batch_size,:]
            log_prob = self.logprobs[idx:idx+self.batch_size,:]
            returns = self.returns[idx:idx+self.batch_size,:]
            rewards = self.rewards[idx:idx+self.batch_size,:]
            advantages = self.advantages[idx:idx+self.batch_size,:]
            idx = idx+self.batch_size
            yield state,log_prob,action,returns,advantages,rewards

    def __len__(self):
          return len(self.states)

def log_episode(log_file,episode,episode_reward,step,act_loss=0,crit_loss=0,new = False):
    data = [[episode,episode_reward,step,act_loss,crit_loss]]
    if new:
        try:
            os.remove(log_file)
        except:
            print("no log file found")
    with open(log_file, 'a', newline="") as file:
        csvwriter = csv.writer(file) # 2. create a csvwriter object
        csvwriter.writerows(data) # 5. write the rest of the data
def log_reward(reward_log,data,new=False):
    if new:
        try:
            os.remove(reward_log)
        except:
            print("reward log file not found")
    else:
        data = data
    with open(reward_log, 'a', newline="") as file:
        csvwriter = csv.writer(file) # 2. create a csvwriter object
        csvwriter.writerow(data) # 5. write the rest of the data

def check_log(filename):
    files = os.listdir("../logs/")
    filecode = len(files)+1
    filename = '../logs/ '+filename + "_"+str(filecode)+ ".csv"
    return filename

def run_episode(actor,critic,STEPS_PER_EPISODE,rubuff,ruff_s,episode):
    eps_states = [[] for i in range(len(ruff_s))]
    eps_actions = [[] for i in range(len(ruff_s))]
    eps_rewards = [[] for i in range(len(ruff_s))]
    eps_critic_value = [[] for i in range(len(ruff_s))]
    eps_log_probs = [[] for i in range(len(ruff_s))]
    masks=[[] for i in range(len(ruff_s))]
    rewards=[[] for i in range(len(ruff_s))]
    rets = [[] for i in range(len(ruff_s))]
    advs = [[] for i in range(len(ruff_s))]
    end_flag = [0 for i in range(len(ruff_s))]
    rews = [0]*13

    total_steps = 0
    for step1 in range(STEPS_PER_EPISODE):
        if sum(end_flag)==4:
            break
        global kc, kd
        kc = kc**kd
        for i,ru in enumerate(ruff_s):
            if not end_flag[i] :
                state_curr = ru.get_state()
                mu,sigma = actor([state_curr])
                critic_value = critic(state_curr)
                pos_inc, freq, actions, log_probs = ru.action_select(mu,sigma)
                ru.set_frequency(freq)
                ru.phase_modulator()
                ru.update_policy(actions)
                ru.update_target_pos(pos_inc)
                ru.move()

                eps_states[i].append(state_curr)
                eps_actions[i].append(np.array(actions).reshape((1,16)))
                eps_log_probs[i].append(log_probs)
                eps_critic_value[i].append(critic_value)

        for i in range(3):
            p.stepSimulation()

        for i,ru in enumerate(ruff_s):
            if not end_flag[i]:
                new_state = ru.get_state()
                reward,re = ru.get_reward(  kc)
                eps_rewards[i].append(np.array(reward).reshape((1,1)))
                for j in range(len(re)):
                    rews[j]+=re[j]
                if ru.is_end() and not end_flag[i]:
                    masks[i].append(0)
                    end_flag[i] = 1
                    critic_value = critic(new_state)
                    eps_critic_value[i].append(critic_value)

                    ret,adv = get_advantages(eps_critic_value[i], masks[i], eps_rewards[i])
                    rets[i].append(ret)
                    advs[i].append(adv)
                    total_steps+=step1

                elif not end_flag[i]:
                    masks[i].append(1)

    for i,ru in enumerate(ruff_s):
        if not end_flag[i]:
            new_state = ru.get_state()
            critic_value = critic(new_state)
            masks[i][-1]=0
            eps_critic_value[i].append(critic_value)
            ret,adv = get_advantages(eps_critic_value[i], masks[i], eps_rewards[i])
            rets[i].append(ret)
            advs[i].append(adv)
            total_steps+=step1

    rubuff.states = np.concatenate([rubuff.states,np.concatenate([np.concatenate(st,axis=0) for st in eps_states],axis=0)],axis=0)
    rubuff.actions = np.concatenate([rubuff.actions,np.concatenate([np.concatenate(act,axis=0) for act in eps_actions],axis=0)],axis=0)
    rubuff.rewards = np.concatenate([rubuff.rewards,np.concatenate([np.concatenate(rew,axis=0) for rew in eps_rewards],axis=0)],axis=0)
    rubuff.values = np.concatenate([rubuff.values,np.concatenate([np.concatenate(cri[:-1],axis=0) for cri in eps_critic_value],axis=0)],axis=0)
    rubuff.logprobs = np.concatenate([rubuff.logprobs,np.concatenate([np.concatenate(lp,axis=0) for lp in eps_log_probs],axis=0)],axis=0)
    rubuff.returns = np.concatenate([rubuff.returns,np.concatenate([np.concatenate(r,axis=0) for r in rets],axis=0).reshape((-1,1))],axis=0)
    rubuff.advantages = np.concatenate([rubuff.advantages,np.concatenate([np.concatenate(a,axis=0) for a in advs],axis=0)],axis=0)
    #rubuff.states = (rubuff.states-np.mean(rubuff.states,0))/(np.std(rubuff.states,0)+1e-10)


    total_steps+=len(ruff_s)
    return total_steps,[i/total_steps for i in rews]

if __name__=="__main__":
    id  = setup_world(n_actors,client_mode)
    filename =check_log(filename)
    ruff_s = [ruff(i) for i in id]
    actor = actor_Model(num_inputs, n_actions,load=LOAD)
    critic = critic_Model(num_inputs, 1,load=LOAD)

    rubuff = buffer(max_buffer,MINIBATCH_SIZE)
    for episode in range(NUM_EPISODES ):
        #
        start_t = time.time()
        if episode == 0:
            if not LOAD:
                log_episode(log_file,"episode","avg_eps_reward","step","act_loss","crit_loss",True)
                log_reward(reward_log,reward_list,new=1)
            save_world(bullet_file)
        reset_world(bullet_file)
        gc.collect()
        rubuff.clear()
        #kc1 = kc
        step = 0
        while len(rubuff)<max_buffer:
            ruff_s = [ruff(i) for i in id]

            st,rew_mean = run_episode(actor,critic,STEPS_PER_EPISODE,rubuff,ruff_s,episode)
            #kc = kc1
            step+=st
            print(st)
            reset_world(bullet_file)


        episode_reward = np.sum(rubuff.rewards)
        print("episode: "+str(episode)+" steps: "+str(step)+" episode_reward: "+str(episode_reward))
        print("kc: "+str(kc)+"   curriculum reward: "+str(sum(rew_mean[4:])))
        print("buffer size = "+str(len(rubuff)))
        if Train:
            for i in range(ppo_epochs):

                for states,logprobs,actions,returns,advantages,rewards in rubuff.batch_gen():

                    act_loss,crit_loss=ruff_train(actor,critic,states,logprobs,actions,returns,advantages,rewards)

            save_model(actor,critic)
            if episode_reward>=max_reward:
                save_model(actor,critic,1)
                max_reward = episode_reward

            #
            log_episode(log_file,episode,episode_reward/len(rubuff.rewards),step,float(act_loss),float(crit_loss))
            log_reward(reward_log,rew_mean,0)
        print("Time elapsed: {:.1f}".format(time.time()-start_t))

    close_world()
