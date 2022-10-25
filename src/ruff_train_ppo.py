#!/usr/bin/env python3

from model_ppo import *
from ruff import *

tfd = tfp.distributions

NUM_EPISODES = 200_000
STEPS_PER_EPISODE = 1000
max_buffer = 12_000
MINIBATCH_SIZE = 4_000
ppo_epochs = 3
timestep =1.0/240.0
num_inputs = (60,)
n_actions = 16
gamma= 0.992
lmbda = 0.95
critic_discount = 0.5
clip_range = 0.2
entropy = 0.0025
kf = 0
ke = 0
kd = 1

bullet_file = "../model/test_ppo.bullet"
log_file = "ppo_ruff_logfile.csv"
reward_log = 'reward_logfile.csv'
graph_count = 0

dummy_n = np.zeros((1, 1, 16))
dummy_1 = np.zeros((1, 1, 1))

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
        self.keys = [i for i in range(max_len)]
        self.batch_size = batch_size

    def append(self,state=None,action=None,reward=None,value=None,logprobs=None,ret=None,adv=None):

        self.states = np.concatenate([self.states,state]) if type(state)!=type(None) else 0
        self.actions = np.concatenate([self.actions,np.array(action).reshape((1,16))]) if action !=None else 0
        self.rewards = np.concatenate([self.rewards,np.array(reward).reshape((1,1))]) if reward!=None else 0
        self.values = np.concatenate([self.values,value]) if value!=None else 0
        self.logprobs = np.concatenate([self.logprobs,logprobs]) if logprobs!=None else 0
        self.advantages = np.concatenate([self.advantages,adv]) if adv!=None else 0
        self.returns = np.concatenate([self.returns,ret]) if ret!=None else 0

        self.states = self.states[-self.max_len:]
        self.actions = self.actions[-self.max_len:]
        self.rewards = self.rewards[-self.max_len:]
        self.values = self.values[-self.max_len:]
        self.logprobs = self.logprobs[-self.max_len:]
        self.advantages = self.advantages[-self.max_len:]
        self.adv = (self.advantages-np.mean(self.advantages))/(np.std(self.advantages)+1e-10)
        self.returns = self.returns[-self.max_len:]
    def batch_gen(self):
        np.random.shuffle(self.keys)
        batch = np.array_split(self.keys,self.max_len/self.batch_size)
        for i in batch:
            state = np.take(self.states,i,0)
            action = np.take(self.actions,i,0)
            log_prob = np.take(self.logprobs,i,0)
            returns = np.take(self.returns,i,0)
            rewards = np.take(self.rewards,i,0)
            advantages = np.take(self.adv,i,0)
            yield state,log_prob,action,returns,advantages,rewards




    def __len__(self):
        #print(len(self.masks))
        return len(self.states)

def log_episode(log_file,episode,episode_reward,step,new = 0):
    data = [[episode,episode_reward,step]]
    if new == 1:
        os.remove(log_file)
    with open(log_file, 'a', newline="") as file:
        csvwriter = csv.writer(file) # 2. create a csvwriter object
        csvwriter.writerows(data) # 5. write the rest of the data

def check_log(filename):
    files = os.listdir("../logs/")
    filecode = len(files)+1
    filename = '../logs/ '+filename + "_"+str(filecode)+ ".csv"
    return filename

def run_episode(actor,critic,STEPS_PER_EPISODE,rubuff,ru,episode):
    for step in range(STEPS_PER_EPISODE):
        state_curr = ru.get_state()

        mu,sigma = actor([state_curr])

        critic_value = critic(state_curr)
        pos_inc, freq, actions, log_probs = ru.action_select(mu,sigma)
        ru.set_frequency(freq)
        ru.phase_modulator()
        ru.update_policy(actions)

        ru.update_target_pos(pos_inc)
        ru.move()
        p.stepSimulation()
        new_state = ru.get_state()
        reward = ru.get_reward(episode,step)
        ret = reward + gamma*critic(new_state)
        adv = ret - critic_value
        if ru.is_end():
            pass

        rubuff.append(state_curr,actions,reward,critic_value,log_probs,ret,adv)
        if ru.is_end():
            break

    return step

if __name__=="__main__":
    filename =check_log(filename)
    ru = ruff(id,kf,ke)
    actor = actor_Model(num_inputs, n_actions)
    critic = critic_Model(num_inputs, 1)
    try:
        actor.load_weights("actor.h5")
        critic.load_weights("critic.h5")

    except:
        pass


    rubuff = buffer(max_buffer,MINIBATCH_SIZE)
    for episode in range(NUM_EPISODES ):
        if episode == 0:
            log_episode(log_file,"episode","avg_eps_reward","step",1)

        reset_world(bullet_file)
        ru = ruff(id,kf,ke)

        step = run_episode(actor,critic,STEPS_PER_EPISODE,rubuff,ru,episode)
        #returns, advantages = get_advantages(rubuff.values, rubuff.masks, rubuff.rewards)
        #rint(len(rubuff.logprobs))


        episode_reward = np.sum(rubuff.rewards[-step:])
        print("episode: "+str(episode)+" steps: "+str(step)+" episode_reward: "+str(episode_reward))

        if len(rubuff)==max_buffer:
            for i in range(ppo_epochs):

                for states,logprobs,actions,returns,advantages,rewards in rubuff.batch_gen():

                    act_loss,crit_loss=ruff_train(actor,critic,states,logprobs,actions,returns,advantages,rewards)

                graph_count+=1
            save_model(actor,critic)
        else:
            print("buffer size = "+str(len(rubuff)))
        log_episode(log_file,episode,episode_reward/step,step)

    close_world()
