#!/usr/bin/env python3

from model_ppo import *
from ruff import *

tfd = tfp.distributions

NUM_EPISODES = 5_000_000
STEPS_PER_EPISODE = 1000
ppo_epochs = 10
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
    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.masks = []
        self.advantages = None
        self.returns = None

    def append(self,state=None,action=None,reward=None,value=None,logprobs=None,masks=None):
        self.states.append(state) if type(state)!=type(None) else 0
        self.actions.append(action) if action !=None else 0
        self.rewards.append(reward) if reward!=None else 0
        self.values.append(value) if value!=None else print("hi")
        self.logprobs.append(logprobs) if logprobs!=None else 0
        self.masks.append(masks) if masks!=None else 0
    def __len__(self):
        print(len(self.masks))
        return len(self.states)

def log_episode(log_file,episode,act_loss,critic_loss,episode_reward,step,new = 0):
    data = [[episode,act_loss,critic_loss,episode_reward,step]]
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
        if ru.is_end():
            reward = -1000
            print("1212")
        mask = 0 if (step==STEPS_PER_EPISODE-1) else 1
        rubuff.append(state_curr,actions,reward,critic_value,log_probs,mask)
        if ru.is_end():
            break

    critic_value = critic(new_state)
    rubuff.append(value = critic_value)
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



    for episode in range(NUM_EPISODES ):
        if episode == 0:
            log_episode(log_file,"episode","act_loss","crit_loss","eps_reward","step",1)
        rubuff = buffer()
        reset_world(bullet_file)
        ru = ruff(id,kf,ke)

        step = run_episode(actor,critic,STEPS_PER_EPISODE,rubuff,ru,episode)
        returns, advantages = get_advantages(rubuff.values, rubuff.masks, rubuff.rewards)
        rubuff.states = np.array(rubuff.states,dtype= "float32")
        rubuff.logprobs = np.array(rubuff.logprobs,dtype= "float32").reshape((-1,16))
        rubuff.states = np.reshape(rubuff.states, newshape=(-1, 60))
        rubuff.values = np.array(rubuff.values,dtype= "float32")
        rubuff.actions = np.array(rubuff.actions,dtype= "float32").reshape((-1,16))
        episode_reward = np.sum(rubuff.rewards)
        print("episode: "+str(episode)+" steps: "+str(step)+" episode_reward: "+str(episode_reward))
        #for i in range(len(rubuff.states)):
        #    if np.max(rubuff.states[i])>1:
        #        print(np.argmax(rubuff.states[i]))

        for i in range(ppo_epochs):
            act_loss,crit_loss=ruff_train(actor,critic,rubuff,returns,advantages)
            log_episode(log_file,graph_count,float(act_loss.numpy()),float(crit_loss.numpy()),episode_reward,step)
            graph_count+=1
        save_model(actor,critic)


    close_world()
