#!/usr/bin/env python3

from model_ppo import *
from ruff import *

tfd = tfp.distributions

NUM_EPISODES = 1_000
STEPS_PER_EPISODE = 1_00
timestep =1.0/240.0
num_inputs = (60,)
n_actions = 16
gamma= 0.992
lmbda = 0.95
critic_discount = 0.5
clip_range = 0.2
entropy = 0.0025
kc = 0
kd = 1

bullet_file = "../model/test_ppo.bullet"
filename = "ruff_logfile"
reward_log = 'reward_logfile.csv'


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


def check_log(filename):
    files = os.listdir("../logs/")
    filecode = len(files)+1
    filename = '../logs/ '+filename + "_"+str(filecode)+ ".csv"
    return filename

def run_episode(actor,critic,STEPS_PER_EPISODE,rubuff,ru,episode):
    for step in range(STEPS_PER_EPISODE):
        state_curr = ru.get_state()
        output = actor([state_curr,dummy_n,dummy_1,dummy_1,dummy_1])
        #print((output.shape))
        (mu,sigma) = tf.split(output, num_or_size_splits=2, axis=1)
        #print(mu.shape)
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
        mask = 0 if (step==STEPS_PER_EPISODE-1) else 1
        rubuff.append(state_curr,actions,reward,critic_value,log_probs,mask)

    critic_value = critic(new_state)
    rubuff.append(value = critic_value)

if __name__=="__main__":
    filename =check_log(filename)
    ru = ruff(id,kc)
    actor = actor_Model(num_inputs, n_actions)
    critic = critic_Model(num_inputs, 1)
    try:
        actor.load_weights("actor.h5")
        critic.load_weights("critic.h5")

    except:
        pass



    for episode in range(NUM_EPISODES ):
        rubuff = buffer()
        reset_world(bullet_file)
        ru = ruff(id,kc)

        run_episode(actor,critic,STEPS_PER_EPISODE,rubuff,ru,episode)
        returns, advantages = get_advantages(rubuff.values, rubuff.masks, rubuff.rewards)
        rubuff.states = np.array(rubuff.states,dtype= "float32")
        rubuff.logprobs = np.array(rubuff.logprobs,dtype= "float32")
        rubuff.states = np.reshape(rubuff.states, newshape=(-1, 60))
        rubuff.values = np.array(rubuff.values,dtype= "float32")
        #rubuff.logprobs = np.zeros((100,1,1))
        #print(np.reshape(rubuff.actions, newshape=(-1, n_actions)))
        print(rubuff.states.shape)
        actor_loss = actor.fit(
        [rubuff.states, rubuff.logprobs, advantages, np.reshape(rubuff.rewards, newshape=(-1, 1, 1)), rubuff.values[:-1]],
        [(np.reshape(rubuff.actions, newshape=(-1, n_actions))),(np.reshape(rubuff.actions, newshape=(-1, n_actions)))], verbose=True, shuffle=True, epochs=8)
        critic_loss = critic.fit([states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                    verbose=True)
        print(len(returns))
        print(len(advantages))
        save_model(actor,critic)


    close_world()
