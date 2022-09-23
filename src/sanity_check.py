#!/usr/bin/env python3

import gym
import numpy as np
from model_ppo import *

tfd = tfp.distributions
env = gym.make("MountainCar-v0")
episodes = 10000
num_inputs = (2,)
actor = actor_Model(num_inputs, 1)
critic = critic_Model(num_inputs, 1)

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
def action_select(mu,sigma):
    dist = tfd.Normal(loc=mu, scale=sigma)
    actions = dist.sample(1)
    actions = actions.numpy()[0,0,0]
    #print(actions[0,0,0])
    if actions < -0.5:
        action = 0
    elif actions > 0.5:
        action = 2
    else:
        action = 1
    log_probs = dist.log_prob(actions)
    return action,log_probs

def ruff_train(buff,returns,advantages):
    with tf.GradientTape(persistent = True) as tape:
        old_log_probs = buff.logprobs
        mu,sigma = actor(buff.states)

        dist = tfd.Normal(loc=mu, scale=sigma)
        new_log_probs = log_probs = dist.log_prob(buff.actions)
        ratio = K.exp(new_log_probs - old_log_probs + 1e-10)
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(returns - critic(buff.states)))
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    act_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    cri_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    return actor_loss,critic_loss

env.reset()
for i in range(episodes):
    state_ = np.array(env.reset()).reshape((1,-1,))
    done=False
    buff = buffer()
    count = 0
    print(i)
    while not done:
        state = state_
        mu,sigma = actor(state)
        critic_value = critic(state)
        action,log_prob = action_select(mu,sigma)
        state_,reward,done,_ = env.step(action)
        state_ = np.array(state_).reshape((1,-1,))
        buff.append(state,action,reward,critic_value,log_prob,not done)
        count+=1
        if i%100==0:
            env.render()
    critic_value = critic(state_)
    buff.append(value = critic_value)
    returns, advantages = get_advantages(buff.values, buff.masks, buff.rewards)
    buff.states = np.array(buff.states,dtype= "float32").reshape((-1,2))
    for i in range(3):
        ruff_train(buff,returns,advantages)
    save_model(actor,critic)
    #print(type(buff.values[5]))
