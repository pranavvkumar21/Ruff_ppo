#!/usr/bin/env python3

import gym
from gym.wrappers import HumanRendering
import numpy as np
import sys
sys.path.append("..")
from model_ppo import *

print("Dependencies and models imported.")

tfd = tfp.distributions
env1 = gym.make("MountainCar-v0",render_mode='rgb_array')
wrapped = HumanRendering(env1)
print(dir)
episodes = 10000
num_inputs = (2,)
actor = actor_Model(num_inputs, 1)
critic = critic_Model(num_inputs, 1)
test_eps = 50

print("Environment setup complete. Models initialized.")

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

    def append(self, state=None, action=None, reward=None, value=None, logprobs=None, masks=None):
        if state is not None: self.states.append(state)
        if action is not None: self.actions.append(action)
        if reward is not None: self.rewards.append(reward)
        if value is not None: self.values.append(value)
        if logprobs is not None: self.logprobs.append(logprobs)
        if masks is not None: self.masks.append(masks)

        #print(f"Data appended to buffer: State={state}, Action={action}, Reward={reward}, Value={value}")

def action_select(mu, sigma):
    dist = tfd.Normal(loc=mu, scale=sigma)
    action = dist.sample()
    log_probs = dist.log_prob(action)
    action = action.numpy()[0,0]
    #print(f"Action selected: {action}, mu={mu}, sigma={sigma}, log_probs={log_probs}")
    return action, log_probs

def ruff_train(buff, returns, advantages):
    with tf.GradientTape(persistent=True) as tape:
        old_log_probs = buff.logprobs
        mu, sigma = actor(buff.states)
        dist = tfd.Normal(loc=mu, scale=sigma)
        new_log_probs = dist.log_prob(buff.actions)
        ratio = tf.exp(new_log_probs - old_log_probs + 1e-10)
        p1 = ratio * advantages
        p2 = tf.clip_by_value(ratio, 1 - clipping_val, 1 + clipping_val) * advantages
        actor_loss = -tf.reduce_mean(tf.minimum(p1, p2))
        critic_loss = tf.reduce_mean(tf.square(returns - critic(buff.states)))
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    act_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    cri_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
        #print(f"Training update: Actor Loss={actor_loss}, Critic Loss={critic_loss}")

env1.reset()
for i in range(episodes):
    if i%test_eps ==0:
        is_test = True
    else:
        is_test = False
    if is_test:
        env = wrapped
    else:
        env = env1
    env.reset()
    state = np.array(env.reset()[0]).reshape((1, -1))
    done = False
    trunc = False
    buff = buffer()
    step_count = 0
    total_reward = 0
    print(f"Episode {i+1} started.")
    while not done and (not trunc):
        mu, sigma = actor(state)
        critic_value = critic(state)
        if is_test:
            sigma = 0
        action, log_prob = action_select(mu, sigma)

        if action < -0.33:
            act = 0
        elif action > 0.33:
            act = 2
        else:
            act = 1
        next_state, reward, done, trunc, _ = env.step(act)
        reward = reward + abs(next_state[0])
        total_reward = total_reward+reward
        next_state = next_state.reshape((1, -1))
        buff.append(state, action, reward, critic_value, log_prob, not done)
        state = next_state
        step_count += 1
        if is_test:
            #print(i)
            env.render()
        #print(step_count)
        #print(trunc)
    #env.close()
    critic_value = critic(state)
    buff.append(value = critic_value)
    print(f"Episode {i+1} completed with {step_count} steps.")
    print("total rewards: "+str(reward))
    returns, advantages = get_advantages(buff.values, buff.masks, buff.rewards)
    buff.states = np.array(buff.states, dtype="float32").reshape((-1, 2))
    if not is_test:
        for _ in range(3):
            ruff_train(buff, returns, advantages)
    save_model(actor, critic,save_path=".")

