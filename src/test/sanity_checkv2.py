#!/usr/bin/env python3

import gym
import numpy as np
import sys
sys.path.append("..")
from model_ppo import *

print("Dependencies and models imported.")

tfd = tfp.distributions
env = gym.make("MountainCar-v0")
episodes = 10000
num_inputs = (2,)
actor = actor_Model(num_inputs, 1)
critic = critic_Model(num_inputs, 1)

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

        print(f"Data appended to buffer: State={state}, Action={action}, Reward={reward}, Value={value}")

def action_select(mu, sigma):
    dist = tfd.Normal(loc=mu, scale=sigma)
    action = dist.sample()
    log_probs = dist.log_prob(action)
    action = action.numpy()
    print(f"Action selected: {action}, mu={mu}, sigma={sigma}, log_probs={log_probs}")
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
        print(f"Training update: Actor Loss={actor_loss}, Critic Loss={critic_loss}")

env.reset()
for i in range(episodes):
    state = np.array(env.reset()).reshape((1, -1))
    done = False
    buff = buffer()
    step_count = 0
    print(f"Episode {i+1} started.")
    while not done:
        mu, sigma = actor(state)
        critic_value = critic(state)
        action, log_prob = action_select(mu, sigma)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape((1, -1))
        buff.append(state, action, reward, critic_value, log_prob, not done)
        state = next_state
        step_count += 1
        if i % 100 == 0:
            env.render()
    print(f"Episode {i+1} completed with {step_count} steps.")
    returns, advantages = get_advantages(buff.values, buff.masks, buff.rewards)
    buff.states = np.array(buff.states, dtype="float32").reshape((-1, 2))
    for _ in range(3):
        ruff_train(buff, returns, advantages)
    save_model(actor, critic)

