import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense
import tensorflow_probability as tfp
import numpy as np
import keras.backend as K
gamma= 0.992
lmbda = 0.95
critic_discount = 0.5
clipping_val = 0.2
entropy = 0.0025
tfd = tfp.distributions

act_optimizer = keras.optimizers.SGD(learning_rate=0.0003,clipnorm=1.0)
cri_optimizer = keras.optimizers.SGD(learning_rate=0.0003,clipnorm = 1.0)

def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])
    #print(len(returns))
    adv = np.array(returns).reshape((-1,1)) - np.array(values[:-1]).reshape((-1,1))
    #print(np.mean(adv))
    #print(np.std(adv))
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    #print(adv.shape)
    return returns, adv

def ruff_train(actor,critic,rubuff,returns,advantages):
    with tf.GradientTape(persistent = True) as tape:
        old_log_probs = rubuff.logprobs
        mu,sigma = actor(rubuff.states)

        dist = tfd.Normal(loc=mu, scale=sigma)
        new_log_probs = log_probs = dist.log_prob(rubuff.actions)
        ratio = K.exp(new_log_probs - old_log_probs + 1e-10)
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.sum(K.minimum(p1, p2))
        critic_loss = K.sum(K.square(returns - critic(rubuff.states)))
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    act_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    cri_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    return actor_loss,critic_loss


def actor_Model(Input_shape,output_size):
    inputs = Input(shape=(Input_shape))
    X = Dense(256, activation="relu", name="fc1")(inputs)
    X = Dense(256, activation="relu", name="fc2")(X)
    mu = Dense(output_size, activation="tanh", name="mean")(X)
    sigma = Dense(output_size, activation="sigmoid", name="sigma")(X)
    model = keras.Model(inputs=inputs, outputs=[mu,sigma])
    try:
        model.load_weights("../model/ppo_actor.h5")
        print("loaded actor weights")
    except:
        print("unable to load actor weights")
    return model

def critic_Model(Input_shape,output_size):
    inputs = Input(shape=(Input_shape))
    X = Dense(256, activation="relu")(inputs)
    X = Dense(256, activation="relu")(X)
    X = Dense(output_size)(X)
    model = keras.Model(inputs=inputs, outputs=X)
    try:
        model.load_weights("../model/ppo_critic.h5")
        print("loaded critic weights")
    except:
        print("unable to load critic weights")
    return model

def save_model(actor,critic):
    act_json = actor.to_json()
    cri_json = critic.to_json()
    with open("../model/ppo_actor.json","w") as json_file:
        json_file.write(act_json)
    with open("../model/ppo_critic.json","w") as json_file:
        json_file.write(cri_json)
    actor.save_weights("../model/ppo_actor.h5")
    critic.save_weights("../model/ppo_critic.h5")
    print("model saved")
