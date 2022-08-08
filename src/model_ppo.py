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
clip_range = 0.2
entropy = 0.0025
tfd = tfp.distributions


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        #print("a")
        print(y_pred.shape)
        (mu,sigma) = tf.split(y_pred, num_or_size_splits=2, axis=1)
        mu,sigma = y_pred.shape
        dist = tfd.Normal(loc=mu, scale=sigma)
        actions = dist.sample(1)
        actions = actions.numpy().tolist()[0][0]
        newpolicy_probs = dist.log_prob(actions)
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        total_loss = K.sum(y_pred)
        return total_loss

    return loss

def actor_Model(Input_shape,output_size):
    inputs = Input(shape=(Input_shape))
    oldpolicy_probs = Input(shape=(1, output_size,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    X = Dense(256, activation="relu", name="fc1")(inputs)
    X = Dense(256, activation="relu", name="fc2")(X)
    X = Dense(256, activation="relu", name="fc3")(X)
    mu = Dense(output_size, activation="tanh", name="mean")(X)
    sigma = Dense(output_size, activation="softplus", name="sigma")(X)
    output = tf.keras.layers.Concatenate()([mu,sigma])

    model = keras.Model(inputs=[inputs, oldpolicy_probs, advantages, rewards, values], outputs=output)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.00035,clipnorm=1.0), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
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
    X = Dense(256, activation="relu")(X)
    X = Dense(output_size)(X)

    model = keras.Model(inputs=inputs, outputs=X)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.00035,clipnorm=1.0), loss='mse')

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
