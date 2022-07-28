import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense

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

    model = keras.Model(inputs=[inputs, oldpolicy_probs, advantages, rewards, values], outputs=[mu,sigma])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    return model

def critic_Model(Input_shape,output_size):
    inputs = Input(shape=(Input_shape))
    X = Dense(256, activation="relu")(inputs)
    X = Dense(256, activation="relu")(X)
    X = Dense(256, activation="relu")(X)
    X = Dense(output_size)(X)

    model = keras.Model(inputs=inputs, outputs=X)
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    return model
