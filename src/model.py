import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input,Dense

def actor_Model(Input_shape,output_size, layer1_size, layer2_size, layer3_size):
    inputs = Input(shape=(Input_shape))
    X = Dense(layer1_size, activation="relu", name="fc1")(inputs)
    X = Dense(layer2_size, activation="relu", name="fc2")(X)
    X = Dense(layer3_size, activation="relu", name="fc3")(X)
    mu = Dense(output_size, activation="tanh", name="mean")(X)
    sigma = Dense(output_size, activation="softplus", name="sigma")(X)

    model = keras.Model(inputs=inputs, outputs=[mu,sigma])
    return model

def critic_Model(Input_shape,output_size, layer1_size, layer2_size, layer3_size):
    inputs = Input(shape=(Input_shape))
    X = Dense(layer1_size, activation="relu")(inputs)
    X = Dense(layer2_size, activation="relu")(X)
    X = Dense(layer3_size, activation="relu")(X)
    X = Dense(output_size)(X)

    model = keras.Model(inputs=inputs, outputs=X)
    return model
