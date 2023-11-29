from keras import layers
from tensorflow import keras
import tensorflow as tf


# Define the network class
class SemiStaticNet(keras.models.Sequential):
    def __init__(self, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), hidden_nodes=32):
        super().__init__()

        # Define the layers 
        self.add(layers.Dense(units=hidden_nodes, activation='relu', input_shape=(1,), kernel_initializer='random_uniform',
                              bias_initializer='random_uniform'))
        self.add(layers.Dense(units=1, activation='linear', kernel_initializer='random_uniform',
                              bias_initializer='random_uniform'))

        # Compile the semi_static_rep with the Adam optimizer and mean squared error (MSE) as the loss and metric
        self.compile(optimizer=optimizer, loss='mean_squared_error')
