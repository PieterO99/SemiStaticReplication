from keras import layers
from tensorflow import keras


# Define the network class
class SemiStaticNet(keras.models.Sequential):
    def __init__(self, optimizer):
        super().__init__()

        # Define the layers 
        self.add(layers.Dense(units=32, activation='relu', input_shape=(1,), kernel_initializer='random_uniform',
                              bias_initializer='random_uniform'))
        self.add(layers.Dense(units=1, activation='linear', kernel_initializer='random_uniform',
                              bias_initializer='random_uniform'))

        # Compile the semi_static_rep with the Adam optimizer and mean squared error (MSE) as the loss and metric
        self.compile(optimizer=optimizer, loss='mean_squared_error')

