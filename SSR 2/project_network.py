from keras import layers
import keras


class SemiStaticNet(keras.models.Sequential):
    def __init__(self, optimizer=keras.optimizers.Adam(learning_rate=0.001), hidden_nodes=32):  # , dropout_rate=0.0
        super().__init__()

        self.add(keras.Input(shape=(1,)))
        # Define the layers 
        self.add(
            layers.Dense(units=hidden_nodes, activation='relu', kernel_initializer='random_uniform',
                         bias_initializer='random_uniform'))
        # self.add(layers.Dropout(dropout_rate))
        self.add(layers.Dense(units=1, activation='linear', kernel_initializer='random_uniform',
                              bias_initializer='random_uniform'))

        self.compile(optimizer=optimizer, loss='mean_squared_error')


class SemiStaticNetMultiDim(keras.models.Sequential):
    def __init__(self, optimizer=keras.optimizers.Adam(learning_rate=0.001), hidden_nodes=32, input_size=2):
        super().__init__()

        self.add(keras.Input(shape=(input_size,)))

        self.add(
            layers.Dense(units=hidden_nodes, activation='relu', kernel_initializer='random_uniform',
                         bias_initializer='random_uniform'))
        self.add(layers.Dense(units=1, activation='linear', kernel_initializer='random_uniform',
                              bias_initializer='random_uniform'))

        self.compile(optimizer=optimizer, loss='mean_squared_error')
