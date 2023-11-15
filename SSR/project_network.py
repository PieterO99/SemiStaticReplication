from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping


# Define the network class
class SemiStaticNet(keras.models.Sequential):
    def __init__(self, weights, optimizer):
        super().__init__()

        # Define the layers 
        self.add(layers.Dense(units=32, activation='relu', input_shape=(1,)))
        self.add(layers.Dense(units=1, activation='linear'))

        # Compile the semi_static_rep with the Adam optimizer and mean squared error (MSE) as the loss and metric
        self.compile(optimizer=optimizer, loss='mean_squared_error',
                     metrics=['mean_squared_error'])

        # Initialize the parameters with default weights
        self.initialize_parameters(weights)

    def initialize_parameters(self, custom_values=None):

        if custom_values is not None:
            if len(custom_values) != 2 * len(self.layers):
                raise ValueError("The number of custom values should match the number of layers in the model.")

            custom_weights_pairs = [custom_values[i:i + 2] for i in range(0, len(custom_values), 2)]

            for layer, custom_weights in zip(self.layers, custom_weights_pairs):
                layer.set_weights(custom_weights)
        else:
            self.layers[0].kernel_initializer = keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
            self.layers[1].kernel_initializer = keras.initializers.random_normal


# Build the backward recursion that fits the network for every interval between monitoring dates
def fitting(stock, option, weights=None, optimizer=keras.optimizers.Adam(learning_rate=0.001)):

    rlnn = SemiStaticNet(weights, optimizer)

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1, restore_best_weights=True)

    # Split the data into a training set and a validation set
    train_size = int(0.7 * len(stock))
    x_train, x_val = stock[:train_size], stock[train_size:]
    y_train, y_val = option[:train_size], option[train_size:]

    rlnn.fit(x_train, y_train, epochs=3000, batch_size=int(len(stock) / 10), verbose=0,
             validation_data=(x_val, y_val), callbacks=[early_stopping])

    final_weights = rlnn.get_weights()

    return final_weights


def pre_training(stock, option, optimizer, epochs, split, weights=None):

    rlnn = SemiStaticNet(weights, optimizer)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)

    rlnn.fit(stock, option, epochs=epochs, batch_size=int(len(stock) / 10), verbose=0,
             validation_split=split, callbacks=[early_stopping])

    return rlnn.get_weights()
