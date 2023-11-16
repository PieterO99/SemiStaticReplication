import csv
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
import keras.backend as KB

from project_helpers import payoff, gen_paths, forward, bs_put, bs_call
from project_network import SemiStaticNet


# Defines a lower triangular matrix describing the stock dynamics
def binomial_tree_stock(S, T, sigma, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    stock_tree = np.zeros((n + 1, n + 1))
    stock_tree[0, 0] = S

    for i in range(1, n + 1):
        stock_tree[i, i] = stock_tree[i - 1, i - 1] * u
        for j in range(i, n + 1):
            stock_tree[j, i - 1] = stock_tree[j - 1, i - 1] * d

    return stock_tree


# Returns a lower triangular matrix describing the Bermudan option price dynamics
def binomial_bermudan_option_pricing(S, K, T, r, sigma, n, early_exercise_dates, style):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize option value matrix
    option_values = np.zeros((n + 1, n + 1))

    # Calculate the option values at expiration
    for i in range(n + 1):
        option_values[n][i] = payoff(S * (u ** i) * (d ** (n - i)), K, style)

    # Calculate the option values at earlier exercise dates
    for t in range(n - 1, -1, -1):
        for i in range(t + 1):
            hold_value = np.exp(-r * dt) * (p * option_values[t + 1][i + 1] + (1 - p) * option_values[t + 1][i])
            if round(t * dt, 2) in early_exercise_dates:
                option_values[t][i] = max(payoff(S * (u ** i) * (d ** (t - i)), K, style), hold_value)
            else:
                option_values[t][i] = hold_value

    return option_values


S0 = 40  # Initial stock price
K = 40  # Strike price
T = 1.0  # Time to maturity
r = 0.06  # Risk-free interest rate
sigma = 0.2  # Volatility
n = 100  # Number of time steps in the binomial tree
early_exercise_dates = np.linspace(0, 1, 11)  # List of early exercise dates
bermudan_option_prices = binomial_bermudan_option_pricing(S0, K, T, r, sigma, n, early_exercise_dates, 'put')
monitored_prices = [bermudan_option_prices[int(k * n)][:] for k in early_exercise_dates]
monitored_stock = [binomial_tree_stock(S0, T, sigma, n)[int(k * n)][:] for k in early_exercise_dates]


def save_details_to_csv(filename, details):
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = details.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if the file is empty and write the header only if it is
        csvfile.seek(0, 2)  # Move the cursor to the end of the file
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(details)


# Compute the continuation value Q for the N paths at time t_{m-1}
def continuation_q(w_1, b_1, w_2, b_2, stock, delta_t, norm):  # norm is normalizing constant from fitting t+1

    p = len(w_2)  # number of hidden nodes in the first hidden layer
    cont = np.zeros(len(stock))  # continuation vector

    normalized_stock = stock  # / norm

    for j in range(len(stock)):

        s_tm = normalized_stock[j]
        sum_cond_exp = b_2[0]
        cond_exp = 0

        for i in range(p):
            w_i = w_1[i]
            b_i = b_1[i]
            omega_i = w_2[i]

            if w_i >= 0 and b_i >= 0:
                cond_exp = w_i * forward(s_tm, r, delta_t) + b_i
            elif w_i > 0 > b_i:
                cond_exp = w_i * bs_call(s_tm, -b_i / w_i, delta_t, r, sigma)
            elif w_i < 0 < b_i:
                cond_exp = - w_i * bs_put(s_tm, -b_i / w_i, delta_t, r, sigma)
            elif w_i <= 0 and b_i <= 0:
                cond_exp = 0

            sum_cond_exp += omega_i * cond_exp

        # Discount by the risk-free rate
        cont[j] = sum_cond_exp * np.exp(- r * delta_t)

    return cont


def pre_training(nnet, early_stopping, sample_paths, option, time_increments, batch, n_epochs, val_ratio, strike,
                 style):
    M = len(option[0]) - 1
    nnet.fit(sample_paths[:, M], option[:, M], epochs=n_epochs, batch_size=batch, verbose=0,
             validation_split=val_ratio, callbacks=[early_stopping])

    # Compute option value at T-1
    weights_layer_1 = np.array(nnet.layers[0].get_weights()[0]).reshape(-1)
    biases_layer_1 = np.array(nnet.layers[0].get_weights()[1])
    weights_layer_2 = np.array(nnet.layers[1].get_weights()[0]).reshape(-1)
    biases_layer_2 = np.array(nnet.layers[1].get_weights()[1])

    q = continuation_q(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, sample_paths[:, M - 1],
                       time_increments[M - 1], np.max(sample_paths[:, M]))  # continuation value
    h = payoff(sample_paths[:, M - 1], strike, style)  # value of exercising now

    option[:, M - 1] = np.maximum(h, q)  # take maximum of both values

    # compute option values by backward regression
    for m in range(M - 1, 0, -1):
        nnet.fit(sample_paths[:, m], option[:, m], epochs=n_epochs, batch_size=batch, verbose=0,
                 validation_split=val_ratio, callbacks=[early_stopping])

        # compute estimated option value one time step earlier
        weights_layer_1 = np.array(nnet.layers[0].get_weights()[0]).reshape(-1)
        biases_layer_1 = np.array(nnet.layers[0].get_weights()[1])
        weights_layer_2 = np.array(nnet.layers[1].get_weights()[0]).reshape(-1)
        biases_layer_2 = np.array(nnet.layers[1].get_weights()[1])

        q = continuation_q(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, sample_paths[:, m - 1],
                           time_increments[m - 1], np.max(sample_paths[:, m]))  # continuation value
        h = payoff(sample_paths[:, m - 1], strike, style)  # value of exercising now
        option[:, m - 1] = np.maximum(h, q)  # take maximum of both values

    return nnet


def model(S0, K, mu, sigma, N, monitoring_dates, style, optimizer):
    """
    monitoring_dates: t0=0, ..., tM=T
    """
    details = {
        'run_datetime': str(datetime.now()),
        'S0': S0,
        'K': K,
        'mu': mu,
        'sigma': sigma,
        'N': N,
        'monitoring_dates': monitoring_dates,
        'style': style,
        'l1': 0.01,
        'l2': 0.0005,
        'l3': optimizer.get_config()['learning_rate'],
        'e1': 1500,
        'e2': 2000
    }

    l1 = details['l1']
    l2 = details['l2']
    e1 = details['e1']
    e2 = details['e2']

    betas = []  # store model weights
    time_increments = np.diff(monitoring_dates)
    M = len(monitoring_dates) - 1

    rlnn = SemiStaticNet(optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0)

    # first pre-run
    sample_pathsS = gen_paths(monitoring_dates, S0, mu, sigma, N)
    option = np.zeros(sample_pathsS.shape)
    option[:, M] = payoff(sample_pathsS[:, M], K, style)
    batch = int(N / 20)
    n_epochs = e1
    val_ratio = 0.2
    KB.set_value(rlnn.optimizer.lr, l1)
    rlnn = pre_training(rlnn, early_stopping, sample_pathsS, option, time_increments, batch,
                        n_epochs, val_ratio, K, style)

    # second pre-run
    sample_pathsS = gen_paths(monitoring_dates, S0, mu, sigma, N)
    option = np.zeros(sample_pathsS.shape)
    option[:, M] = payoff(sample_pathsS[:, M], K, style)
    batch = int(N / 10)
    n_epochs = e2
    val_ratio = 0.2
    KB.set_value(rlnn.optimizer.lr, l2)
    rlnn = pre_training(rlnn, early_stopping, sample_pathsS, option, time_increments, batch,
                        n_epochs, val_ratio, K, style)

    # run the model
    sample_pathsS = gen_paths(monitoring_dates, S0, mu, sigma, N)
    # evaluate maturity time option values
    M = len(monitoring_dates) - 1
    option = np.zeros(sample_pathsS.shape)
    option[:, M] = payoff(sample_pathsS[:, M], K, style)

    KB.set_value(rlnn.optimizer.lr, 0.001)
    rlnn.fit(sample_pathsS[:, M], option[:, M], epochs=3000, batch_size=int(N / 10), verbose=0,
             validation_split=0.3, callbacks=[early_stopping])

    # Compute option value at T-1
    weights_layer_1 = np.array(rlnn.layers[0].get_weights()[0]).reshape(-1)
    biases_layer_1 = np.array(rlnn.layers[0].get_weights()[1])
    weights_layer_2 = np.array(rlnn.layers[1].get_weights()[0]).reshape(-1)
    biases_layer_2 = np.array(rlnn.layers[1].get_weights()[1])
    betas.append([weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2])

    q = continuation_q(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, sample_pathsS[:, M - 1],
                       time_increments[M - 1], np.max(sample_pathsS[:, M]))  # continuation value
    h = payoff(sample_pathsS[:, M - 1], K, style)  # value of exercising now

    option[:, M - 1] = np.maximum(h, q)  # take maximum of both values

    predictions = np.array(rlnn.predict(sample_pathsS[:, M])).reshape(-1)
    visualize_fit(predictions, option[:, M], sample_pathsS[:, M], M)

    # compute option values by backward regression
    for m in range(M - 1, 0, -1):

        rlnn.fit(sample_pathsS[:, m], option[:, m], epochs=3000, batch_size=int(N / 10), verbose=0,
                 validation_split=0.3, callbacks=[early_stopping])
        fitted_beta = rlnn.get_weights()

        # compute estimated option value one time step earlier
        weights_layer_1 = np.array(rlnn.layers[0].get_weights()[0]).reshape(-1)
        biases_layer_1 = np.array(rlnn.layers[0].get_weights()[1])
        weights_layer_2 = np.array(rlnn.layers[1].get_weights()[0]).reshape(-1)
        biases_layer_2 = np.array(rlnn.layers[1].get_weights()[1])
        betas.append([weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2])

        q = continuation_q(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, sample_pathsS[:, m - 1],
                           time_increments[m - 1], np.max(sample_pathsS[:, m]))  # continuation value
        h = payoff(sample_pathsS[:, m - 1], K, style)  # value of exercising now
        option[:, m - 1] = np.maximum(h, q)  # take maximum of both values

        predictions = np.array(rlnn.predict(sample_pathsS[:, m])).reshape(-1)
        visualize_fit(predictions, option[:, m], sample_pathsS[:, m], m)

        # append the model weights to the beta list
        betas.append(fitted_beta)

    details['initial_option_value'] = option[0, 0] * S0
    # Save details to a CSV file
    save_details_to_csv('run_details.csv', details)

    return betas, option


# %%
def visualize_fit(predictions, option_values, stock_values, time):
    plt.figure()

    plt.scatter(stock_values * S, option_values * S, label=f'Option value at {time}-th monitoring date via RLNN',
                color='b', s=0.4)

    plt.scatter(stock_values * S, predictions * S,
                label=f'Regressed option value at {time}-th monitoring date', color='r', s=0.4)

    plt.scatter(monitored_stock[time], monitored_prices[time],
                label=f'Option value at {time}-th monitoring date via Binomial Model', color='m', s=0.4)

    plt.xlabel('Stock Value')
    plt.ylabel('Option Value')
    plt.legend()
    plt.show()


# %%
# Simulation parameters initialization
# The asset prices follow a Geometric Brownian Motion
S = 40  # Initial price
mu = 0.06  # Drift
sigma = 0.2  # Volatility
K_strike = 40  # Strike price
r = 0.06  # Risk-free rate
T = 1  # Maturity
M = 10  # Number of monitoring dates
N = 20000  # Number of sample paths
pf_style = 'put'  # Payoff type
monitoring_dates = np.linspace(0, T, M + 1)
# %%
weights, option_value = model(S / S, K_strike / S, mu, sigma, N, monitoring_dates, pf_style,
                              optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001))
v_0 = option_value[:, 0][0]
print(S * v_0)
