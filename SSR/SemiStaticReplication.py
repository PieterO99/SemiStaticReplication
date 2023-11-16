import matplotlib.pyplot as plt
import numpy as np
from project_helpers import payoff
from project_helpers import gen_paths, forward, bs_put, bs_call
from project_network import SemiStaticNet, fitting, pre_training
import tensorflow as tf
import csv
from datetime import datetime

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
def continuation_q(trained_weights, stock, delta_t):  # normalizingC is normalizing constant from fitting t+1

    w_1 = trained_weights[0]  # first layer weights
    b_1 = trained_weights[1]  # first layer biases
    w_2 = trained_weights[2]  # second layer weights
    b_2 = trained_weights[3][0]  # second layer bias
    p = len(w_2)  # number of hidden nodes in the first hidden layer
    cont = []  # continuation vector

    for s_tm in stock:
        sum_cond_exp = b_2
        cond_exp = 0
        for i in range(p):
            w_i = w_1[0][i]
            b_i = b_1[i] + w_1[0][i]
            omega_i = w_2[i][0]

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
        cont.append(sum_cond_exp * np.exp(- r * delta_t))

    return cont


def model(S0, K, mu, sigma, N, monitoring_dates, style, optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
          model_weights=None):
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
        'l1': 0.001,
        'l2': 0.0005,
        'l3': optimizer.get_config()['learning_rate'],
        'e1': 2000,
        'e2': 2000
    }

    # generate sample paths of asset S
    sample_pathsS = gen_paths(monitoring_dates, S0, mu, sigma, N)

    # evaluate maturity time option values
    M = len(monitoring_dates) - 1
    option = np.zeros(sample_pathsS.shape)
    option[:, M] = payoff(sample_pathsS[:, M], K, style)

    # fit the model at T, store fitted weights to initialize next regression
    betas = []  # store model weights

    l1 = details['l1']
    l2 = details['l2']
    e1 = details['e1']
    e2 = details['e2']

    # double pre-training because why not
    first_pre_weights = pre_training(sample_pathsS[:, M], option[:, M],
                                     tf.keras.optimizers.legacy.Adam(learning_rate=l1), e1, 0.9, weights=model_weights)

    second_pre_weights = pre_training(sample_pathsS[:, M], option[:, M],
                                      tf.keras.optimizers.legacy.Adam(learning_rate=l2), e2, 0.8, weights=first_pre_weights)

    # now it gets serious
    fitted_beta = fitting(sample_pathsS[:, M], option[:, M],
                          weights=second_pre_weights, optimizer=optimizer)
    betas.append(fitted_beta)

    # Compute option value at T-1
    h = payoff(sample_pathsS[:, M - 1], K, style)  # value of exercising now
    time_increments = np.diff(monitoring_dates)
    q = continuation_q(fitted_beta, sample_pathsS[:, M - 1], time_increments[M - 1])  # continuation value
    option[:, M - 1] = np.maximum(h, q)  # take maximum of both values

    visualize_fit(fitted_beta, option[:, M], sample_pathsS[:, M], M, optimizer)

    # compute option values by backward regression
    for m in range(M - 1, 0, -1):
        # fit new weights, initialise with previous weights
        fitted_beta = fitting(sample_pathsS[:, m], option[:, m], weights=fitted_beta, optimizer=optimizer)

        # compute estimated option value one time step earlier
        h = payoff(sample_pathsS[:, m - 1], K, style)  # value of exercising now
        q = continuation_q(fitted_beta, sample_pathsS[:, m - 1], time_increments[m - 1])  # continuation value
        option[:, m - 1] = np.maximum(h, q)  # take maximum of both values

        visualize_fit(fitted_beta, option[:, m], sample_pathsS[:, m], m, optimizer)

        # append the model weights to the beta list
        betas.append(fitted_beta)

    details['initial_option_value'] = option[0, 0]
    # Save details to a CSV file
    save_details_to_csv('run_details.csv', details)

    return betas, option


# %%
def visualize_fit(trained_weights, option_values, stock_values, time, optimizer):
    rlnn = SemiStaticNet(None, optimizer)

    rlnn.initialize_parameters(trained_weights)
    predicted_values = np.array(rlnn.predict(stock_values))

    plt.figure()
    plt.scatter(stock_values, option_values, label=f'Option value at {time}-th monitoring date', color='b', s=0.4)
    plt.scatter(stock_values, predicted_values.reshape(-1), label=f'Regressed option value at {time}-th monitoring date',
             color='r', s=0.4)
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
K = 40  # Strike price
r = 0.06  # Risk-free rate
T = 1  # Maturity
M = 10  # Number of monitoring dates
N = 20000  # Number of sample paths
pf_style = 'put'  # Payoff type
monitoring_dates = np.linspace(0, T, M + 1)
# %%
weights, option_value = model(S, K, mu, sigma, N, monitoring_dates, pf_style,
                              optimizer=tf.keras.optimizers.legacy.Adamax(learning_rate=0.001))
v_0 = option_value[:, 0][0]
print(v_0)

