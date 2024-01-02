import csv
from datetime import datetime

import numpy as np

import keras
from keras.callbacks import EarlyStopping

from project_helpers import payoff, gen_paths, bs_put, bs_call
from project_network import SemiStaticNet

from model import continuation_q



# %%
def save_details_to_csv(filename, details):
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = details.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Check if the file is empty and write the header only if it is
        csvfile.seek(0, 2)  # Move the cursor to the end of the file
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow(details)


# %%
def continuation_q(w_1, b_1, w_2, b_2, stock, delta_t, rfr, vol, normalizing_constant):
    normalized_stock = stock / normalizing_constant

    cont = np.full_like(stock, b_2[0] * np.exp(-rfr * delta_t))

    mask1_indices = np.where((w_1 >= 0) & (b_1 >= 0))[0]
    mask2_indices = np.where((w_1 > 0) & (b_1 < 0))[0]
    mask3_indices = np.where((w_1 < 0) & (b_1 > 0))[0]

    strikes_call = -b_1[mask2_indices] / w_1[mask2_indices]
    strikes_put = -b_1[mask3_indices] / w_1[mask3_indices]

    cont += np.sum(w_2[mask1_indices] * (w_1[mask1_indices] *
                                         np.tile(normalized_stock[:, None], len(mask1_indices)) + b_1[
                                             mask1_indices] * np.exp(-rfr * delta_t)), axis=1)
    cont += np.sum(w_2[mask2_indices] * w_1[mask2_indices] * bs_call(normalized_stock, strikes_call, delta_t, rfr, vol),
                   axis=1)
    cont -= np.sum(w_2[mask3_indices] * w_1[mask3_indices] * bs_put(normalized_stock, strikes_put, delta_t, rfr, vol),
                   axis=1)

    return cont


def model(initial_stock, strike, rfr, vol, sample_size, mon_dates, style, optimizer, hidd_nds, l1, l2):
    """
    monitoring_dates: t0=0, ..., tM=T
    """

    details = {
        'run_datetime': str(datetime.now()),
        'S0': initial_stock,
        'K': strike,
        'mu': rfr,
        'sigma': vol,
        'N': sample_size,
        'monitoring_dates': mon_dates,
        'style': style,
        'hidden_nodes': hidd_nds,
        'optimizer': optimizer,
        'lr_pre_training': l1,
        'lr_training': l2,
        'initial_option_value': 0
    }

    betas = []  # store model weights
    time_increments = np.diff(mon_dates)
    num_mon = len(time_increments)

    rlnn = SemiStaticNet(optimizer, hidd_nds)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0, restore_best_weights=True,
                                   start_from_epoch=100)

    # first pre-run
    sample_pathsS = gen_paths(np.array([0., 1.]), initial_stock, rfr, vol, sample_size)
    option_pff = payoff(sample_pathsS[:, 1], strike, style)

    rlnn.optimizer.learning_rate.assign(l1)

    normalizer = np.mean(sample_pathsS[:, 1])
    rlnn.fit(sample_pathsS[:, 1] / normalizer, option_pff, epochs=1000,
             batch_size=int(sample_size / 10), verbose=0, validation_split=0.3, callbacks=[early_stopping])
    print('END OF FIRST PRE-RUN')

    normalizing_sequence = np.zeros(num_mon, dtype=float)

    # run the model
    sample_pathsS = gen_paths(mon_dates, initial_stock, rfr, vol, sample_size)

    option = np.zeros(sample_pathsS.shape)
    option[:, num_mon] = payoff(sample_pathsS[:, num_mon], strike, style)

    rlnn.optimizer.learning_rate.assign(l2)

    normalizer = np.mean(sample_pathsS[:, num_mon])
    normalizing_sequence[num_mon - 1] = normalizer
    rlnn.fit(sample_pathsS[:, num_mon] / normalizer, option[:, num_mon], epochs=3000,
             batch_size=int(sample_size / 10), verbose=0, validation_split=0.2, callbacks=[early_stopping])

    # Compute option value at T-1
    weights_layer_1 = np.array(rlnn.layers[0].get_weights()[0]).reshape(-1)
    biases_layer_1 = np.array(rlnn.layers[0].get_weights()[1])
    weights_layer_2 = np.array(rlnn.layers[1].get_weights()[0]).reshape(-1)
    biases_layer_2 = np.array(rlnn.layers[1].get_weights()[1])
    betas.append(rlnn.get_weights())

    q = continuation_q(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2,
                       sample_pathsS[:, num_mon - 1],
                       time_increments[num_mon - 1], rfr, vol, normalizer)
    h = payoff(sample_pathsS[:, num_mon - 1], strike, style)

    option[:, num_mon - 1] = np.maximum(h, q)

    # compute option values by backward regression
    for m in range(num_mon - 1, 0, -1):
        normalizer = np.mean(sample_pathsS[:, m])
        normalizing_sequence[m - 1] = normalizer
        rlnn.fit(sample_pathsS[:, m] / normalizer, option[:, m], epochs=3000, batch_size=int(sample_size / 10),
                 verbose=0, validation_split=0.2, callbacks=[early_stopping])

        # compute estimated option value one time step earlier
        weights_layer_1 = np.array(rlnn.layers[0].get_weights()[0]).reshape(-1)
        biases_layer_1 = np.array(rlnn.layers[0].get_weights()[1])
        weights_layer_2 = np.array(rlnn.layers[1].get_weights()[0]).reshape(-1)
        biases_layer_2 = np.array(rlnn.layers[1].get_weights()[1])
        betas.append(rlnn.get_weights())

        q = continuation_q(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2,
                           sample_pathsS[:, m - 1],
                           time_increments[m - 1], rfr, vol, normalizer)
        h = payoff(sample_pathsS[:, m - 1], strike, style)
        option[:, m - 1] = np.maximum(h, q)


    details['initial_option_value'] = option[0, 0]

    save_details_to_csv('run_details.csv', details)

    return betas, option, normalizing_sequence


# %%
def upper_bound(rfr, vol, trained_weights, normalizers, stock_paths, strike, monitoring, style, nodes_n):
    sample_size = len(stock_paths[:, 0])
    n_mon = len(monitoring)
    differences = np.diff(monitoring)

    b = np.exp(- rfr * np.cumsum(differences))
    b = np.insert(b, 0, 1)

    rlnn = SemiStaticNet(hidden_nodes=nodes_n)

    martingale = np.zeros((sample_size, n_mon))

    for m in range(1, n_mon):
        normalizer = normalizers[m - 1]
        current_weights = trained_weights[- m]
        rlnn.set_weights(current_weights)
        weights_layer_1 = np.array(current_weights[0]).reshape(-1)
        biases_layer_1 = np.array(current_weights[1])
        weights_layer_2 = np.array(current_weights[2]).reshape(-1)
        biases_layer_2 = np.array(current_weights[3])
        q = continuation_q(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2,
                           stock_paths[:, m - 1], differences[m - 1], rfr, vol, normalizer)

        q_part = q * b[m - 1]
        g_part = (rlnn.predict(stock_paths[:, m] / normalizer, verbose=0) * b[m]).reshape(-1)

        martingale[:, m] = (g_part - q_part)

    martingale = np.cumsum(martingale, axis=1)

    payoffs = payoff(stock_paths, strike, style)
    upr = np.mean(np.max(payoffs * b - martingale, axis=1))

    return upr


def lower_bound(rfr, vol, trained_weights, normalizers, stock_paths, strike, monitoring, style):
    sample_size, n_mon = len(stock_paths[:, 0]), len(monitoring)
    differences = np.diff(monitoring)

    tau = np.full(sample_size, n_mon - 1)
    h_of_s = payoff(stock_paths[:, n_mon - 1], strike, style)

    for m in range(n_mon - 1):
        normalizer = normalizers[m]
        s = stock_paths[:, m]  # stock values at time m
        h = payoff(s, strike, style)
        # weights is going to be the outcome of model(), so the weights relative to the first time interval are the
        # ones stored for last
        current_weights = trained_weights[n_mon - m - 2]
        weights_layer_1 = np.array(current_weights[0]).reshape(-1)
        biases_layer_1 = np.array(current_weights[1])
        weights_layer_2 = np.array(current_weights[2]).reshape(-1)
        biases_layer_2 = np.array(current_weights[3])
        q = continuation_q(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, s,
                           differences[m], rfr, vol, normalizer)
        exceed = np.logical_and(h > q, tau > m)
        tau[exceed] = m
        h_of_s[exceed] = h[exceed]

    discounted_values = np.zeros(sample_size, dtype=float)
    for j in range(sample_size):
        discounted_values[j] = h_of_s[j] * np.exp(-rfr * monitoring[tau[j]])

    lowr = np.mean(discounted_values)

    return lowr


# %%
S = 40  # Initial price
mu = 0.06  # Drift
sigma = 0.2  # Volatility
K_strike = 40  # Strike price
r = 0.06  # Risk-free rate
T = 1.  # Maturity
M = 10  # Number of monitoring dates
N = 50000  # Number of sample paths
nodes = 32
pf_style = 'put'  # Payoff type
monitoring_dates = np.linspace(0, T, M + 1)
