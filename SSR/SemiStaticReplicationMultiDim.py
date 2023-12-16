import numpy as np
from scipy.stats import norm

from project_helpers import covariance_from_correlation, gen_paths_multivariate, arithmetic_payoff
from project_network import SemiStaticNetMultiDim

import keras
from keras.callbacks import EarlyStopping

from tqdm import tqdm


# %%

def continuation_q_log_multi_dim(current_stock, cov, vols, sample_size, asset_dim, w1, w2, b1, b2, dt):
    m = np.log(current_stock) + np.tile(((r - 0.5 * np.square(vols)) * dt).reshape(1, asset_dim),
                                        (sample_size, 1))
    opt_val = np.zeros((sample_size, 1))

    for node in range(no_of_hidden_nodes):
        w_o = w1[:, node]
        w_o = w_o.reshape(asset_dim, 1)
        mu = np.dot(m, w_o) + b1[node]
        var = np.dot(np.dot(w_o.T, cov), w_o)
        sd = var ** 0.5
        ft = mu * (1 - norm.cdf(-mu, 0, sd))
        st = (sd / (2 * np.pi) ** 0.5) * np.exp(-0.5 * (mu / sd) ** 2)
        opt_val = opt_val + w2[node] * (ft + st)

    q = (opt_val + b2) * np.exp(-r * dt)

    return q


def model_multi_dim(strike, initial_stock, rfr, vols, correlations, exercise_dates, option_weights, sample_size, style,
                    optimizer, hidd_nds, l1, l2):
    no_of_assets = len(initial_stock)
    no_of_exercise_days = len(exercise_dates)
    batch_size = int(no_of_paths / 10)
    mon_days = np.insert(exercise_dates, 0, 0)
    delta_t = exercise_dates[0]
    covariance = covariance_from_correlation(correlations, vols, delta_t)

    rlnn = SemiStaticNetMultiDim(optimizer=optimizer, hidden_nodes=hidd_nds, input_size=no_of_assets)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True,
                                   start_from_epoch=100)
    betas = []

    # pre-run
    rlnn.optimizer.learning_rate.assign(l1)
    stock_paths = gen_paths_multivariate(initial_stock, correlations, vols, rfr, sample_size, [0., exercise_days[-1]])
    stock_vec = stock_paths[:, 1]
    intrinsic_value = arithmetic_payoff(strike, stock_vec, option_weights, style)
    x_train = np.log(stock_vec)
    y_train = intrinsic_value
    y_train = y_train.reshape(-1, 1, 1)

    rlnn.fit(x_train, y_train, epochs=2000, batch_size=batch_size, verbose=0,
             validation_split=0.3, callbacks=[early_stopping])

    # main run
    rlnn.optimizer.learning_rate.assign(l2)
    stock_paths = gen_paths_multivariate(initial_stock, correlations, vols, rfr, sample_size, mon_days)
    continuation_value = np.zeros((no_of_paths, 1))

    for day in tqdm(range(no_of_exercise_days - 1, -1, -1)):
        stock_vec = stock_paths[:, day + 1]
        intrinsic_value = arithmetic_payoff(strike, stock_vec, option_weights, style)
        option_value = np.maximum(continuation_value, intrinsic_value)

        x_train = np.log(stock_vec)
        y_train = option_value
        y_train = y_train.reshape(-1, 1, 1)

        rlnn.fit(x_train, y_train, epochs=2000, batch_size=batch_size, verbose=0,
                 validation_split=0.2, callbacks=[early_stopping])

        w_1 = np.array(rlnn.layers[0].get_weights()[0])
        w_2 = np.array(rlnn.layers[1].get_weights()[0])
        b_1 = np.array(rlnn.layers[0].get_weights()[1])
        b_2 = np.array(rlnn.layers[1].get_weights()[1])
        betas.append(rlnn.get_weights())

        stock_vec = stock_paths[:, day]
        continuation_value = continuation_q_log_multi_dim(stock_vec, covariance, vols, sample_size, no_of_assets,
                                                          w_1, w_2, b_1, b_2, delta_t)

    option_value = continuation_value[0, 0]
    return betas, option_value


def upper_bound_multi_dim(trained_weights, stock_paths, strike, rfr, vols, correlations, option_weights,
                          monitoring, style, hidd_nds):
    sample_size, n_mon = len(stock_paths), len(monitoring)
    no_of_assets = len(stock_paths[0, 0])
    differences = np.diff(monitoring)

    covariance = covariance_from_correlation(correlations, vols, differences[0])

    b = np.exp(- rfr * np.cumsum(differences))
    b = np.insert(b, 0, 1)

    rlnn = SemiStaticNetMultiDim(hidden_nodes=hidd_nds, input_size=no_of_assets)

    martingale = np.zeros((sample_size, n_mon))

    for m in range(1, n_mon):
        stock_vec = stock_paths[:, m - 1, :]

        current_weights = trained_weights[- m]
        rlnn.set_weights(current_weights)
        w_1 = np.array(current_weights[0])
        b_1 = np.array(current_weights[1])
        w_2 = np.array(current_weights[2])
        b_2 = np.array(current_weights[3])

        q = continuation_q_log_multi_dim(stock_vec, covariance, vols, sample_size, no_of_assets,
                                         w_1, w_2, b_1, b_2, differences[m - 1])

        q_part = q.reshape(-1) * b[m - 1]
        g_part = (rlnn.predict(np.log(stock_paths[:, m, :]), verbose=0) * b[m]).reshape(-1)

        martingale[:, m] = (g_part - q_part)

    martingale = np.cumsum(martingale, axis=1)

    payoffs = arithmetic_payoff(strike, stock_paths, option_weights, style).reshape(sample_size, n_mon)
    upr = np.mean(np.max(payoffs * b - martingale, axis=1))

    return upr


def lower_bound_multi_dim(trained_weights, stock_paths, strike, rfr, vols, correlations, option_weights,
                          monitoring, style):
    sample_size, n_mon = len(stock_paths), len(monitoring)
    no_of_assets = len(stock_paths[0, 0])
    differences = np.diff(monitoring)

    covariance = covariance_from_correlation(correlations, vols, differences[0])

    tau = np.full(sample_size, n_mon - 1)
    h_of_s = arithmetic_payoff(strike, stock_paths[:, n_mon - 1, :], option_weights, style)

    for m in range(n_mon - 1):
        stock_vec = stock_paths[:, m, :]
        h = arithmetic_payoff(strike, stock_vec, option_weights, style)

        current_weights = trained_weights[n_mon - m - 2]
        w_1 = np.array(current_weights[0])
        b_1 = np.array(current_weights[1])
        w_2 = np.array(current_weights[2])
        b_2 = np.array(current_weights[3])

        q = continuation_q_log_multi_dim(stock_vec, covariance, vols, sample_size, no_of_assets,
                                         w_1, w_2, b_1, b_2, differences[m])

        exceed = np.logical_and(h.reshape(-1) > q.reshape(-1), tau > m)
        tau[exceed] = m
        h_of_s[exceed] = h[exceed]

    discounted_values = np.zeros(sample_size, dtype=float)
    h_of_s = h_of_s.reshape(-1)
    for j in range(sample_size):
        discounted_values[j] = h_of_s[j] * np.exp(-rfr * monitoring[tau[j]])

    lowr = np.mean(discounted_values)

    return lowr


# %%
no_of_paths = 1000
no_of_hidden_nodes = 10

s_0 = 1
no_of_stocks = 5
initial_stock_price = np.ones(no_of_stocks) * s_0
cor_mat = [[1.0, 0.79, 0.82, 0.91, 0.84],
           [0.79, 1.0, 0.73, 0.80, 0.76],
           [0.82, 0.73, 1.0, 0.77, 0.72],
           [0.91, 0.80, 0.77, 1.0, 0.90],
           [0.84, 0.76, 0.72, 0.90, 1.0]]
vol_list = np.array([0.518, 0.648, 0.623, 0.570, 0.530])
basket_weights = np.array([0.381, 0.065, 0.057, 0.270, 0.227])
basket_weights = basket_weights.reshape(-1, 1)
r = 0.05
K = 1
T = 1
M = 10
exercise_days = np.array(
    [float(i / M) for i in range(1, M + 1)])  # to be fixed, in the rest we use monitoring dates with 0

net_weights, v = model_multi_dim(K, initial_stock_price, r, vol_list, cor_mat, exercise_days, basket_weights,
                                 no_of_paths, 'put', keras.optimizers.Adam(), no_of_hidden_nodes,
                                 0.0005, 0.001)
print(v)
# %%
monitoring_dates = np.insert(exercise_days, 0, 0)
stock_bounds = gen_paths_multivariate(initial_stock_price, cor_mat, vol_list, r, no_of_paths, monitoring_dates)

low = lower_bound_multi_dim(net_weights, stock_bounds, K, r, vol_list, cor_mat, basket_weights, monitoring_dates, 'put')
print(low)

up = upper_bound_multi_dim(net_weights, stock_bounds, K, r, vol_list, cor_mat, basket_weights, monitoring_dates,
                           'put', no_of_hidden_nodes)

print(up)
