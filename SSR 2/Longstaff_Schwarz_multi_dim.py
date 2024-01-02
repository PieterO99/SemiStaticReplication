import numpy as np
from project_helpers import payoff, bs_put
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import laguerre, hermite, legendre
from sklearn.preprocessing import FunctionTransformer


def covariance_from_correlation(cor_mat, vols, dt):
    vol_diag_mat = np.diag(vols)
    cov_mat = np.dot(np.dot(vol_diag_mat, cor_mat), vol_diag_mat) * dt
    return cov_mat


def gen_paths_multivariate(initial_stocks, cor, vols, rfr, sample_size, mon_dates):
    zero_mean = np.zeros(len(initial_stocks))
    dt = mon_dates[-1] / (len(mon_dates) - 1)
    cov = covariance_from_correlation(cor, vols, dt)

    # antithetic sampling
    gaussian_samples = np.random.multivariate_normal(zero_mean, cov, (sample_size // 2, len(mon_dates) - 1))
    dw_mat = np.concatenate([gaussian_samples, -gaussian_samples])

    sim_ln_stock = np.zeros((sample_size, len(mon_dates), len(initial_stocks)))
    sim_ln_stock[:, 0] = np.tile(np.log(initial_stocks), (sample_size, 1))
    base_drift = np.tile((np.add(np.full(len(initial_stocks), rfr), - 0.5 * np.square(vols))), (sample_size, 1)) * dt

    for day in range(1, len(mon_dates)):
        curr_drift = sim_ln_stock[:, day - 1] + base_drift
        sim_ln_stock[:, day] = curr_drift + dw_mat[:, day - 1]

    sim_stock_mat = np.exp(sim_ln_stock)
    return sim_stock_mat


def arithmetic_payoff(strike, spot, weights, style):
    weighted_sum = np.sum(weights * spot, axis=1)
    return payoff(weighted_sum, strike, style)


def laguerre_poly(X, order):
    return laguerre(order)(X)


def hermite_poly(X, order):
    return hermite(order)(X / np.sqrt(2))


def legendre_poly(X, order):
    return legendre(order)(X)


def longstaff_schwartz_dyn_prog(initial_stocks, correlation, strike, vols, rfr, weights, sample_size, mon_dates, style):
    time_increments = np.diff(mon_dates)
    num_mon = len(time_increments)

    sample_paths = gen_paths_multivariate(initial_stocks, correlation, vols, rfr, sample_size, mon_dates)

    option = np.zeros((sample_size, num_mon + 1), dtype=float)
    option[:, num_mon] = arithmetic_payoff(strike, sample_paths[:, num_mon, :], weights, style)

    order = 4
    poly = PolynomialFeatures(degree=order)
    # legendre_transformer = FunctionTransformer(func=legendre_poly, kw_args={'order': order})
    scaler = StandardScaler()

    for m in range(num_mon - 1, -1, -1):
        stock = sample_paths[:, m, :]
        next_option = option[:, m + 1]

        current_stock = poly.fit_transform(stock)
        current_stock_scaled = scaler.fit_transform(current_stock)
        model = Ridge(alpha=10.)
        model.fit(current_stock_scaled, next_option)

        current_stock_predict = poly.transform(stock)
        current_stock_predict_scaled = scaler.transform(current_stock_predict)
        q = model.predict(current_stock_predict_scaled) * np.exp(-rfr * time_increments[m])

        h = arithmetic_payoff(strike, sample_paths[:, m, :], weights, style)
        # assign to the option value the max between continuation and exercising value
        option[:, m] = np.maximum(q, h)

    return option[0, 0]


def longstaff_schwartz_opt_stopping(initial_stocks, correlation, strike, vols, rfr, weights, sample_size, mon_dates,
                                    style):
    time_increments = np.diff(mon_dates)
    num_mon = len(time_increments)
    cumulative_time = np.cumsum(time_increments)
    cumulative_time_with_zero = np.insert(cumulative_time, 0, 0)

    sample_paths = gen_paths_multivariate(initial_stocks, correlation, vols, rfr, sample_size, mon_dates)

    # stopping_rule = np.zeros((sample_size, num_mon + 1))
    # stopping_rule[:, num_mon] = np.ones(sample_size)
    value_mat = np.zeros((sample_size, num_mon + 1), dtype=float)
    value_mat[:, num_mon] = arithmetic_payoff(strike, sample_paths[:, num_mon, :], weights, style)

    order = 20
    poly = PolynomialFeatures(degree=order)
    scaler = StandardScaler()

    for m in range(num_mon - 1, -1, -1):
        stock = sample_paths[:, m, :]
        h = arithmetic_payoff(strike, stock, weights, style)
        itm_mask_indices = np.where(h > 0)[0]

        if len(itm_mask_indices) > 0:
            disc_cash_flow = np.max(value_mat[itm_mask_indices, :] * np.exp(-rfr * cumulative_time_with_zero), axis=1)
            current_stock = poly.fit_transform(stock[itm_mask_indices])
            current_stock_scaled = scaler.fit_transform(current_stock)
            model = Ridge(alpha=10.)
            model.fit(current_stock_scaled, disc_cash_flow)

            current_stock_predict = poly.transform(stock[itm_mask_indices])
            current_stock_predict_scaled = scaler.transform(current_stock_predict)
            q = model.predict(current_stock_predict_scaled)
            exercise_or_continue = np.where(h[itm_mask_indices] > q)[0]
            current_value = np.zeros(sample_size, dtype=float)
            current_value[exercise_or_continue] = h[exercise_or_continue]

            new_optimal_mask = np.all(current_value[:, np.newaxis] > value_mat[:, m + 1:], axis=1)

            # stopping_rule[new_optimal_mask, :] = 0
            # stopping_rule[new_optimal_mask, m] = 1
            value_mat[new_optimal_mask, :] = 0.
            value_mat[new_optimal_mask, m] = current_value[new_optimal_mask]

    value_mat = value_mat * np.exp(-rfr * cumulative_time_with_zero)
    positive_values = value_mat[value_mat > 0]
    return np.sum(positive_values) / sample_size


s = [1., 1., 1., 1., 1.]
# s=[40]
correlation_matrix = [[1, 0.79, 0.82, 0.91, 0.84],
                      [0.79, 1., 0.73, 0.8, 0.76],
                      [0.82, 0.73, 1., 0.77, 0.72],
                      [0.91, 0.8, 0.77, 1., 0.9],
                      [0.84, 0.76, 0.72, 0.9, 1.]]
# correlation_matrix = [[1]]
K = 1
sigmas = [0.518, 0.648, 0.623, 0.570, 0.53]
# sigmas=[0.2]
r = 0.05
w = [0.381, 0.065, 0.057, 0.27, 0.227]
# w=[1]
num_samples = 1000000
pf_style = 'put'
monitoring_dates = np.linspace(0, 1., 11)


#s=[40]
#K = 40
#r = 0.06
#w = [1]
#correlation_matrix = [[1]]
#sigmas = [0.2]
#monitoring_dates = np.linspace(0, 1., 2)

option_value = longstaff_schwartz_dyn_prog(s, correlation_matrix, K, sigmas, r, w, num_samples, monitoring_dates,
                                           pf_style)
print(f'Option value using dynamic programming approach: {option_value}')

option_value = longstaff_schwartz_opt_stopping(s, correlation_matrix, K, sigmas, r, w, num_samples, monitoring_dates,
                                               pf_style)
print(f'Option value using optimal stopping: {option_value}')
