import numpy as np
from project_helpers import arithmetic_payoff, gen_paths_multivariate
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import laguerre, hermite, legendre
from sklearn.preprocessing import FunctionTransformer


# %%
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

    order = 2
    poly = PolynomialFeatures(degree=order)
    # legendre_transformer = FunctionTransformer(func=legendre_poly, kw_args={'order': order})
    scaler = StandardScaler()

    for m in range(num_mon - 1, -1, -1):
        stock = sample_paths[:, m, :]
        next_option = option[:, m + 1]

        scaled_stock = scaler.fit_transform(stock)
        current_stock = poly.fit_transform(scaled_stock)
        model = Ridge(alpha=1.)
        model.fit(current_stock, next_option)

        q = model.predict(current_stock) * np.exp(-rfr * time_increments[m])

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

    value_mat = np.zeros((sample_size, num_mon + 1), dtype=float)
    value_mat[:, num_mon] = arithmetic_payoff(strike, sample_paths[:, num_mon, :], weights, style)

    order = 3
    poly = PolynomialFeatures(degree=order)
    scaler = StandardScaler()

    for m in range(num_mon - 1, -1, -1):
        stock = sample_paths[:, m, :]
        h = arithmetic_payoff(strike, stock, weights, style)
        itm_mask_indices = np.where(h > 0)[0]

        if len(itm_mask_indices) > 0:
            disc_cash_flow = np.max(value_mat[itm_mask_indices, :] * np.exp(-rfr * cumulative_time_with_zero), axis=1)
            # log_stock = np.log(stock[itm_mask_indices])
            scaled_stock = scaler.fit_transform(stock[itm_mask_indices])
            current_stock = poly.fit_transform(scaled_stock)
            model = Ridge(alpha=1.)
            model.fit(current_stock, disc_cash_flow)

            q = model.predict(current_stock)

            exercise_or_continue = np.where(h[itm_mask_indices] > q)[0]
            indices_in_itm_mask = itm_mask_indices[exercise_or_continue]

            current_value = h[indices_in_itm_mask]

            value_mat[indices_in_itm_mask, :] = 0.
            value_mat[indices_in_itm_mask, m] = current_value

    value_mat = value_mat * np.exp(-rfr * cumulative_time_with_zero)
    positive_values = value_mat[value_mat > 0]
    return np.sum(positive_values) / sample_size


# %%
s = [1., 1., 1., 1., 1.]
correlation_matrix = [[1, 0.79, 0.82, 0.91, 0.84],
                      [0.79, 1., 0.73, 0.8, 0.76],
                      [0.82, 0.73, 1., 0.77, 0.72],
                      [0.91, 0.8, 0.77, 1., 0.9],
                      [0.84, 0.76, 0.72, 0.9, 1.]]
K = 1
sigmas = [0.518, 0.648, 0.623, 0.570, 0.53]
r = 0.05
w = [0.381, 0.065, 0.057, 0.27, 0.227]
num_samples = 80000
pf_style = 'put'
monitoring_dates = np.linspace(0, 1., 11)
# %%
s=[40]
K = 40
r = 0.06
w = [1]
correlation_matrix = [[1]]
sigmas = [0.2]
num_samples = 50000
pf_style = 'put'
monitoring_dates = np.linspace(0, 1., 11)
# %%
option_value = longstaff_schwartz_dyn_prog(s, correlation_matrix, K, sigmas, r, w, num_samples, monitoring_dates,
                                           pf_style)
print(f'Option value using dynamic programming approach: {option_value}')

option_value = longstaff_schwartz_opt_stopping(s, correlation_matrix, K, sigmas, r, w, num_samples, monitoring_dates,
                                               pf_style)
print(f'Option value using optimal stopping: {option_value}')
