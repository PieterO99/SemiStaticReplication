import numpy as np
from scipy.stats import norm


def payoff(S0, K, style):
    if style == 'call':
        return np.maximum(S0 - K, np.zeros_like(S0))
    elif style == 'put':
        return np.maximum(K - S0, np.zeros_like(S0))
    else:
        raise ValueError("Invalid option style. Style must be 'call' or 'put'.")


def arithmetic_payoff(strike, spot, weights, style):
    weighted_sum = np.dot(spot, weights)
    return payoff(weighted_sum, strike, style)


def d1(S0, K, T, r, sigma):
    denominator = sigma * np.sqrt(T)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return (np.log(S0 / K) + (r + sigma ** 2 / 2.) * T) / denominator


def d2(S0, K, T, r, sigma):
    return d1(S0, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call(S0, K, T, r, sigma):
    S0 = S0[:, np.newaxis]
    K = K[np.newaxis, :]

    d1_values = d1(S0, K, T, r, sigma)
    d2_values = d2(S0, K, T, r, sigma)

    call_values = S0 * norm.cdf(d1_values) - K * np.exp(-r * T) * norm.cdf(d2_values)

    return call_values


def bs_put(S0, K, T, r, sigma):
    S0 = S0[:, np.newaxis]
    K = K[np.newaxis, :]

    d1_values = d1(S0, K, T, r, sigma)
    d2_values = d2(S0, K, T, r, sigma)

    put_values = norm.cdf(-d2_values) * K * np.exp(-r * T) - norm.cdf(-d1_values) * S0

    return put_values


def gen_paths(monitoring_dates, S0, mu, sigma, N, antithetic=False):
    """
    N: sample size
    """
    delta_t = np.diff(monitoring_dates)
    num_intervals = len(delta_t)
    rng = np.random.default_rng()
    if antithetic:
        gaussian_samples = rng.normal(0, 1, (N // 2, num_intervals))
        normal_samples = np.concatenate([gaussian_samples, -gaussian_samples])
    else:
        normal_samples = rng.normal(0, 1, (N, num_intervals))
    exp_term = (mu - 0.5 * sigma ** 2) * delta_t
    vol_term = sigma * np.sqrt(delta_t)

    cum_sum_exp = np.cumsum(exp_term)
    cum_sum_vol = [np.cumsum(vol_term * normal_samples[i]) for i in range(N)]

    stock_prices = np.zeros((N, num_intervals + 1))
    stock_prices[:, 0] = S0

    for i in range(N):
        stock_prices[i, 1:] = S0 * np.exp(cum_sum_exp + cum_sum_vol[i])

    return stock_prices


def covariance_from_correlation(cor_mat, vols, dt):
    vol_diag_mat = np.diag(vols)
    cov_mat = np.dot(np.dot(vol_diag_mat, cor_mat), vol_diag_mat) * dt
    return cov_mat


def gen_paths_multivariate(initial_stocks, cor, vols, rfr, sample_size, mon_dates, antithetic=False):
    zero_mean = np.zeros(len(initial_stocks))
    dt = mon_dates[-1] / (len(mon_dates) - 1)
    cov = covariance_from_correlation(cor, vols, dt)

    # antithetic sampling
    if antithetic:
        gaussian_samples = np.random.multivariate_normal(zero_mean, cov, (sample_size // 2, len(mon_dates) - 1))
        dw_mat = np.concatenate([gaussian_samples, -gaussian_samples])
    else:
        dw_mat = np.random.multivariate_normal(zero_mean, cov, (sample_size, len(mon_dates) - 1))

    sim_ln_stock = np.zeros((sample_size, len(mon_dates), len(initial_stocks)))
    sim_ln_stock[:, 0] = np.tile(np.log(initial_stocks), (sample_size, 1))
    base_drift = np.tile((np.add(np.full(len(initial_stocks), rfr), - 0.5 * np.square(vols))), (sample_size, 1)) * dt

    for day in range(1, len(mon_dates)):
        curr_drift = sim_ln_stock[:, day - 1] + base_drift
        sim_ln_stock[:, day] = curr_drift + dw_mat[:, day - 1]

    sim_stock_mat = np.exp(sim_ln_stock)
    return sim_stock_mat


def gen_paths_heston(initial_stock, initial_vol, rfr, mr_speed, mr_mean, vol_vol, cov, sample_size, mon_dates):
    differences = np.diff(mon_dates)
    n_mon = len(mon_dates)
    paths = np.zeros((sample_size, len(mon_dates), 2), dtype=float)
    initial_conditions = np.array([initial_stock, initial_vol])
    paths[:, 0, :] = initial_conditions.reshape(1, 2)

    cov = cov # * differences[0]
    gaussian_samples = np.random.multivariate_normal(np.array([0, 0]), cov, (sample_size, len(mon_dates) - 1))

    s = paths[:, 0, 0]
    v = paths[:, 0, 1]

    for i in range(1, n_mon):
        s = s * np.exp((rfr - 0.5 * v) * differences[i - 1] + np.sqrt(v * differences[i - 1])
                       * gaussian_samples[:, i - 1, 0])
        v = np.maximum(v + mr_speed * (mr_mean - v) * differences[i - 1] + vol_vol * np.sqrt(v * differences[i - 1])
                       * gaussian_samples[:, i - 1, 0], 0)
        paths[:, i, :] = np.column_stack((s, v))

    return paths
