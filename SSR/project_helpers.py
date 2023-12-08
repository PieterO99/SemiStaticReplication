import numpy as np
from scipy.stats import norm


def payoff(S0, K, style):
    if style == 'call':
        return np.maximum(S0 - K, np.zeros_like(S0))
    elif style == 'put':
        return np.maximum(K - S0, np.zeros_like(S0))
    else:
        raise ValueError("Invalid option style. Style must be 'call' or 'put'.")


def forward(S0):  # r, T
    return S0  # * np.exp(r * T)


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


def gen_paths(monitoring_dates, S0, mu, sigma, N):
    """
    N: sample size
    """

    delta_t = np.diff(monitoring_dates)
    num_intervals = len(delta_t)
    rng = np.random.default_rng()
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
