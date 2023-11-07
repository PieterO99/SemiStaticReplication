import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm


def stockprice_gbm(t, s, mu, sigma, d=1):
    wiener_difference = np.random.normal(0, t, d)
    return s * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * wiener_difference)


def discretization_scheme(times, s, mu, sigma, d=1):
    stock_prices = np.array([s])
    deltas = [0]

    for time in times:
        deltas.append(time - deltas[-1])
        current = stock_prices[-1]
        new = stockprice_gbm(deltas[-1], current, mu, sigma, d)
        stock_prices = np.concatenate((stock_prices, new))

    return stock_prices


def gen_paths(monitoring_dates, s, mu, sigma, sample_size):
    p = []
    for n in range(sample_size):
        path = discretization_scheme(monitoring_dates, s, mu, sigma)
        p.append(path)
    paths = np.array(p)
    return paths


def payoff(s, k, style):
    if style == 'call':
        return max(s - k, 0)
    elif style == 'put':
        return max(k - s, 0)
    else:
        raise ValueError("Invalid option style. Style must be 'call' or 'put'.")


def forward(s, r, t):
    return s * np.exp(r * t)


def d1(s, k, t, r, sigma):
    return (log(s / k) + (r + sigma ** 2 / 2.) * t) / (sigma * sqrt(t))


def d2(s, k, t, r, sigma):
    return d1(s, k, t, r, sigma) - sigma * sqrt(t)


def bs_call(s, k, t, r, sigma):
    return s * norm.cdf(d1(s, k, t, r, sigma)) - k * exp(-r * t) * norm.cdf(d2(s, k, t, r, sigma))


def bs_put(s, k, t, r, sigma):
    return norm.cdf(-d2(s, k, t, r, sigma)) * k * np.exp(-r * t) - norm.cdf(-d1(s, k, t, r, sigma)) * s
