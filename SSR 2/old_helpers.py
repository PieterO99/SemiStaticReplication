from math import log, sqrt, pi, exp
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def stockprice_gbm(T, S0, mu, sigma, d=1):

    wiener_difference = np.random.normal(0, np.sqrt(T), d)
    return S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * wiener_difference)


def discretization_scheme(times, S0, mu, sigma, d=1):
    """
    times: t0=0, ... ,tM=T
    """

    stock_prices = np.zeros(len(times)+1)
    stock_prices[0] = S0
    deltas = np.diff(times)

    for i in range(len(deltas)):
        t_i = deltas[i]
        Si = stock_prices[i]
        Siplus1 = stockprice_gbm(t_i, Si, mu, sigma, d)
        stock_prices[i+1] = Siplus1

    return stock_prices


def gen_paths_old(monitoring_dates, S0, mu, sigma, N):
    """
    N: sample size
    monitoring_dates: t0=0, ... ,tM=T.
    """

    paths = np.empty((N, len(monitoring_dates)+1),dtype=float)
    for n in range(N):
        paths[n] = discretization_scheme(monitoring_dates, S0, mu, sigma)
    
    return paths
