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

def unpack_weights(weights_t):
    w_1 = np.array(weights_t[0]).reshape(-1)
    b_1 = np.array(weights_t[1])
    w_2 = np.array(weights_t[2]).reshape(-1) 
    b_2 = np.array(weights_t[3])
    return w_1, b_1, w_2, b_2

# %%
def binomial_pricer(S0, strike, T_m, rfr, vol, n, exercise_dates, pf_style):
    m = len(exercise_dates) - 1
    dim = n * m

    dt = T_m / dim
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(rfr * dt) - d) / (u - d)

    option_values = np.zeros((dim + 1, dim + 1))

    i_values = np.arange(dim + 1)
    option_values[dim] = payoff(S0 * (u ** i_values) * (d ** (dim - i_values)), strike, pf_style)

    for t in range(dim - 1, -1, -1):

        i_values = np.arange(t + 1)  # Array [0, 1, ..., t]
        hold_values = np.exp(-rfr * dt) * (
                p * option_values[t + 1, i_values + 1] + (1 - p) * option_values[t + 1, i_values])

        if (t % n) != 0:
            option_values[t, : t + 1] = hold_values

        else:
            option_values[t, : t + 1] = np.maximum(hold_values,
                                                   payoff(S0 * (u ** i_values) * (d ** (t - i_values)), strike,
                                                          pf_style))

    return option_values
