from cProfile import label
import numpy as np
from project_helpers import binomial_pricer, d1, unpack_weights
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from SemiStaticReplication import model
import keras
from tqdm import tqdm
import csv
import time


def delta_nn(S_t, normalizer_t, delta_t, r, sigma, weights_t):
    w_1, b_1, w_2, _ = unpack_weights(weights_t)
    mask1_indices = np.where((w_1 >= 0) & (b_1 >= 0))[0] # delta = w_i
    mask2_indices = np.where((w_1 > 0) & (b_1 < 0))[0]   # delta = w_i (N(d1)-1)
    mask3_indices = np.where((w_1 < 0) & (b_1 > 0))[0]   # delta = -w_i N(d_1)
    d1_vec = d1(S_t / normalizer_t, np.abs(b_1 / w_1), delta_t, r, sigma) # weights correspond to normalized stock

    # compute delta wrt normalized stock
    delta = np.sum( w_2[mask1_indices] * w_1[mask1_indices] ) 
    delta += np.sum( w_2[mask2_indices] * w_1[mask2_indices] * norm.cdf(d1_vec[mask2_indices]) )
    delta += np.sum( w_2[mask3_indices] * w_1[mask3_indices] * norm.cdf(-d1_vec[mask3_indices]) )
    return delta / normalizer_t

def delta_binomial(bump, S_t, strike, delta_t, rfr, vol, n, exercise_dates, pf_style):
    v0 = binomial_pricer(S_t, strike, delta_t, rfr, vol, n, exercise_dates, pf_style)[0,0]
    v0_bumped = binomial_pricer(S_t + bump, strike, delta_t, rfr, vol, n, exercise_dates, pf_style)[0,0]
    return (v0_bumped - v0) / bump

S = 44  # Initial stock price
K = 40  # Strike price
T = 1.0  # Time to maturity
r = 0.06  # Risk-free interest rate
sigma = 0.2  # Volatility
N = 50000
nodes = 64
M = 10
monitoring_dates = np.linspace(0, T, M + 1)
style = 'put'
weights36, option_value, normalizers = model(36, K, r, sigma, N, monitoring_dates, style,
                                           keras.optimizers.Adam(), nodes, 0.001, 0.001)
weights40, option_value, normalizers = model(40, K, r, sigma, N, monitoring_dates, style,
                                           keras.optimizers.Adam(), nodes, 0.001, 0.001)
weights44, option_value, normalizers = model(44, K, r, sigma, N, monitoring_dates, style,
                                           keras.optimizers.Adam(), nodes, 0.001, 0.001)

Ss = np.array(range(30, 51))
deltas_put_nn_36 = []
deltas_put_nn_40 = []
deltas_put_nn_44 = []
deltas_put_binomial = []
for S0 in Ss:
    deltas_put_nn_36.append(delta_nn(S0, S, T/M, r, sigma, weights36[0]))
    deltas_put_nn_40.append(delta_nn(S0, S, T/M, r, sigma, weights40[0]))
    deltas_put_nn_44.append(delta_nn(S0, S, T/M, r, sigma, weights44[0]))
    deltas_put_binomial.append(delta_binomial(0.1, S0, K, T/M, r, sigma, 100, monitoring_dates, 'put'))

# Plotting
fig, ax = plt.subplots()

ax.plot(100 * Ss / K, deltas_put_nn_36, label=f'SSR [{100 * 36 / K}%]', color='green', marker='*')
ax.plot(100 * Ss / K, deltas_put_nn_40, label=f'SSR [{100 * 40 / K}%]', color='yellow', marker='*')
ax.plot(100 * Ss / K, deltas_put_nn_44, label=f'SSR [{100 * 44 / K}%]', color='red', marker='*')

ax.plot(100 * Ss / K, deltas_put_binomial, label='Binomial tree', color='black')

# Add legend
ax.legend()

# Set x and y axis labels
plt.xlabel('Moneyness S/K (%)')
plt.ylabel('Delta est.')
plt.title('Delta estimates SSR [moneyness training] vs Binomial tree method')

# Save the plot to a file
plt.savefig(f'DeltaSSR_{nodes}nodes_{S/K}moneyness{time.time()}.png', dpi=200, transparent=True)

# Show the plot
plt.show()

# deltas_nn_S0 = []
# runs = 30
# for _ in range(runs):
#     weights, option_value, normalizers = model(S, K, r, sigma, N, monitoring_dates, style,
#                                            keras.optimizers.Adam(), nodes, 0.001, 0.001)
#     delta_nn_S0 = delta_nn(S, S, T/M, r, sigma, weights[0])
#     deltas_nn_S0.append(delta_nn_S0)
#     print(f"run: {_}, delta_est: {delta_nn_S0} ")

# # Plotting
# fig, ax = plt.subplots()

# # Plot histogram for hedging_error_SS in blue with 50% transparency
# ax.scatter(range(runs), deltas_nn_S0)

# # Plot histogram for hedging_error_Delta in orange with 50% transparency
# # ax.hist(hedging_error_Delta, range=[-v0,v0], bins=200, color='orange', alpha=0.5,
# #       label='Delta hedging')

# # Add legend
# ax.legend()

# # Set x and y axis labels
# plt.xlabel('run')
# plt.ylabel('Delta est. SS')

# # Save the plot to a file
# plt.savefig(f'DeltaSS_{nodes}nodes_{S/K}moneyness{time.time()}.png', dpi=200, transparent=True)

# # Show the plot