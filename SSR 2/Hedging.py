import numpy as np
from project_helpers import gen_paths, payoff, bs_put, bs_call, d1
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from SemiStaticReplication import model
import keras
from tqdm import tqdm
import csv
import time


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
class Portfolio:
    def __init__(self, cash=0, stock=0, pnl=0):
        self.cash = cash
        self.stock = stock
        self.pnl = pnl

    def add_cash(self, amount):
        self.cash += amount

    def add_interest(self, r, t):
        self.cash *= np.exp(r * t)

    def buy_stock(self, shares, stock_price):
        cost = shares * stock_price
        self.cash -= cost
        self.stock += shares

    def get_portfolio_value(self, stock_price):
        total_value = self.cash + (self.stock * stock_price)
        return total_value


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


def delta_value_bermudan(s, strike, T_m, rfr, vol, n, exercise_dates, pf_style):
    option = binomial_pricer(s, strike, T_m, rfr, vol, n, exercise_dates, pf_style)
    v0 = option[0, 0]
    v1 = option[1, 1]
    v2 = option[1, 0]

    m = len(exercise_dates) - 1
    dim = n * m

    dt = T_m / dim
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u

    return (v1 - v2) / (s * (u - d)), v0


def delta_hedged_portfolio(stock_path, strike, T_m, rfr, vol, n, exercise_dates, pf_style):
    portfolio = Portfolio()

    s0 = stock_path[0]
    delta_0, v_0 = delta_value_bermudan(s0, strike, T_m, rfr, vol, n, exercise_dates, pf_style)

    portfolio.add_cash(v_0)
    portfolio.buy_stock(delta_0, s0)

    days_until_maturity = len(stock_path)
    dt = T_m / (days_until_maturity - 1)
    new_exercise_dates = exercise_dates

    tolerance = dt / 2

    for i in range(1, days_until_maturity):
        s_i = stock_path[i]
        new_exercise_dates = new_exercise_dates - dt
        new_exercise_dates = new_exercise_dates[new_exercise_dates > 0]
        new_exercise_dates = np.insert(new_exercise_dates, 0, 0.)

        new_T = new_exercise_dates[-1]
        delta_i, v_i = delta_value_bermudan(s_i, strike, new_T, rfr, vol, n, new_exercise_dates, pf_style)

        if (any(abs(i * dt - exercise_date) < tolerance for exercise_date in exercise_dates) & (strike - s_i > v_i)) or (
                i == days_until_maturity - 1):
            break

        portfolio.add_interest(rfr, dt)
        portfolio.buy_stock(delta_i - portfolio.stock, s_i)

    portfolio.add_interest(rfr, dt)
    portfolio.pnl = portfolio.get_portfolio_value(s_i) - payoff(s_i, strike, pf_style)

    return portfolio


def delta_hedged_european_bs_portfolio(stock_path, strike, T_m, rfr, vol, pf_style):
    portfolio = Portfolio()

    s0 = stock_path[0]
    v_0 = bs_put(np.array([s0]), np.array([strike]), T_m, rfr, vol)[0, 0]

    portfolio.add_cash(v_0)

    delta_0 = - norm.cdf(-d1(s0, strike, T_m, rfr, vol))

    portfolio.buy_stock(delta_0, s0)

    days_until_maturity = len(stock_path)
    dt = T_m / (days_until_maturity - 1)

    for i in range(1, days_until_maturity - 1):
        s_i = stock_path[i]

        portfolio.add_interest(rfr, dt)

        delta_i = - norm.cdf(-d1(s_i, strike, T_m - i * dt, rfr, vol))
        portfolio.buy_stock(delta_i - portfolio.stock, s_i)

    portfolio.add_interest(rfr, dt)
    portfolio.pnl = portfolio.get_portfolio_value(stock_path[-1]) - payoff(stock_path[-1], strike, pf_style)

    return portfolio


# %%

def cost_of_hedge(w_1, b_1, w_2, b_2, stock, delta_t, rfr, vol, normalizing_constant):
    normalized_stock = stock / normalizing_constant

    cost = b_2[0] * np.exp(-rfr * delta_t)

    mask1_indices = np.where((w_1 >= 0) & (b_1 >= 0))[0]
    mask2_indices = np.where((w_1 > 0) & (b_1 < 0))[0]
    mask3_indices = np.where((w_1 < 0) & (b_1 > 0))[0]

    strikes_call = -b_1[mask2_indices] / w_1[mask2_indices]
    strikes_put = -b_1[mask3_indices] / w_1[mask3_indices]

    cost += np.sum(w_2[mask1_indices] * (w_1[mask1_indices] *
                                         np.tile(normalized_stock[:, None], len(mask1_indices))
                                         + b_1[mask1_indices] * np.exp(-rfr * delta_t)), axis=1)[0]
    cost += np.sum(w_2[mask2_indices] * w_1[mask2_indices] *
                   bs_call(normalized_stock, strikes_call, delta_t, rfr, vol), axis=1)[0]
    cost -= np.sum(w_2[mask3_indices] * w_1[mask3_indices] *
                   bs_put(normalized_stock, strikes_put, delta_t, rfr, vol), axis=1)[0]

    return cost


def basket_payoff(w_1, b_1, w_2, b_2, stock_price_now, normalizing_constant):
    normalized_stock = stock_price_now / normalizing_constant
    mask1_indices = np.where((w_1 >= 0) & (b_1 >= 0))[0]
    mask2_indices = np.where((w_1 > 0) & (b_1 < 0))[0]
    mask3_indices = np.where((w_1 < 0) & (b_1 > 0))[0]

    strikes_call = -b_1[mask2_indices] / w_1[mask2_indices]
    strikes_put = -b_1[mask3_indices] / w_1[mask3_indices]

    total = b_2[0]
    total += np.sum(w_2[mask1_indices] * (w_1[mask1_indices] *
                                          normalized_stock
                                          + b_1[mask1_indices]))
    total += np.sum(w_2[mask2_indices] * (w_1[mask2_indices] *
                                          payoff(np.array([normalized_stock]), strikes_call, 'call')))
    total -= np.sum(w_2[mask3_indices] * (w_1[mask3_indices] *
                                          payoff(np.array([normalized_stock]), strikes_put, 'put')))

    return total


# semi-static hedge performance for the writer of the option
def semi_static_hedge(trained_weights, normalizing_sequence, dates, stock_path, premium, strike, rfr, vol, pf_style):
    portfolio = Portfolio()
    portfolio.add_cash(premium)

    n_mon = len(dates)
    differences = np.diff(dates)
    m = 0
    for m in range(1, n_mon):
        normalizer = normalizing_sequence[m - 1]
        current_weights = trained_weights[- m]

        weights_layer_1 = np.array(current_weights[0]).reshape(-1)
        biases_layer_1 = np.array(current_weights[1])
        weights_layer_2 = np.array(current_weights[2]).reshape(-1)
        biases_layer_2 = np.array(current_weights[3])

        q = cost_of_hedge(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2,
                          np.array([stock_path[m - 1]]), differences[m - 1], rfr, vol, normalizer)
        
        if strike - stock_path[m-1] > q:
            break

        portfolio.add_cash(-q)

        portfolio.add_interest(rfr, differences[m - 1])
        portfolio.add_cash(
            basket_payoff(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, stock_path[m], normalizer))


    portfolio.pnl = portfolio.get_portfolio_value(stock_path[m]) - payoff(stock_path[m], strike, pf_style)

    return portfolio


# %%
def calculate_var(pnl_array, confidence_level=0.95):
    sorted_pnls = np.sort(pnl_array)
    confidence_index = int((1 - confidence_level) * len(sorted_pnls))

    var_value = sorted_pnls[confidence_index]

    return var_value


def calculate_cvar(pnl_array, confidence_level=0.95):
    sorted_pnls = np.sort(pnl_array)
    confidence_index = int((1 - confidence_level) * len(sorted_pnls))

    tail_losses = sorted_pnls[:confidence_index]

    # Calculate CVaR as the mean of the tail losses
    cvar_value = np.mean(tail_losses)

    return cvar_value


# %%

S = 44  # Initial stock price
K = 40  # Strike price
T = 1.0  # Time to maturity
r = 0.06  # Risk-free interest rate
sigma = 0.2  # Volatility
n = 30  # Number of time steps in the binomial tree
M = 10
style = 'put'
early_exercise_dates = tuple(int(j * 360 / M) for j in range(M + 1))  # List of early exercise dates
discretized_time = np.linspace(0, T, int(360 * T) + 1)
monitoring_dates = np.array([date / 360. for date in early_exercise_dates])
# %%
sample_size = 10000
sample_paths = gen_paths(discretized_time, S, r, sigma, sample_size)
hedging_error_Delta = np.zeros(sample_size, dtype=float)
for i in tqdm(range(sample_size)):
    hedging_error_Delta[i] = delta_hedged_portfolio(sample_paths[i], K, T, r, sigma, n, monitoring_dates, style).pnl
# %%
nodes = 32
N = 30000
weights, option_value, normalizers = model(S, K, r, sigma, N, monitoring_dates, style,
                                           keras.optimizers.Adam(), nodes, 0.001, 0.001)
v0 = option_value[0, 0]
print(v0)
# %%
monitored_sample_path = sample_paths[:, early_exercise_dates]
hedging_error_SS = np.zeros(sample_size, dtype=float)
for i in tqdm(range(sample_size)):
    hedging_error_SS[i] = semi_static_hedge(weights, normalizers, monitoring_dates, monitored_sample_path[i],
                                            option_value[0, 0], K, r, sigma, style).pnl
# %%
# Plotting
fig, ax = plt.subplots()

# Plot histogram for hedging_error_SS in blue with 50% transparency
ax.hist(hedging_error_SS, range=[-v0,v0], bins=200, color='blue', alpha=0.5,
        label=f'RLNN {nodes} nodes')

# Plot histogram for hedging_error_Delta in orange with 50% transparency
ax.hist(hedging_error_Delta, range=[-v0,v0], bins=200, color='orange', alpha=0.5,
       label='Delta hedging')

# Add legend
ax.legend()

# Add vertical line at x=0
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)

# Set x and y axis labels
plt.xlabel('PnL')
plt.ylabel('Frequency')

# Set y-axis as a percentage
plt.gca().yaxis.set_major_formatter(PercentFormatter(sample_size))

# Save the plot to a file
plt.savefig(f'Hedging_Error_Comparison_{nodes}nodes_{S/K}moneyness{time.time()}.png', dpi=200, transparent=True)

# Show the plot
plt.show()

# %%
print(f'95% VaR Delta hedged portfolio: {calculate_var(hedging_error_Delta)}')
print(f'95% VaR semi-static hedged portfolio: {calculate_var(hedging_error_SS)}')
print(f'95% CVaR Delta hedged portfolio: {calculate_cvar(hedging_error_Delta)}')
print(f'95% CVaR semi-static hedged portfolio: {calculate_cvar(hedging_error_SS)}')

# %%
details = {
    'S0': S,
    'K': K,
    'mu': r,
    'sigma': sigma,
    'monitoring_dates': monitoring_dates,
    'nodes': nodes,
    'number_of_sample_paths': sample_size,
    'loss_probability_Delta': np.mean(hedging_error_Delta < 0),
    'loss_probability_SS': np.mean(hedging_error_SS < 0),
    '95%_VaR_Delta': calculate_var(hedging_error_Delta),
    '95%_VaR_SS': calculate_var(hedging_error_SS),
    '95%_CVaR_Delta': calculate_cvar(hedging_error_Delta),
    '95%_CVaR_SS': calculate_cvar(hedging_error_SS)
}
save_details_to_csv('bermudan_hedging_details.csv', details)
