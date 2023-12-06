import numpy as np
from project_helpers import gen_paths, payoff, bs_put, bs_call, forward, d1
from scipy.stats import norm
from project_network import SemiStaticNet
import matplotlib.pyplot as plt


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
def binomial_pricer(S0, strike, T_m, rfr, vol, n, exercise_dates, style):
    m = len(exercise_dates) - 1
    dim = n * m

    dt = T_m / dim
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(rfr * dt) - d) / (u - d)

    option_values = np.zeros((dim + 1, dim + 1))

    i_values = np.arange(dim + 1)
    option_values[dim] = payoff(S0 * (u ** i_values) * (d ** (dim - i_values)), strike, style)

    for t in range(dim - 1, -1, -1):

        i_values = np.arange(t + 1)  # Array [0, 1, ..., t]
        hold_values = np.exp(-rfr * dt) * (
                p * option_values[t + 1, i_values + 1] + (1 - p) * option_values[t + 1, i_values])

        if (t % n) != 0:
            option_values[t, : t + 1] = hold_values

        else:
            option_values[t, : t + 1] = np.maximum(hold_values,
                                                   payoff(S0 * (u ** i_values) * (d ** (t - i_values)), strike, style))

    return option_values


def delta_value_bermudan(s, strike, T_m, rfr, vol, n, exercise_dates, style):
    option = binomial_pricer(s, strike, T_m, rfr, vol, n, exercise_dates, style)
    v0 = option[0, 0]
    v1 = option[1, 1]
    v2 = option[1, 0]

    m = len(exercise_dates) - 1
    dim = n * m

    dt = T_m / dim
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u

    return (v1 - v2) / (s * (u - d)), v0


def delta_hedged_portfolio(stock_path, strike, T_m, rfr, vol, n, exercise_dates, style):
    portfolio = Portfolio()

    s0 = stock_path[0]
    delta_0, v_0 = delta_value_bermudan(s0, strike, T_m, rfr, vol, n, exercise_dates, style)

    portfolio.add_cash(v_0)
    portfolio.buy_stock(delta_0, s0)

    days_until_maturity = len(stock_path)
    dt = T_m / (days_until_maturity - 1)
    new_exercise_dates = exercise_dates

    tolerance = 1e-10

    for i in range(1, days_until_maturity):
        s_i = stock_path[i]

        if (any(abs(i * dt - exercise_date) < tolerance for exercise_date in exercise_dates) & (s_i < strike)) or (
                i == days_until_maturity - 1):
            break

        portfolio.add_interest(rfr, dt)

        new_exercise_dates = new_exercise_dates - dt
        new_exercise_dates = new_exercise_dates[new_exercise_dates > 0]
        new_exercise_dates = np.insert(new_exercise_dates, 0, 0.)

        new_T = new_exercise_dates[-1]
        delta_i, v_i = delta_value_bermudan(s_i, strike, new_T, rfr, vol, n, new_exercise_dates, style)
        portfolio.buy_stock(delta_i - portfolio.stock, s_i)

    portfolio.add_interest(rfr, dt)
    portfolio.pnl = portfolio.get_portfolio_value(stock_path[i]) - payoff(stock_path[i], strike, style)

    return portfolio


def delta_hedged_european_bs_portfolio(stock_path, strike, T_m, rfr, vol, style):
    portfolio = Portfolio()

    s0 = stock_path[0]
    v_0 = bs_put(np.array([s0]), np.array([strike]), T_m, rfr, vol)[0, 0]

    portfolio.add_cash(v_0)

    delta_0 = - norm.cdf(-d1(s0, strike, T_m, rfr, vol))

    portfolio.buy_stock(delta_0, s0)

    days_until_maturity = len(stock_path)
    dt = T_m / (days_until_maturity - 1)

    for i in range(1, days_until_maturity):
        s_i = stock_path[i]

        portfolio.add_interest(rfr, dt)

        delta_i = - norm.cdf(-d1(s_i, strike, T_m - i * dt, rfr, vol))
        portfolio.buy_stock(delta_i - portfolio.stock, s_i)

    portfolio.add_interest(rfr, dt)
    portfolio.pnl = portfolio.get_portfolio_value(stock_path[-1]) - payoff(stock_path[-1], strike, style)

    return portfolio


# %%

def continuation_q(w_1, b_1, w_2, b_2, stock, delta_t):
    cont = np.full_like(stock, b_2[0])
    mask1_indices = np.where((w_1 >= 0) & (b_1 >= 0))[0]
    mask2_indices = np.where((w_1 > 0) & (b_1 < 0))[0]
    mask3_indices = np.where((w_1 < 0) & (b_1 > 0))[0]

    strikes_call = -b_1[mask2_indices] / w_1[mask2_indices]
    strikes_put = -b_1[mask3_indices] / w_1[mask3_indices]

    cont += np.sum(w_2[mask1_indices] * (w_1[mask1_indices] *
                                         np.tile(forward(stock, r, delta_t)[:, None], len(mask1_indices)) + b_1[
                                             mask1_indices]), axis=1)
    cont += np.sum(w_2[mask2_indices] * w_1[mask2_indices] * bs_call(stock, strikes_call, delta_t, r, sigma), axis=1)
    cont -= np.sum(w_2[mask3_indices] * w_1[mask3_indices] * bs_put(stock, strikes_put, delta_t, r, sigma), axis=1)

    return cont


def basket_payoff(w_1, b_1, w_2, b_2, stock_price_now, stock_price_prev, rfr, delta_t):
    mask1_indices = np.where((w_1 >= 0) & (b_1 >= 0))[0]
    mask2_indices = np.where((w_1 > 0) & (b_1 < 0))[0]
    mask3_indices = np.where((w_1 < 0) & (b_1 > 0))[0]

    strikes_call = -b_1[mask2_indices] / w_1[mask2_indices]
    strikes_put = -b_1[mask3_indices] / w_1[mask3_indices]

    total = b_2
    total += np.sum(w_2[mask1_indices] * (w_1[mask1_indices] *
                                          (stock_price_now - stock_price_prev * np.exp(rfr * delta_t))
                                          + b_1[mask1_indices]))
    total += np.sum(w_2[mask2_indices] * (w_1[mask2_indices] *
                                          payoff(np.array([stock_price_now]), strikes_call, 'call')
                                          + b_1[mask2_indices]))
    total += np.sum(w_2[mask3_indices] * (- w_1[mask3_indices] *
                                          payoff(np.array([stock_price_now]), strikes_put, 'put')
                                          + b_1[mask3_indices]))

    return total

# semi-static hedge performance for the writer of the option
def semi_static_hedge(trained_weights, dates, stock_path, premium, strike, rfr, style):
    portfolio = Portfolio()
    portfolio.add_cash(premium)

    nodes_n = len(trained_weights[0][0])
    rlnn = SemiStaticNet(hidden_nodes=nodes_n)
    n_mon = len(dates)
    differences = np.diff(dates)
    for m in range(1, n_mon):
        normalizer = stock_path[m]
        current_weights = trained_weights[- m]
        rlnn.set_weights(current_weights)
        weights_layer_1 = np.array(current_weights[0]).reshape(-1)
        biases_layer_1 = np.array(current_weights[1])
        weights_layer_2 = np.array(current_weights[2]).reshape(-1)
        biases_layer_2 = np.array(current_weights[3])
        q = continuation_q(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2,
                           stock_path[m - 1] / normalizer, differences[m - 1])

        portfolio.add_cash(-q)
        portfolio.add_interest(rfr, differences[m - 1])
        portfolio.add_cash(basket_payoff(weights_layer_1, biases_layer_1, weights_layer_2, biases_layer_2, stock_path[m]
                                         , stock_path[m - 1], rfr, differences[m - 1]))

    portfolio.pnl = portfolio.get_portfolio_value(stock_path[-1]) - payoff(stock_path[-1], strike, style)

    return portfolio


# %%

S = 40  # Initial stock price
K = 40  # Strike price
T = 1.0  # Time to maturity
r = 0.06  # Risk-free interest rate
sigma = 0.2  # Volatility
n = 100  # Number of time steps in the binomial tree
M = 1
style = 'put'
sample_size = 1
early_exercise_dates = np.linspace(0, T, M + 1)  # List of early exercise dates
discretized_time = np.linspace(0, T, int(360 * T) + 1)

# generate sample paths to test the performance of the hedging strategies
sample_paths = gen_paths(discretized_time, S, r, sigma, sample_size)

#monitored_sample_path = sample_paths[:, (M * np.arange(1, sample_paths.shape[1] / M + 1)).astype(int)]

# for i in range(sample_size):
#    plt.plot(discretized_time, sample_paths[i])
# plt.axhline(y=K, linestyle='--', c='black')
# plt.show()

pfl = delta_hedged_portfolio(sample_paths[0], K, T, r, sigma, n, early_exercise_dates, style)
print(pfl.pnl)

bs_pfl = delta_hedged_european_bs_portfolio(sample_paths[0], K, T, r, sigma, style)
print(bs_pfl.pnl)
