import numpy as np
from project_helpers import payoff
import matplotlib.pyplot as plt


# %%
# Defines a lower triangular matrix describing the stock dynamics
def binomial_tree_stock(S, T, sigma, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    stock_tree = np.zeros((n + 1, n + 1))
    stock_tree[0, 0] = S

    for i in range(1, n + 1):
        stock_tree[i, i] = stock_tree[i - 1, i - 1] * u
        for j in range(i, n + 1):
            stock_tree[j, i - 1] = stock_tree[j - 1, i - 1] * d

    return stock_tree


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


# Defines a lower triangular matrix describing the Delta parameter in all possible states for any option in the binomial model
def binomial_delta(option_values, sigma, T, n, S):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    delta = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            delta[i][j] = (option_values[i + 1][j + 1] - option_values[i + 1][j]) / (
                    S * (d ** (i - j)) * (u ** (j + 1)) - S * (d ** (i + 1 - j)) * (u ** j))

    return delta


# %%

S0 = 40  # Initial stock price
K = 40  # Strike price
T = 1.0  # Time to maturity
r = 0.06  # Risk-free interest rate
sigma = 0.2  # Volatility
n = 1000  # Number of time steps in the binomial tree
M = 1  # Number of exercise dates
early_exercise_dates = np.linspace(0, T, M + 1)  # List of early exercise dates

# martingale measure parameters
dt = T / (M * n)
u = np.exp(sigma * np.sqrt(dt))  # Up movement
d = 1 / u  # Down movement
p = (np.exp(r * dt) - d) / (u - d)  # Probability of Up movement

# %%
stock_tree = binomial_tree_stock(S0, T, sigma, M * n)
bermudan_option_prices = binomial_pricer(S0, K, T, r, sigma, n, early_exercise_dates, 'put')


# %%
class Portfolio:
    def __init__(self, cash=0, stock=0):
        self.cash = cash
        self.stock = stock

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
# Randomly (according to the risk-neutral probability p) picks a possible path in the matrix of the stock price
# (and the corresponding paths for the delta and the option value)
def sample_path(stock, delta, option, p):
    n = len(stock[0])
    stock_path = [stock[0][0]]
    delta_path = [delta[0][0]]
    option_path = [option[0][0]]

    j = 0
    for i in range(1, n):

        up = np.random.binomial(1, p)
        if up:
            j += 1
            stock_path.append(stock[i][j])
            option_path.append(option[i][j])
            if i != n - 1:
                delta_path.append(delta[i][j])
        else:
            stock_path.append(stock[i][j])
            option_path.append(option[i][j])
            if i != n - 1:
                delta_path.append(delta[i][j])

    return stock_path, delta_path, option_path


# %%
# Computes the dynamics of the Delta-neutral portfolio for a possible path described by stock_path, delta_path and option_path
# The output is the history of the portfolio value, stored as 'path'
def delta_hedging_portfolio(stock_path, delta_path, option_path, T, r):
    n = len(stock_path)
    dt = T / n
    # the initial position is given by buying the stock and short selling the option
    initial_shares = delta_path[0]
    initial_cash = -option_path[0] - initial_shares * stock_path[0]
    pf = Portfolio(cash=initial_cash, stock=initial_shares)
    path = [pf.get_portfolio_value(stock_path[0]) + option_path[0]]

    for i in range(1, n - 1):
        pf.add_interest(r, dt)
        path.append(pf.get_portfolio_value(stock_path[i]) + option_path[i])
        shares_adjust = delta_path[i] - delta_path[i - 1]
        pf.buy_stock(shares_adjust, stock_path[i])
    pf.add_interest(r, dt)
    path.append(pf.get_portfolio_value(stock_path[n - 1]) + option_path[n - 1])

    return path


# %%
bermudan_delta = binomial_delta(bermudan_option_prices, sigma, T, M * n, S0)

stock_path, delta_path, option_path = sample_path(stock_tree, bermudan_delta, bermudan_option_prices, p)
portfolio_path = delta_hedging_portfolio(stock_path, delta_path, option_path, T, r)

time_values_s = np.linspace(0, 1, M * n + 1)

fig = plt.figure()
ax = fig.add_subplot(111, label="1")
ax2 = fig.add_subplot(111, label="2", frame_on=False)

ax.plot(time_values_s, portfolio_path, color='b', linestyle='-')
plt.axhline(y=K, color='g', linestyle='--', label='Strike Price')
ax.set_xlabel('Time Step')
ax.set_ylabel('Portfolio Value', color='b')

ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()
ax2.plot(time_values_s, stock_path, color='r', linestyle='-')
ax2.set_ylabel('Stock Value', color='r')
plt.legend()

#plt.show()
