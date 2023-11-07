import numpy as np
from project_helpers import gen_paths, payoff, forward, bs_put, bs_call
from project_network import fitting
# import matplotlib.pyplot as plt


# from sklearn.preprocessing import StandardScaler


# Compute the continuation value Q for the N paths at time t_{m-1}
def continuation_q(weights, stock, delta_t):
    w_1 = weights[0]  # first layer weights
    b_1 = weights[1]  # first layer biases
    w_2 = weights[2]  # second layer weights
    b_2 = weights[3][0]  # second layer bias
    p = len(w_2)  # number of hidden nodes in the first hidden layer
    cont = []  # continuation vector

    for s_tm in stock:
        sum_cond_exp = b_2
        cond_exp = 0
        for i in range(p):
            w_i = w_1[0][i]
            b_i = b_1[i]
            omega_i = w_2[i][0]

            if w_i >= 0 and b_i >= 0:
                cond_exp = w_i * forward(s_tm, r, delta_t) + b_i
            elif w_i > 0 > b_i:
                cond_exp = w_i * bs_call(s_tm, -b_i / w_i, delta_t, r, sigma)
            elif w_i < 0 < b_i:
                cond_exp = - w_i * bs_put(s_tm, -b_i / w_i, delta_t, r, sigma)
            elif w_i <= 0 and b_i <= 0:
                cond_exp = 0

            sum_cond_exp += omega_i * cond_exp

        # Discount by the risk-free rate
        cont.append(sum_cond_exp * np.exp(- r * delta_t))

    return cont


# Compute the value of exercising the option today, for the N samples
def exercise_h(stock, strike, style):
    h = [payoff(s, strike, style) for s in stock]
    return h


# Run the entire algorithm, stock is a Nx(M+1) matrix of the stock sample paths monitored at the time points
# in monitoring, option is the NxM matrix representing the option values, it is assumed to be already
# initialised with zeros apart from the last column, containing the payoffs at maturity
def model(initial_stock, strike, drift, vol, sample_size, monitoring, style):
    # Generate N=sample_size independent paths, with initial value S, visited at monitoring dates
    # stock is a Nx(M+1) matrix, the first column is all initial_stock, each row is a path
    stock = gen_paths(monitoring, initial_stock, drift, vol, sample_size)

    # Plot some paths for visualization
    # for i in range(int(sample_size / 10.)):
    #     plt.plot(np.linspace(0, T, len(monitoring) + 1), stock[i])
    # plt.xlabel('Time')
    # plt.ylabel('Stock price')
    # plt.axhline(y=strike, color='black', linestyle='-.', label='Strike Price')
    # plt.legend()
    # plt.show()

    # Evaluate final time option value for each path
    # option is of the same shape as stock, we initialize the last column as payoffs at maturity
    n_mon = len(monitoring)
    option = np.zeros(stock.shape)
    for n in range(sample_size):
        s = stock[n, n_mon]
        option[n, n_mon] = payoff(s, strike, style)

    first_difference = monitoring[0]
    differences = np.diff(monitoring)
    # We account for the fact that the first exercise date is assumed to be greater than 0
    differences = np.insert(differences, 0, first_difference)
    beta = []  # Initialize a list to store the model weights

    # Fit the model for the last time interval, store the fitted weights to initialize the next network
    # For the first iteration the weights are initialized randomly
    model_weights = fitting(stock[:, n_mon], option[:, n_mon], weights=None)
    beta.append(model_weights)
    # Compute V_{T-1}=max(h(S_{T-1},Q_{T-1})
    h = exercise_h(stock[:, n_mon - 1], strike, style)
    q = continuation_q(model_weights, stock[:, n_mon - 1], differences[n_mon - 1])
    option[:, n_mon - 1] = [max(h[i], q[i]) for i in range(sample_size)]

    # Plot the realized payoffs and the estimates given by the network

    # Iteratively compute the option values at the previous time steps
    for m in range(n_mon - 1, 0, -1):
        # Call the fitting function with initial weights given by the previous step's final weights
        model_weights = fitting(stock[:, m], option[:, m], weights=model_weights)

        # Compute the option value as maximum between the values of exercising the option or holding it
        h = exercise_h(stock[:, m - 1], strike, style)
        q = continuation_q(model_weights, stock[:, m - 1], differences[m - 1])
        option[:, m - 1] = [max(h[i], q[i]) for i in range(sample_size)]

        # Append the model weights to the beta list
        beta.append(model_weights)

    # Compute the estimate for the initial option value as a mean of the sample_size results
    initial_option_value = np.mean(option[:, 0])

    return beta, initial_option_value


# Simulation parameters initialization
# The asset prices follow a Geometric Brownian Motion
S = 40  # Initial price
mu = 0.06  # Drift
sigma = 0.2  # Volatility
K = 40  # Strike price
r = 0.06  # Risk-free rate
T = 1  # Maturity
M = 10  # Number of monitoring dates
N = 1000  # Number of sample paths
pf_style = 'put'  # Payoff type

monitoring_dates = np.linspace(0, T, M + 1)[1:]

parameters, v_0 = model(S, K, mu, sigma, N, monitoring_dates, pf_style)
print(v_0)
