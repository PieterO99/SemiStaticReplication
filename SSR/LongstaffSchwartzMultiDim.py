import numpy as np
from project_helpers import arithmetic_payoff, gen_paths_multivariate
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import laguerre
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm


# %%
def laguerre_poly(X, order):
    return laguerre(order)(X)


def longstaff_schwartz2(initial_stocks, correlation, strike, vols, rfr, weights, sample_size, mon_dates,
                        style, order=2):
    time_increments = np.diff(mon_dates)
    num_mon = len(time_increments)
    sample_paths = gen_paths_multivariate(initial_stocks, correlation, vols, rfr, sample_size, mon_dates, True)

    cashflow = arithmetic_payoff(strike, sample_paths[:, num_mon, :], weights, style)

    poly = PolynomialFeatures(degree=order)
    #laguerre_transformer = FunctionTransformer(laguerre_poly, kw_args={'order': order})

    for i in range(num_mon - 1, 0, -1):
        df = np.exp(- rfr * time_increments[i])
        cashflow = cashflow * df
        current_stock = sample_paths[:, i, :]
        h = arithmetic_payoff(strike, current_stock, weights, style)
        itm = h > 0

        x = poly.fit_transform(np.log(current_stock))
        #x = laguerre_transformer.fit_transform(np.log(current_stock))
        model = Ridge(alpha=0.01)
        model.fit(x, cashflow)

        q = model.predict(x)
        exercise_index = itm & (h > q)

        cashflow[exercise_index] = h[exercise_index]

    cashflow = cashflow * np.exp(- rfr * time_increments[0])
    return np.mean(cashflow), np.std(cashflow)


# %%
s = np.array([1., 1., 1., 1., 1.])
sigmas = [0.518, 0.648, 0.623, 0.57, 0.53]
w = np.array([0.381, 0.065, 0.057, 0.27, 0.227])
correlation_matrix = [[1, 0.79, 0.82, 0.91, 0.84],
                      [0.79, 1., 0.73, 0.8, 0.76],
                      [0.82, 0.73, 1., 0.77, 0.72],
                      [0.91, 0.8, 0.77, 1., 0.9],
                      [0.84, 0.76, 0.72, 0.9, 1.]]
K = 1
r = 0.05
num_samples = 200000
pf_style = 'put'
M = 10
monitoring_dates = np.linspace(0, 1., M + 1)
