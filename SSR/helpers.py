from math import log, sqrt, pi, exp
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

mu = 0.001
sigma = 0.01
start_price = 5


def stockpriceGBM(T, S0, mu, sigma, d=1):
  WienerDifference = np.random.normal(0, T, d)
  return S0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * WienerDifference)


def discretizationScheme(times, S0, mu, sigma, d=1):
  stockPrices = [S0]
  deltas = [0]

  for time in times:
    deltas.append(time - deltas[-1])
    Scurrent = stockPrices[-1]
    Snew = stockpriceGBM(deltas[-1], Scurrent, mu, sigma, d)
    stockPrices.append(Snew)

  return stockPrices

def forward(S0, r, T):
  return S0*np.exp(r*T)

def d1(S0,K,T,r,sigma):
  return(log(S0/K)+(r+sigma**2/2.)*T)/(sigma*sqrt(T))

def d2(S0,K,T,r,sigma):
  return d1(S0,K,T,r,sigma)-sigma*sqrt(T)

def bs_call(S0,K,T,r,sigma):
  return S0*norm.cdf(d1(S0,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S0,K,T,r,sigma))
  
def bs_put(S0,K,T,r,sigma):
  return K*exp(-r*T)-S0*bs_call(S0,K,T,r,sigma)
