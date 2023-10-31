import numpy as np
import matplotlib.pyplot as plt

mu = 0.001
sigma = 0.01
start_price = 5

def StockpriceGBM(T,S0,mu,sigma,d=1):
  WienerDifference = np.random.normal(0,T,d)
  return S0*np.exp((mu - 0.5*sigma**2)*T + sigma*WienerDifference)

def discretizationScheme(times,S0,mu,sigma,d=1):
  stockPrices = [S0]
  deltas = [0]

  for time in times:
    deltas.append(time - deltas[-1])
    Scurrent = stockPrices[-1]
    Snew = StockpriceGBM(deltas[-1],Scurrent,mu,sigma,d)
    stockPrices.append(Snew)

  return stockPrices