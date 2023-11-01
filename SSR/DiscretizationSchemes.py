import numpy as np
import matplotlib.pyplot as plt

mu = 0.001
sigma = 0.01
start_price = 5

def StockpriceGBM(t,S0,mu,sigma):
  """
  ---I played around with more dimensions but single simulation seems to be the best choice for now---
  """
  WienerDifference = np.random.normal(0,t)

  return S0*np.exp((mu - 0.5*sigma**2)*t + sigma*WienerDifference)


def discretizationScheme(times, S0, mu, sigma):
  stockPrices = [S0]

  for i in range(1,len(times)):
    delta = times[i] - times[i-1]
    Scurrent = stockPrices[-1]
    Snew = StockpriceGBM(delta,Scurrent,mu,sigma)
    stockPrices.append(Snew)

  return stockPrices