import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from main import log_ret_pd

n_assets = log_ret_pd.shape[1]
weights = np.ones(n_assets) / n_assets

port_ret = log_ret_pd.dot(weights)

#port_ret.plot()

# Cumulative returns

port_cum = np.exp(port_ret.cumsum()) - 1
benchmark_cum = np.exp(log_ret_pd['^GSPC'].cumsum()) - 1

port_cum.plot(label='Portfolio')
benchmark_cum.plot(label='S&P 500')
plt.legend()
plt.title('Cumulative Returns: Portfolio vs S&P 500')
plt.savefig('portfolio_vs_benchmark.png')

plt.show()