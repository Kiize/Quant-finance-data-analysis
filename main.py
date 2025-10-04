import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import batched
from scipy.stats import norm
from scipy.optimize import curve_fit

"""
We study the log returns of the S&P 500 index from 1990 to 2025, focusing on their behavior during the market crashes of 2000, 2008, 2020.
"""
start = "2010-01-01"
end = "2025-01-01"

tick_list = ["^GSPC", "^IXIC", "GOOG", "TSLA", "AAPL"]

data = yf.download(tickers=tick_list, start=start, end=end, interval="1d", auto_adjust=True, group_by='ticker')
log_ret_pd = pd.DataFrame()
close_pd = pd.DataFrame()
#df = pd.DataFrame()


for (i, tick) in enumerate(tick_list):
    daily_closes = data[tick]['Close']
    log_ret = np.log(daily_closes) - np.log(daily_closes.shift(1))
    df_tmp = pd.DataFrame({tick: log_ret})
    df_tmp2 = pd.DataFrame({tick: daily_closes})
    #log_ret = log_ret.dropna()

    log_ret_pd = pd.concat([log_ret_pd, df_tmp], axis=1)
    close_pd = pd.concat([close_pd, df_tmp2], axis=1)
    #df = pd.concat([df, log_ret_pd, close_pd], axis=1)

log_ret_pd.dropna(inplace=True)
#print(log_ret_pd)
#close_pd.plot(subplots=True, figsize=(10, 12))
#plt.savefig("close.png")
#log_ret_pd.plot(subplots=True, figsize=(10, 12))
#plt.savefig("log_ret.png")

# Mean, Var, Skew, Kurtosis

stats_pd = pd.DataFrame(
    {
        "Mean": log_ret_pd.mean(),
        "Std": log_ret_pd.std(),
        "Skew": log_ret_pd.skew(),
        "Kurtosis": log_ret_pd.kurtosis()
    }
)

def radar_chart(ax, data):
    # data is transposed
    values = data.values
    indices = data.index
    N = len(indices)

    values = np.append(values, values[0])
    angles = np.arange(N + 1) / float(N) * 2 * np.pi

    plt.xticks(angles[:-1], indices)
    ax.yaxis.get_major_locator().base.set_params(nbins=3)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    plt.title(data.name)

#fig, axs = plt.subplots(2, 3, projection="polar", figsize=(10, 8))
fig = plt.figure(figsize=(15, 15))

""" for i in range(1, len(tick_list) + 1):
    ax = fig.add_subplot(2, 3, i, projection='polar')
    radar_chart(ax, stats_pd.T[tick_list[i - 1]]) """

for i in range(1, len(stats_pd.columns) + 1):
    ax = fig.add_subplot(2, 2, i, projection='polar')
    radar_chart(ax, stats_pd.iloc[:, i - 1])

plt.show()