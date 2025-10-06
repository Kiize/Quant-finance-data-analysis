import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helper import radar_chart

"""
In this file one can find the analysis of the log returns and of various stats for 5 tickers: S&P500, Nasdaq, Google, Tesla and Apple.
"""
start = "2010-01-01"
end = "2025-01-10"

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

""" fig = plt.figure(figsize=(15, 15))

for i in range(1, len(stats_pd.columns) + 1):
    ax = fig.add_subplot(len(stats_pd.index), len(stats_pd.columns), i, projection='polar')
    radar_chart(ax, stats_pd.iloc[:, i - 1]) """

#plt.savefig("statistics.png")

# Rolling volatility

w = 30 # window of 30 days
roll_vol = log_ret_pd.rolling(w).std(ddof=0) * (252**0.5) # volatility annualized

""" roll_vol.plot(subplots=True, figsize=(10, 6))
plt.savefig("rolling_vol.png") """

# Corr matrix

""" plt.matshow(log_ret_pd.corr())
data = log_ret_pd.corr().to_numpy()
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        plt.text(j, i, "{:.2f}".format(data[i,j]), ha="center", va="center")    # annotations

plt.colorbar()  # legend
plt.xticks(range(len(log_ret_pd.columns)), log_ret_pd.columns, rotation=45) # x axis ticks
plt.yticks(range(len(log_ret_pd.columns)), log_ret_pd.columns)  # y axis ticks
plt.savefig("correlation_matrix.png") """
#plt.show()