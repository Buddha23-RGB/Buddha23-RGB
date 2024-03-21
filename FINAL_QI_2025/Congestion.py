# %%
import matplotlib.pyplot as plt

import commons
from commons import *
import numpy as np
import pandas as pd
import yfinance as yf
import time
from dotenv import load_dotenv
import os
# !pip install schedule
# !pip install matplotlib
# Load environment variables from .env file if it exists

symbols = commons.short_list
windows = commons.windows
style = matplotlib.style.use('dark_background')
plt.style.use('dark_background')
# create our custom_dark theme from the plotly_dark template
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]

# set the paper_bgcolor and the plot_bgcolor to a new color
plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = '#30404D'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = '#30404D'

# you may also want to change gridline colors if you are modifying background
plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'

sns.set_style('whitegrid')
# %%


class CongestionIndexTrader:
    def __init__(self, symbol, start, now, interval, windows):
        self.symbol = symbol
        self.start = start
        self.now = now
        self.interval = interval
        self.windows = windows
        self.df = None

    def IMO_function(self, indicator, n):
        LL = indicator.rolling(window=n).min()
        HH = indicator.rolling(window=n).max()
        IMO_val = (indicator - LL) / (HH - LL) * 100
        return IMO_val.ewm(span=2, min_periods=3).mean()

    def signal_gen(self, IMO_df):
        Trade_ID = np.where(
            ((IMO_df < 75) & (IMO_df.shift(1) > 75)) | (IMO_df < 25), -1, 0)
        Trade_ID = np.where(
            ((IMO_df > 25) & (IMO_df.shift(1) < 25)) | (IMO_df > 75), 1, Trade_ID)
        return Trade_ID

    def calculate_window_indicators(self):
        for window in self.windows:
            HH = self.df['High'].rolling(window=window).max()
            LL = self.df['Low'].rolling(window=window).min()
            CI = self.df['Close'].pct_change(
                window) / (HH.pct_change(window) + 1e-9) * 100
            IDX = (self.df['Close'] - LL) / (HH - LL) * 100
            self.df[f'CI_{window}'] = CI
            self.df[f'IDX_{window}'] = IDX
            self.df[f'CI_IMO_{window}'] = self.IMO_function(CI, window)
            self.df[f'IDX_IMO_{window}'] = self.IMO_function(IDX, window)
            self.df[f'CI_signal_{window}'] = self.signal_gen(
                self.df[f'CI_IMO_{window}'])
            self.df[f'IDX_signal_{window}'] = self.signal_gen(
                self.df[f'IDX_IMO_{window}'])
            self.df[f'CI_trend_{window}'] = np.where(
                CI > 20, "Bullish", np.where(CI < -20, "Bearish", "Congested"))
            self.df[f"ROC_{window}"] = (
                (self.df.Close - self.df.Close[-window]) / self.df.Close[-window]) * 100
            std = self.df['Close'].rolling(window=window, min_periods=5).std()
            mid_band = self.df['Close'].rolling(
                window=window, min_periods=5).mean()
            ATR = self.df['TR'].rolling(window=window).mean()
            DS = ((self.df['Close'] - mid_band)/mid_band) * 100
            ID = ((self.df['Close']-mid_band) + (2*std))/(4*std)
            self.df[f'std_{window}'] = std
            self.df[f'mid_band_{window}'] = mid_band
            self.df[f'upper_band_{window}'] = mid_band + 2*std
            self.df[f'lower_band_{window}'] = mid_band - 2*std
            self.df[f'ATR_{window}'] = ATR
            self.df[f'lower_keltner_{window}'] = mid_band - (ATR * 1.5)
            self.df[f'upper_keltner_{window}'] = mid_band + (ATR * 1.5)
            self.df[f'DS_{window}'] = DS.ewm(span=3, adjust=False).mean()
            self.df[f'ID_{window}'] = ID.ewm(span=3, adjust=False).mean()

    def snapshot(self):
        self.df = yf.download(self.symbol, start=self.start,
                              end=self.now, interval=self.interval)
        self.df['date'] = self.df.index
        self.df['returns'] = np.log(
            self.df['Adj Close'] / self.df['Adj Close'].shift(1))
        self.df['log_returns'] = np.log(
            self.df['Close'] / self.df['Close'].shift(1))
        self.df['TR'] = abs(self.df['High'] - self.df['Low'])
        self.calculate_window_indicators()
        return self.df.fillna(0)

    def plot_trends_and_signals(self, window):
        # Ensure the DataFrame is not empty
        if self.df is None or self.df.empty:
            print("DataFrame is empty. No data to plot.")
            return

        # Slice the DataFrame to the last 'slice_' rows
        slice_ = 255
        df_slice = self.df[-slice_:]

        # Calculate the buffer for y-axis limits
        buffer_percent = 0.05  # 5% buffer above and below the price range
        price_min = df_slice['Close'].min()
        price_max = df_slice['Close'].max()
        price_buffer = (price_max - price_min) * buffer_percent

        # Plot the closing price
        # Save the figure and axis reference
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df_slice.index, df_slice['Close'],
                label='Closing Price', color='skyblue')

        # Plot CI_signal as markers
        ci_long_signals = df_slice[df_slice[f'CI_signal_{window}'] > 0]
        ci_short_signals = df_slice[df_slice[f'CI_signal_{window}'] < 0]
        ax.scatter(ci_long_signals.index,
                   ci_long_signals['Close'], label='CI Long Signal', marker='^', color='lime')
        ax.scatter(ci_short_signals.index,
                   ci_short_signals['Close'], label='CI Short Signal', marker='v', color='red')

        # Plot IDX_signal as markers
        idx_long_signals = df_slice[df_slice[f'IDX_signal_{window}'] > 0]
        idx_short_signals = df_slice[df_slice[f'IDX_signal_{window}'] < 0]
        ax.scatter(idx_long_signals.index,
                   idx_long_signals['Close'], label='IDX Long Signal', marker='o', color='aqua', alpha=0.5)
        ax.scatter(idx_short_signals.index,
                   idx_short_signals['Close'], label='IDX Short Signal', marker='x', color='fuchsia', alpha=0.5)

        # Highlight CI_trend areas with more opaque colors
        bullish_trend = df_slice[df_slice[f'CI_trend_{window}'] == 'Bullish']
        bearish_trend = df_slice[df_slice[f'CI_trend_{window}'] == 'Bearish']
        congested_trend = df_slice[df_slice[f'CI_trend_{window}']
                                   == 'Congested']

        # Plot bullish trend areas
        ax.fill_between(
            bullish_trend.index, bullish_trend['Close'], color='green', alpha=0.3, label='Bullish Trend')
        # Plot bearish trend areas
        ax.fill_between(
            bearish_trend.index, bearish_trend['Close'], color='red', alpha=0.5, label='Bearish Trend')
        # Plot congested trend areas
        ax.fill_between(congested_trend.index,
                        congested_trend['Close'], color='orange', alpha=0.3, label='Congested Trend')

        # Customize and show the plot
        ax.set_title(
            f'Closing Price with CI and IDX Signals for {self.symbol}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        # Set y-axis limits with buffer
        ax.set_ylim(price_min - price_buffer, price_max + price_buffer)

        # Save the figure
        chart_filename = f'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/charts/trending_chart_{self.symbol}.jpg'
        fig.savefig(chart_filename)
        plt.close(fig)  # Close the figure to free up memory


# %%
"""Plot"""
# Set up date variables
now = datetime.datetime.now()
start_daily = now - datetime.timedelta(days=2500)
start_hourly = now - datetime.timedelta(days=720)
start_quarter = now - datetime.timedelta(days=80)

# Set the style to 'dark_background'
plt.style.use('dark_background')
# Usage
# Initialize a DataFrame to store the best window for each symbol
best_windows_df = pd.DataFrame(
    columns=['Symbol', 'Best Window', 'Performance'])
windows = commons.windows

for symbol in short_list:
    # Initialize the trader with the necessary parameters
    trader = CongestionIndexTrader(
        symbol=symbol, start=start_hourly, now=now, interval='1h', windows=windows)

    # Download the data and calculate indicators
    df_snapshot = trader.snapshot()
    df = pd.DataFrame(df_snapshot)

    # Plot trends and signals
    # Use the first window size for plotting
    trend_path = f"C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/tables/{symbol}.csv"
    df.to_csv(trend_path)
    # Assuming this method exists in your class
    trader.plot_trends_and_signals(window=40)
#%%