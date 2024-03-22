#%%
!pip install yahoo-fin

# %% [markdown]
# ## Code the breakout backtesting function
# %%
# !pip install matplotlib
# !pip install plotly
# !pip install python_utils
# !pip install seaborn
# !pip install pydantic
# !pip install scipy
from datetime import datetime
from sqlalchemy import create_engine
import plotly.offline as pyo
import plotly.express as px
from distutils.version import LooseVersion
from plotly import optional_imports
import plotly.io as pio
import os

import yfinance as yf
from flask import Flask, render_template, request
from IPython.display import HTML
from jinja2 import Template
from numpy.core.fromnumeric import squeeze
from PIL import Image
from pydantic import BaseModel
from scipy import integrate, stats
from setuptools import find_packages, setup
from sqlalchemy import (Boolean, Column, ForeignKey, Integer, Numeric, String,
                        create_engine)
from sqlalchemy.orm import Session, relationship
import copy
import csv
import datetime
import datetime as dt
import json
import os
import warnings
from datetime import date
from urllib.parse import urljoin
import commons
from commons import *
import matplotlib
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import plotly.io as plt_io
import python_utils
import plotly.graph_objects as go
sns.set_style('whitegrid')

pd.core.common.is_list_like = pd.api.types.is_list_like

# this helps us get the theme settings
# Necessary imports:
# this is for simple plotting with plotly express
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

windows = [15, 20, 25, 30, 40, 50,
           80, 90, 100, 125, 150, 180, 240]

def figures_to_html(figs, filename):
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)

def calculate_positions(df, starting_capital):
    # Create a new DataFrame to hold the positions
    positions = pd.DataFrame(index=df.index)

    # Calculate the position for each stock
    for symbol in df['Symbol'].unique():
        # Get the data for this stock
        stock_data = df[df['Symbol'] == symbol]

        # Calculate the position for this stock
        positions[symbol] = stock_data['Signal'] * \
            stock_data['Multiplier'] * starting_capital

    return positions


starting_capital = 10000  # Replace with the actual starting capital
positions = calculate_positions(df, starting_capital)
# Calculate total return
total_return = positions.iloc[-1].sum() / starting_capital - 1

# Calculate daily returns
daily_returns = positions.pct_change().mean(axis=1)

# Calculate Sharpe ratio (assuming a risk-free rate of 0)
sharpe_ratio = daily_returns.mean() / daily_returns.std()

# Calculate maximum drawdown
drawdown = (positions / positions.cummax() - 1).min()

# Calculate hit ratio
hit_ratio = (daily_returns > 0).mean()



def main():

    # Use a context manager to handle the file open/close operations
    with open(table_path, 'r') as file:
        df = pd.read_csv(file)

    # Convert datetime column to timezone-naive
    df['date'] = df.index

    # Ensure the DataFrame is not empty
    if df is None or df.empty:
        print("DataFrame is empty. No data to plot.")
    else:
        plot_multiplier(df, symbol)


if __name__ == "__main__":
    main()
# %%


symbol = "QQQ"
table_path = f"C:/workspaces/Congestion/hourly/signals/{symbol}.csv"

# Use a context manager to handle the file open/close operations
with open(table_path, 'r') as file:
    df = pd.read_csv(file)

# Ensure the DataFrame is not empty
if df is None or df.empty:
    print("DataFrame is empty. No data to plot.")
else:
    plot_multiplier(df, symbol)
    
# %%


table_path = os.path.join(
    "C:/workspaces/Congestion/hourly/final_tables", "QQQ.csv")
df = pd.read_csv(table_path, index_col=0, parse_dates=True)
df
# %%
df['Signal'].value_counts()
# %%
df.iloc[:, 3:]
# %%
# Shift the 'Signal' column up by one to align signals with the price movements they predict
df['Shifted_Signal'] = df['Signal'].shift(-1)

# Calculate the actual price movements
df['Price_Movement'] = df['Price'].diff()

# Calculate the predicted price movements
df['Predicted_Movement'] = df['Shifted_Signal'] * df['Price_Movement']

# Calculate the number of correct predictions
correct_predictions = (df['Predicted_Movement'] > 0).sum()

# Calculate the total number of predictions
total_predictions = df['Predicted_Movement'].count()

# Calculate the accuracy of the strategies
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy}")

# %%
# Calculate the cumulative return of the strategy
df['Strategy_Return'] = (df['Predicted_Movement'] + 1).cumprod()
# Calculate the excess return of the strategy over the benchmark

df['Benchmark_Return'] = (df['Price_Movement'] + 1).cumprod()

excess_return = df['Strategy_Return'].iloc[-1] - \
    df['Benchmark_Return'].iloc[-1]
print(f"Excess Return: {excess_return}")

# %%


# Get the list of symbols
symbols = commons.short_list
# for symbol in symbols:
#     table_path = f"C:/workspaces/Congestion/hourly/final_tables/{symbol}.csv"
#     df = pd.read_csv(table_path,index_col=0, parse_dates=True)
#     df2 = pd.DataFrame(df[['signal_ds', 'signal_cor','signal_div', 'signal_idx', 'signal_ci', 'Multiplier', 'Price']])
#     df2.style.applymap(color_negative_red).set_properties(**{'background-color': 'black', 'border-color': 'white'})

# styled = df2.style.applymap(color_negative_red).set_properties(
#     **{'background-color': 'black', 'border-color': 'white'})
# styled
# styled.to_excel('styled.xlsx')
# %%
def breakout_profitability(ticker):
  '''A function that returns a histogram and probabilistic information for all breakouts of a stock using its historical daily prices'''

  # Import libraries
  from yahoo_fin.stock_info import get_data
  import pandas as pd
  import numpy as np
  import seaborn as sns
  import matplotlib.pyplot as plt

  ## PREPARE OUR DATAFRAME FOR ANALYSIS
  # Get the historical weekly prices from the specified start date and end date (both YYYY-mm-dd)
  hist = get_data(ticker, index_as_date=False)

  # Drop the adjusted close column
  prices = hist.drop(['adjclose'], axis=1)

  # Get the selling pressure, which is distance between high and close
  prices['SellingPressure'] = prices['high'] - prices['close']

  # Get the length of candle's body (from open to close)
  prices['O-to-C'] = prices['close'] - prices['open']

  # Get the rolling mean of the candle's body for recent 20 candles
  prices['OC-20D-Mean'] = prices['O-to-C'].rolling(20).mean()

  # Get the % increase or decrease of the current body length from the rolling mean
  prices['OC-%-from-20D-Mean'] = 100*(prices['O-to-C'] - prices['OC-20D-Mean'])/prices['OC-20D-Mean']

  # Get the maximum OC compared to the recent 10 candles
  prices['MaxOC_Prev10'] = prices['O-to-C'].rolling(10).max()

  # Get the rolling mean of volume for the recent 20 candles
  prices['Volume-20D-Mean'] = prices['volume'].rolling(20).mean()

  # Get the % increase or decrease of the current volume from the rolling mean
  prices['Volume-%-from-20D-Mean'] = 100*(prices['volume'] - prices['Volume-20D-Mean'])/prices['Volume-20D-Mean']

  # Drop the null values for the first 19 rows, where no mean can be computed yet.
  prices = prices.dropna()

  # Rearrange columns
  prices = prices[['ticker', 'date', 'open', 'high', 'low', 'close',
                   'O-to-C', 'OC-20D-Mean', 'volume', 'Volume-20D-Mean',
                   'MaxOC_Prev10', 'SellingPressure', 'OC-%-from-20D-Mean',
                   'Volume-%-from-20D-Mean',
                  ]]

  # Select the subset of dataframe where breakout conditions apply
  # Conditions: 1. green candle, 2. candle's body is longest in 10 days,
  # 3. breakout volume is 50% higher than the rolling 20-day average, and
  # 4. breakout candle has body that is 100% higher than the rolling 20-day average

  condition = (prices['O-to-C'] > 0.0) & (prices['O-to-C'] == prices['MaxOC_Prev10']) & (prices['OC-%-from-20D-Mean'] >= 100.0) & (prices['SellingPressure']/prices['O-to-C'] <= 0.40) & (prices['Volume-%-from-20D-Mean'] >= 50.0)

  breakouts = prices[condition]

  ## GET THE PROFIT (OR LOSS) PER CANDLE
  # Get the index (from the dataframe) of each breakout row, which is necessary for looping later
  breakouts_indices = breakouts.index.tolist()

  profits = []
  for index in breakouts_indices:
    # For a given breakout candle index, slice the historical prices dataframe 10 rows RIGHT AFTER the breakout row
    ten_rows_after_a_breakout = prices.iloc[index+1:index+11]

    # Compute the highest price within the next 10 days RIGHT AFTER the breakout candle
    highest_price_within10days = ten_rows_after_a_breakout['high'].max()

    # Compute the lowest price within the next 10 days RIGHT AFTER the breakout candle
    lowest_price_within10days = ten_rows_after_a_breakout['low'].min()

    # Get the row index corresponding for the highest_price_within10days
    highest_price_index = ten_rows_after_a_breakout[ten_rows_after_a_breakout['high'] == highest_price_within10days].index[0]

    # Get the row index corresponding for the lowest_price_within10days
    lowest_price_index = ten_rows_after_a_breakout[ten_rows_after_a_breakout['low'] == lowest_price_within10days].index[0]

    # Calculate our Buy Price, which is the breakout candle's close
    breakout_close = breakouts.loc[index, 'close']

    # Calculate our Stop Loss Price, which is the breakout candle's open
    breakout_open = breakouts.loc[index, 'open']

    ## GET THE PROFITS:
    # If lowest_price_index is lower (or earlier) than highest_price_index, then we sold at stop loss before reaching the highest_price_within10days
    # This counts as negative profit (a loss)

    # If highest_price_index is lower (or earlier) than the lowest_price_index, this should count as a win
    # This means we were able to exit with a profit before the stock goes to the stop loss within 10 days

    if lowest_price_within10days <= breakout_open:
      if highest_price_index < lowest_price_index:
        profit = round(100*(highest_price_within10days - breakout_close)/breakout_close, 2)
        profits.append(profit)

      elif lowest_price_index <= highest_price_index:
        profit = round(100*(breakout_open - breakout_close)/breakout_close, 2)
        profits.append(profit)

    else:
      profit = round(100*(highest_price_within10days - breakout_close)/breakout_close, 2)
      profits.append(profit)

  ## GET PROFIT PER TYPE TO CALCULATE SOME PROBABILITIES
  wins = []
  breakevens = []
  losses = []
  for profit in profits:
    if profit > 0.0:
      wins.append(profit)
    elif profit == 0.0:
      breakevens.append(profit)
    elif profit < 0.0:
      losses.append(profit)

  # Calculate some trading probabilities
  win_rate = round(100*len(wins)/len(profits), 2)
  breakeven_rate = round(100*len(breakevens)/len(profits), 2)
  loss_rate = round(100*len(losses)/len(profits), 2)

  ave_positive_profit = round(sum(wins)/len(wins), 2)
  ave_negative_profit = round(sum(losses)/len(losses), 2)

  # VISUALIZE DISTRIBUTION OF PROFITS
  sns.histplot(pd.Series(profits), bins=20)
  plt.title(f"Distribution of Breakout Profits for {ticker.upper()}")
  plt.text(0.95, 0.95, f"Total Breakouts: {len(profits)} \n Ave. Positive Profit: {ave_positive_profit}% \n Ave. Negative Profit: {ave_negative_profit}% \n Win Rate: {win_rate}% \n Loss Rate: {loss_rate}% \n Breakeven Rate: {breakeven_rate}%",
           ha='right', va='top', transform=plt.gca().transAxes)
  plt.ylabel('Number of Breakouts')
  plt.xlabel('Profit (%)')
  plt.show()

  # Supply other information, in addition to the chart, for the output
  # NOTE: breakout_dates are in timestamp, so we have to convert to date format
  breakout_dates = pd.to_datetime(breakouts['date'])
  earliest_breakout = breakout_dates.min().strftime('%Y-%m-%d')
  latest_breaktout = breakout_dates.max().strftime('%Y-%m-%d')

  supplementary_info = f"Additional Info: The first breakout for {ticker} was observed on {earliest_breakout} while the most recent breakout was on {latest_breaktout}. The holding period for each breakout trade is maximum of 10 days."

  return supplementary_info

# %% [markdown]
# ## Use the `breakout_profitability()` function

# %%
breakout_profitability('GOOGL')

# %%
breakout_profitability('AAPL')


