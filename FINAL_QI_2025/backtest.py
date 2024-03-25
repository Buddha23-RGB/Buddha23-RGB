#%%
!pip install yahoo-fin

# %% [markdown]
# ## Code the breakout backtesting function
# %%

import commons 
from commons import *
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
from sqlalchemy import inspect
from sqlalchemy import create_engine, text
import pandas as pd
sns.set_style('whitegrid')

pd.core.common.is_list_like = pd.api.types.is_list_like

style = matplotlib.style.use('dark_background')
plt.style.use('dark_background')
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]

plt_io.templates["custom_dark"]['layout']['paper_bgcolor'] = '#30404D'
plt_io.templates["custom_dark"]['layout']['plot_bgcolor'] = '#30404D'

plt_io.templates['custom_dark']['layout']['yaxis']['gridcolor'] = '#4f687d'
plt_io.templates['custom_dark']['layout']['xaxis']['gridcolor'] = '#4f687d'

windows = [15, 20, 25, 30, 40, 50, 80, 90, 100, 125, 150, 180, 240]

engine = create_engine('sqlite:///C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\stock.db')
connection = engine.connect()

symbols = []
stock_ids = {}

def figures_to_html(figs, filename):
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)

def calculate_positions(df, starting_capital):
    positions = df['Signal'] * df['Multiplier'] * starting_capital
    return positions

def get_final_data(tickers):
    def data(ticker):
        return pd.read_csv(f"{table_path}/{ticker}.csv", index_col='Datetime')
    datas = map(data, tickers)
    return pd.concat(datas, keys=tickers, names=['Ticker', 'Datetime'])

df = get_final_data(commons.short_list)
df.fillna(0, inplace=True)

pivot_table = df['Multiplier'].unstack(level='Ticker')
pivot_table.fillna(0, inplace=True)

signal_pivot = df['Signal'].unstack(level='Ticker')
multiplier_sum = pivot_table.sum(axis=1)
signal_sum = signal_pivot.sum(axis=1)
df = df.reset_index(level='Ticker')
df['MultiplierSum'] = multiplier_sum
df['SignalSum'] = signal_sum

#%%

df.reset_index(inplace=True)
df.set_index(['Ticker', 'Datetime'], inplace=True)

df
df.to_csv(f"{final_path}/main_data.csv")
# portfolio = pd.DataFrame(data)

# portfolio.sort_values(by='Multiplier', ascending=False, inplace=True)
# #%%
# portfolio.set_index('Symbols', inplace = True)
# #%%
# html_table = portfolio.to_html('C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/templates/tables/user_portfolio.html')

#%%
import pandas as pd


def create_portfolio(df):
    data = []
    for symbol in commons.short_list:
        table = df.loc[symbol]
        data.append({
            'Symbols': symbol,
            'Price': table.Price[-1],
            'Signal_div': table['signal_div'][-1],
            'Signal_ds': table['signal_ds'][-1],
            'Signal_cor': table['signal_cor'][-1],
            'Signal_ci': table['signal_ci'][-1],
            'Signal_idx': table['signal_idx'][-1],
            'Multiplier': int(table['Multiplier'][-1]),
            'Signal': int(table['Signal'][-1])
        })
    portfolio = pd.DataFrame(data)
    portfolio.sort_values(by='Multiplier', ascending=False, inplace=True)
    portfolio.set_index('Symbols', inplace=True)
    return portfolio


def style_table(portfolio):
    styled_table = portfolio.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ]).applymap(lambda x: 'color: green' if x < 0 else 'color: black')

    html_table = f"""
    <html>
    <head>
    <style>
        table {{
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #ddd;
        }}
    </style>
    </head>
    <body>
    {styled_table.to_html()}
    </body>
    </html>
    """
    return html_table

# Load data
df = get_final_data(commons.short_list)
df.fillna(0, inplace=True)

# Create portfolio
portfolio = create_portfolio(df)

# Style table and convert to HTML
html_table = style_table(portfolio)

# Save HTML table to file
with open('C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/templates/tables/user_portfolio.html', 'w') as f:
    f.write(html_table)
#%%

portfolio

#%%
# ///continue from here
#%%
# Display the last 7 rows of the table
