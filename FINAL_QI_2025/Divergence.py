# %%
from dotenv import load_dotenv
import warnings
from sqlalchemy import create_engine
import plotly.offline as pyo
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as py
import plotly.graph_objects as go
import cProfile
from commons import table_path
import os
import commons
from commons import *
import datetime
import pandas as pd
# Usage# Usage
symbols = commons.short_list
# Set up date variables
now = datetime.datetime.now()
start_hourly = now - datetime.timedelta(days=720)
# Ignore warnings
warnings.filterwarnings('ignore')
# Set the style to 'dark_background'
plt.style.use('dark_background')
load_dotenv(
    'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/.gethub/.env')
# %%

class DivergenceIndexTrader:
    def __init__(self, start, now, interval, windows):
        self.start = start
        self.now = now
        self.interval = interval
        self.windows = windows
        self.data = {}

    def get_data(self, symbol):
        table_path = os.path.join(commons.table_path, f"{symbol}.csv")
        df = pd.read_csv(table_path, index_col='Datetime', parse_dates=True)
        self.data = pd.DataFrame(df)
        return self.data

    def IMO_function(self, indicator, n):
        LL_200 = indicator.rolling(window=n).min()
        HH_200 = indicator.rolling(window=n).max()
        IMO_val = (indicator - LL_200)*100 / (HH_200-LL_200)
        return IMO_val.ewm(span=2, min_periods=3).mean()

    def reverse_signal_gen(self, Signal_ID_data):
        Trade_ID = np.where(((Signal_ID_data < 75) & (
            Signal_ID_data.shift(1) > 75)) | (Signal_ID_data < 25), 1, 0)
        Trade_ID = np.where(((Signal_ID_data > 25) & (Signal_ID_data.shift(1) < 25)) | (
            Signal_ID_data > 75), -1, Trade_ID)
        return Trade_ID

    def signal_gen(self, Signal_ID_data):
        Trade_ID = np.where(((Signal_ID_data < 75) & (
            Signal_ID_data.shift(1) > 75)) | (Signal_ID_data < 25), -1, 0)
        Trade_ID = np.where(((Signal_ID_data > 25) & (Signal_ID_data.shift(1) < 25)) | (
            Signal_ID_data > 75), 1, Trade_ID)
        return Trade_ID

    def main(self, df_dep, df_ben, dependant, benchmark, n):
        corr = df_ben[f'ROC_{n}'].rolling(n).corr(df_dep[f'ROC_{n}'])
        corr_ewm = corr.ewm(span=3, min_periods=3).mean()

        DIV_ID = df_ben[f'ID_{n}']-df_dep[f'ID_{n}']
        ID = DIV_ID.divide(df_dep[f'ID_{n}']) * 100
        DS = df_dep[f'DS_{n}'] - df_ben[f'DS_{n}']

        df_dep[f'Signal_DS_{benchmark}_{n}'] = np.multiply(DS, corr_ewm)
        df_dep[f'Signal_ID_{benchmark}_{n}'] = np.multiply(ID, corr_ewm)

        df_dep[f'IMO_ID_{benchmark}_{n}'] = self.IMO_function(
            df_dep[f'Signal_ID_{benchmark}_{n}'], n)
        df_dep[f'IMO_DS_{benchmark}_{n}'] = self.IMO_function(
            df_dep[f'Signal_DS_{benchmark}_{n}'], n)
        df_dep[f'IMO_corr_{benchmark}_{n}'] = self.IMO_function(corr_ewm, n)

        df_dep[f'Trade_ID_{benchmark}_{n}'] = self.reverse_signal_gen(
            df_dep[f'IMO_ID_{benchmark}_{n}'])
        df_dep[f'Trade_DS_{benchmark}_{n}'] = self.reverse_signal_gen(
            df_dep[f'IMO_DS_{benchmark}_{n}'])
        df_dep[f'Trade_corr_{benchmark}_{n}'] = self.signal_gen(
            df_dep[f'IMO_corr_{benchmark}_{n}'])

        return df_dep[[f'Trade_corr_{benchmark}_{n}', f'Trade_DS_{benchmark}_{n}', f'Trade_ID_{benchmark}_{n}']]


# %%



# Initialize DivergenceIndexTrader
divergence_trader = DivergenceIndexTrader(start_hourly, now, "1h", windows=40)

# Initialize an empty dictionary to store benchmark data
benchmark_data = {}

# Loop over each benchmark to get the data
for benchmark in commons.benchmarks:
    benchmark_data[benchmark] = divergence_trader.get_data(benchmark)


for symbol in symbols:
    print(f"Processing symbol: {symbol}")
    dependant_data = divergence_trader.get_data(symbol)
    results = []

    # Loop over each benchmark
    for benchmark in commons.benchmarks:
        divergence_result = divergence_trader.main(
            dependant_data, benchmark_data[benchmark], symbol, benchmark, 40)
        results.append(divergence_result)

    # Concatenate results
    df_results = pd.concat(results, axis=1)

    # Concatenate results with dependant_data
    df_final = pd.concat([dependant_data, df_results], axis=1)

    # Save to a CSV file
    table_path = os.path.join(commons.table_path, f"{symbol}_divergence.csv")
    df_final.to_csv(table_path)
# %%


# %%

