#%%
import warnings
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
from sqlalchemy import create_engine
import commons
from commons import *

# Define the color_negative_red function


def color_negative_red(value):
    color = 'red' if value < 0 else 'green' if value > 0 else 'white'
    return f'color: {color}'

# Define the figures_to_html function


def figures_to_html(figs, filename):
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)


def IMO_function(indicator, n):
    LL_200 = indicator.rolling(window=n).min()
    HH_200 = indicator.rolling(window=n).max()
    IMO_val = (indicator - LL_200)*100 / (HH_200-LL_200)
    return IMO_val.ewm(span=2, min_periods=3).mean()


def reverse_signal_gen(Signal_ID_data):
    Trade_ID = np.where(((Signal_ID_data < 75) & (
        Signal_ID_data.shift(1) > 75)) | (Signal_ID_data < 25), 1, 0)
    Trade_ID = np.where(((Signal_ID_data > 25) & (Signal_ID_data.shift(1) < 25)) | (
        Signal_ID_data > 75), -1, Trade_ID)
    return Trade_ID


def signal_gen(Signal_ID_data):
    Trade_ID = np.where(((Signal_ID_data < 75) & (
        Signal_ID_data.shift(1) > 75)) | (Signal_ID_data < 25), -1, 0)
    Trade_ID = np.where(((Signal_ID_data > 25) & (Signal_ID_data.shift(1) < 25)) | (
        Signal_ID_data > 75), 1, Trade_ID)
    return Trade_ID


# Set up date variables
now = datetime.datetime.now()
start_daily = now - datetime.timedelta(days=2500)
start_hourly = now - datetime.timedelta(days=720)
start_quarter = now - datetime.timedelta(days=80)

# Set up database connection
rel_path = commons.table_path
engine = create_engine(
    'sqlite:///C:/workspaces/Congestion/hourly/sql/datastore.db')

# Get the list of symbols
symbols = commons.short_list
# %%
# Define the final_table function


def final_table(symbol):
    table_path = os.path.join(commons.table_path, f"{symbol}_divergence.csv")
    df = pd.read_csv(table_path, index_col=0, parse_dates=True)
    df_final = pd.DataFrame()
    df_final['Datetime'] = df.index
    df_final.set_index('Datetime', inplace=True)
    # Multiplier Calculation
    filter_col_div = [col for col in df if col.startswith('Trade_ID')]
    filter_col_ds = [col for col in df if col.startswith('Trade_DS')]
    filter_col_corr = [col for col in df if col.startswith('Trade_corr')]
    filter_col_ci = [col for col in df if col.startswith('CI_signal')]
    filter_col_idx = [col for col in df if col.startswith('IDX_signal')]

    df_final['multiplier_div'] = df[filter_col_div].replace(
        to_replace=0, method='ffill').sum(axis=1)
    df_final['multiplier_cor'] = df[filter_col_corr].replace(
        to_replace=0, method='ffill').sum(axis=1)
    df_final['multiplier_ds'] = df[filter_col_ds].replace(
        to_replace=0, method='ffill').sum(axis=1)

    df_final_IMO = IMO_function(df_final, 14)

    df_final['signal_idx'] = df['IDX_signal_40'].replace(
        to_replace=0, method='ffill')
    df_final['signal_idx'] = df_final['signal_idx'] * 2
    df_final['signal_ci'] = df['CI_signal_40'].replace(
        to_replace=0, method='ffill')
    df_final['signal_div'] = reverse_signal_gen(
        df_final_IMO['multiplier_div'])
    df_final['signal_ds'] = reverse_signal_gen(
        df_final_IMO['multiplier_ds'])
    df_final['signal_cor'] = signal_gen(
        df_final_IMO['multiplier_cor'])
    df_final['Multiplier'] = df_final[['signal_ds', 'signal_cor',
                                       'signal_div', 'signal_idx', 'signal_ci']].sum(axis=1)
    df_final['IMO_Signal'] = IMO_function(df_final['Multiplier'], 40)
    df_final['Signal'] = signal_gen(df_final.IMO_Signal)
    df_final['Mult_change'] = df_final['Signal'].diff()
    df_final['symbol'] = symbol
    df_final['Price'] = df.Close.round(2)
    return df_final.dropna()


def plot_price_and_signals(df_final):
    # Ensure the DataFrame is not empty
    if df_final is None or df_final.empty:
        print("DataFrame is empty. No data to plot.")
    else:
        # Calculate the buffer for y-axis limits
        buffer_percent = 0.05  # 5% buffer above and below the price range
        df_final = df_final.iloc[-400:]
        price_min = df_final['Price'].min()
        price_max = df_final['Price'].max()
        price_buffer = (price_max - price_min) * buffer_percent
        df_final = df_final.iloc[-400:]
        # Plot the Price
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df_final.index, df_final['Price'],
                label='Price', color='skyblue')

        # Plot signal changes as markers
        long_signals = df_final[df_final['Signal'].diff() > 0]
        short_signals = df_final[df_final['Signal'].diff() < 0]
        ax.scatter(long_signals.index,
                   long_signals['Price'], label='Long Signal', color='g', marker='^')
        ax.scatter(short_signals.index,
                   short_signals['Price'], label='Short Signal', color='r', marker='v')

        # Customize and show the plot
        ax.set_title(f'Price with Signal Changes for {symbol}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        # Set y-axis limits with buffer
        ax.set_ylim(price_min - price_buffer, price_max + price_buffer)

        # Save the figure
        chart_filename = f'C:/workspaces/Congestion/hourly/charts_divergence/price_chart_{symbol}.jpg'
        fig.savefig(chart_filename)
        plt.close(fig)  # Close the figure to free up memory


warnings.filterwarnings('ignore')

# Concatenate results and save to a CSV file


engine = create_engine(
    'sqlite:///C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\stock.db')
db_file = r"C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\stock.db"
for symbol in symbols:
    table = final_table(symbol)

    # Fill NaN values with the method 'ffill'
    table.fillna(method='ffill', inplace=True)

    table_path = os.path.join(
        "C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/final_tables", f"{symbol}.csv")
    table.to_csv(table_path)

    plot_price_and_signals(table)

    # Save the last 20 rows to the SQL database
    with engine.connect() as connection:
        table.tail(20).to_sql(symbol, connection, if_exists='replace')
    
  
#%%
table
# %%

