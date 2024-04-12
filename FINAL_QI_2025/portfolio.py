# %%
from datetime import datetime
from Congestion import *
from Divergence import *

# %%
import json
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

        # Create a 4-row subplot figure
        fig, axs = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

        # Plot the Price on the first row
        axs[0].plot(df_final.index, df_final['Price'],
                    label='Price', color='skyblue')

        # Plot signal changes as markers
        long_signals = df_final[df_final['Signal'].diff() > 0]
        short_signals = df_final[df_final['Signal'].diff() < 0]
        axs[0].scatter(long_signals.index, long_signals['Price'],
                       label='Long Signal', color='g', marker='^')
        axs[0].scatter(short_signals.index, short_signals['Price'],
                       label='Short Signal', color='r', marker='v')

        # Customize the first plot
        axs[0].set_title(f'Price with Signal Changes for {symbol}')
        axs[0].set_ylabel('Price')
        axs[0].legend()

        # Set y-axis limits with buffer
        axs[0].set_ylim(price_min - price_buffer, price_max + price_buffer)

        # Plot the multipliers on the remaining rows
        axs[1].plot(df_final.index, df_final['signal_div'],
                    label='Multiplier Div', color='purple')
        axs[2].plot(df_final.index, df_final['signal_cor'],
                    label='Multiplier Cor', color='orange')
        axs[3].plot(df_final.index, df_final['signal_ds'],
                    label='Multiplier DS', color='brown')

        # Customize the multiplier plots
        for i, label in enumerate(['Multiplier Div', 'Multiplier Cor', 'Multiplier DS'], start=1):
            axs[i].set_ylabel(label)
            axs[i].legend()

        axs[3].set_xlabel('Date')  # Set x-label on the last plot

        # Save the figure
        chart_filename = f"C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\charts\\price_chart_{symbol}.jpg"
        fig.savefig(chart_filename)
        plt.close(fig)  # Close the figure to free up memory


warnings.filterwarnings('ignore')

# Concatenate results and save to a CSV file


engine = create_engine(
    'sqlite:///C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\stock.db')


# Load the configuration file
with open('config.json') as f:
    config = json.load(f)

# Get the paths from the configuration file
root_dir = config['root_dir']
table_path = config['table_path']

for symbol in symbols:
    table = final_table(symbol)

    # Fill NaN values with the method 'ffill'
    table.fillna(method='ffill', inplace=True)

    # Use the table_path from the configuration file
    table_file_path = os.path.join(table_path, f"{symbol}.csv")

    column_names = ['symbol', 'signal_idx', 'signal_ci',
                    'signal_div', 'signal_ds', 'signal_cor', 'Multiplier', 'Signal', 'Price']
    table = pd.DataFrame(table, columns=column_names, index=table.index)

    table.to_csv(table_file_path)

    plot_price_and_signals(table)

    # Save the last 20 rows to the SQL database
    with engine.connect() as connection:
        table.tail(20).to_sql(symbol, connection, if_exists='replace')


# %%
def get_final_data(tickers):
    def data(ticker):
        return pd.read_csv(f"{table_path}/{ticker}.csv", index_col='Datetime')
    datas = map(data, tickers)
    return pd.concat(datas, keys=tickers, names=['Ticker', 'Datetime'])


# Load data
df = get_final_data(commons.short_list)
df.fillna(0, inplace=True)
# %%
pivot_table = df['Multiplier'].unstack(level='Ticker')
pivot_table.fillna(0, inplace=True)
# %%
signal_pivot = df['Signal'].unstack(level='Ticker')
multiplier_sum = pivot_table.sum(axis=1)
signal_sum = signal_pivot.sum(axis=1)
dd = pd.DataFrame(multiplier_sum)
# $%%
dd.columns = ['MultiplierSum']
dd['SignalSum'] = signal_sum
dd.to_csv("C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/final_tables/final_table.csv")
# %%
signal_sum
# %%

df = df.reset_index(level='Ticker')
df['MultiplierSum'] = multiplier_sum
df['SignalSum'] = signal_sum
# %%

df.MultiplierSum
# %%
df.reset_index(inplace=True)
df.set_index(['Ticker', 'Datetime'], inplace=True)
df.to_csv(f"{final_path}/main_data.csv")
pivot_table.to_csv(f"{final_path}/multiplier_data.csv")

# %%


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


# Create portfolio
portfolio = create_portfolio(df)
# %%

# Style table and convert to HTML

html_table = style_table(portfolio)
now = datetime.datetime.now()

# Format the timestamp to only include the date and replace colons with underscores
formatted_now = now.strftime('%Y-%m-%d_%H_%M_%S')

# Save CSV to file
df.to_csv(
    f'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/user_portfolio/user_portfolio_{formatted_now}.csv')

# Save HTML table to file
with open('C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/templates/tables/user_portfolio.html', 'w') as f:
    # Assuming you want to write the `html_table` to the file
    f.write(html_table)
    f.write(html_table)
# %%
mult_bull = portfolio[portfolio.Multiplier > 0]
mult_bear = portfolio[portfolio.Multiplier < 0]
mult_bear.Multiplier = mult_bear.Multiplier * -1
bearish = mult_bear.Multiplier.sum()
bullish = mult_bull.Multiplier.sum()
summ = bearish + bullish
wp = np.round(bearish/summ * 100, 2)
wb = np.round(bullish/summ * 100, 2)
weights = pd.DataFrame({"Bearish Portfolio": [wp], "Bullish Portfolio": [wb]})
print(weights)
weights.to_csv("db/weights_portfolio.csv")
cash = 100000 / 0.5
cash

# %%

fig = go.Figure(data=[go.Pie(labels=mult_bull.index, values=mult_bull.Multiplier,
                textinfo='label+percent', insidetextorientation='radial')])
fig.update_layout(title_text='Bullish Portfolio',
                  template='plotly_dark', font=dict(size=20))
pyo.plot(fig, filename='C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/static/pie_table_long.html', auto_open=True)
fig.show()
# %%
fig2 = go.Figure(data=[go.Pie(labels=mult_bear.index, values=mult_bear.Multiplier,
                 textinfo='label+percent', insidetextorientation='radial')])
fig2.update_layout(title_text='Bearish Portfolio',
                   template='plotly_dark', font=dict(size=20))
pyo.plot(fig2, filename='C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/static/pie_table_short.html', auto_open=True)
fig2.show()

# df.loc[['MultplierSum']

plot_multiplier = multiplier_sum.iloc[-400:]
# %%
ax = plot_multiplier.plot(figsize=(14, 6))
ax.axhline(65, color='r', linestyle='--')  # Add horizontal line at y=70
ax.axhline(0, color='w', linestyle='-')  # Add horizontal line at y=70
ax.axhline(-65, color='g', linestyle='--')  # Add horizontal line at y=-70
fig2 = plot_multiplier.plot(
    ax=ax, secondary_y='MultiplierSum', style='--', title="QI Custom Hourly Weightings Indicator")
ax.figure.savefig(
    "C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/static/total_weightings_indicator.jpg")
# Create a line plot of total_weightings

# %%
plot_signals = signal_sum.iloc[-400:]
plot_signals = pd.DataFrame(
    plot_signals, index=plot_signals.index, columns=['SignalSum'])
# %%
plot_signals
# %%
ax = plot_signals.plot(figsize=(14, 6))
fig = go.Figure()
fig.update_layout(
    autosize=False,
    width=1200,
    height=600,
)
fig.add_trace(go.Scatter(x=plot_signals.index,
              y=plot_signals['SignalSum'], mode='lines', name='total_weightings'))

# Add horizontal lines
fig.add_shape(type="line", x0=plot_signals.index.min(), x1=plot_signals.index.max(
), y0=15, y1=15, line=dict(color="Red", width=1, dash="dash"))
fig.add_shape(type="line", x0=plot_signals.index.min(
), x1=plot_signals.index.max(), y0=0, y1=0, line=dict(color="White", width=1))
fig.add_shape(type="line", x0=plot_signals.index.min(), x1=plot_signals.index.max(
), y0=-15, y1=-15, line=dict(color="Green", width=1, dash="dash"))

# Apply the custom dark theme
fig.update_layout(template='custom_dark',
                  title="QI Custom Hourly Weightings Indicator")
fig.show()
# Save the figure as an HTML file
pio.write_html(
    fig, 'C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/templates/charts/total_signals_indicator.html')

# %%
columns_to_drop = ['Signal_div', 'Signal_ds',
                   'Signal_cor', 'Signal_ci', 'Signal_idx']
portfolio.drop(columns=columns_to_drop, inplace=True)

# Apply the color_negative_red function to style the DataFrame
styled_df = portfolio.reset_index(drop=False).style.applymap(
    color_negative_red, subset=['Multiplier', 'Signal'])
styled_df.to_html(
    "C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/templates/tables/table_css.html")
# %%
styled_df
# %%
