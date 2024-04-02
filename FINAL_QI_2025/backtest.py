#%%

def figures_to_html(figs, filename):
    dashboard.write(inner_html)

def calculate_positions(df, starting_capital):
    positions = pd.DataFrame(index=df.index)

    positions =df['Signal'] * df['Multiplier'] * starting_capital
    positions = df['Signal'] * df['Multiplier'] * starting_capital
    return positions



def create_portfolio(df):
    data = []
    df['Price']=df.Close
    for symbol in commons.short_list:
        table = df.loc[symbol]

        fig = go.Figure()
        fig.update_layout(autosize=False, width=1200, height=600)
        fig.add_trace(go.Scatter(x=table.index,
                                 y=table['Price'], mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=table.index,
                                 y=table['Multiplier'], mode='lines', name='Multiplier', yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'))
        fig.update_layout(template='custom_dark')
        pio.write_html(
            fig, f'C:/users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/templates/charts/{symbol}.html')
        data.append({
            'Symbols': symbol,
            'Price': table.Close[-1],
            'Sig_div': table['signal_div'][-1],
            'Sig_ds': table['signal_ds'][-1],
            'Sig_cor': table['signal_cor'][-1],
            'Sig_ci': table['signal_ci'][-1],
            'Sig_idx': table['signal_idx'][-1],
            'Multiplier': int(table['Multiplier'][-1]),
            'Signal': int(table['Signal'][-1])
        })
    portfolio = pd.DataFrame(data)
    portfolio.sort_values(by='Multiplier', ascending=False, inplace=True)
    portfolio.set_index('Symbols', inplace=True)
    return portfolio


warnings.filterwarnings('ignore')


def style_table(portfolio):
    styled_table = portfolio.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
    ]).applymap(lambda x: 'color: green' if x < 0 else 'color: black')

    html_table = f"""
    <html>
    <head>
    <style>
        body {{
            background-color: #000000;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            color: #FFFFFF;
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


#%%
symbol = "AAPL"
table_file_path = os.path.join(table_path, f"{symbol}_divergence.csv")
df = pd.read_csv(table_file_path)

dd=final_table(df)
dd
#%%
# Use the table_path from the configuration file


column_names = ['symbol', 'signal_idx', 'signal_ci',
                'signal_div', 'signal_ds', 'signal_cor', 'Multiplier', 'Signal', 'Price']
table = pd.DataFrame(table, columns=column_names, index=table.index)
table

#%%
# Create portfolio
portfolio = create_portfolio(df)

# Style table and convert to HTML
html_table = style_table(portfolio)

# Save HTML table to file
with open('C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/templates/tables/user_portfolio.html', 'w') as f:
    f.write(html_table)
# %%

portfolio
# %%

mult_bull = df[df.Multiplier > 0]
mult_bear = df[df.Multiplier < 0]


# %%
mult_bear.Multiplier = mult_bear.Multiplier * -1
bearish = mult_bear.Multiplier.sum()
bullish = mult_bull.Multiplier.sum()
summ = bearish + bullish

wp = np.round(bearish/summ * 100, 2)
wb = np.round(bullish/summ * 100, 2)


weights = pd.DataFrame({"Bearish Portfolio": [wp], "Bullish Portfolio": [wb]})
print(weights)
weights.to_csv("db/weights_portfolio.csv")
# %%#%%
pivot_table = df['Multiplier'].unstack(level='Ticker')
pivot_table.fillna(0, inplace=True)
# %%
signal_pivot = df['Signal'].unstack(level='Ticker')
multiplier_sum = pivot_table.sum(axis=1)
signal_sum = signal_pivot.sum(axis=1)
df = df.reset_index(level='Ticker')
df['MultiplierSum'] = multiplier_sum
df['SignalSum'] = signal_sum

# %%

df.reset_index(inplace=True)
df.set_index(['Ticker', 'Datetime'], inplace=True)

df
df.to_csv(f"{final_path}/main_data.csv")
pivot_table.to_csv(f"{final_path}/multiplier_data.csv")
# %%
df = get_final_data(commons.short_list, table_path)
df.fillna(0, inplace=True)
df

final_path= "C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/db/final_tables"
symbol='TSLA'
df=pd.read_csv(f"{final_path}/{symbol}.csv")
column_names = ['Datetime', 'symbol','signal_idx', 'signal_ci', 'signal_div', 'signal_ds', 'signal_cor', 'Multiplier', 'Signal','Price']
df=pd.DataFrame(df[column_names])
df.set_index("Datetime", inplace=True)
starting_capital=60000
positions = calculate_positions(df, starting_capital)
df['positions']= positions
#%%
df['change']=df['Signal'].diff()
df.dropna(inplace=True)
#%%
filtered_change=df.loc[df.change !=0]

#%%

trades_log=filtered_change.loc[filtered_change['positions'] !=0]


final= trades_log.loc[trades_log['Signal'].shift(1) != trades_log['Signal']]
final['change']=final['Signal'].diff()
#%%
final.dropna(inplace=True)



#%%
# Initialize an empty DataFrame to store the portfolio
# Initialize an empty list to store the DataFrames
portfolio = []

# Iterate over all tables
for table_name in table_names:
    # Get column names for the table
    columns = inspector.get_columns(table_name)
    column_names = [column['name'] for column in columns]

    # Check if 'id' is in column names, if not replace it with the correct column name
    if 'id' not in column_names:
        # Replace 'id' with your correct column name
        correct_column_name = 'symbol'
    else:
        correct_column_name = 'id'

    # Query to get the last row from the table
    query = f"SELECT * FROM {table_name} ORDER BY {correct_column_name} DESC LIMIT 1"
    
    # Execute the query and store the result in a DataFrame
    df = pd.read_sql(query, engine)
    
    # Append the DataFrame to the list
    portfolio.append(df)

# Concatenate all the DataFrames in the list into a single DataFrame
portfolio = pd.concat(portfolio)
portfolio
df = get_final_data(commons.short_list)
df.fillna(0, inplace=True)

for symbol in commons.short_list:
    table = final_table(symbol)
    table.to_csv(f"{final_path}/{symbol}.csv")
    table.to_sql(symbol, engine, if_exists='replace')
    symbols.append(symbol)
    stock_ids[symbol] = table['id'].iloc[0]
pivot_table = df['Multiplier'].unstack(level='Ticker')
pivot_table.fillna(0, inplace=True)

signal_pivot = df['Signal'].unstack(level='Ticker')
multiplier_sum = pivot_table.sum(axis=1)
signal_sum = signal_pivot.sum(axis=1)
df = df.reset_index(level='Ticker')
df['MultiplierSum'] = multiplier_sum
df['SignalSum'] = signal_sum

#%%
# Create inspector
inspector = inspect(engine)

# Get table names
table_names = inspector.get_table_names()
column_names = ['date', 'symbol','signal_idx', 'signal_ci', 'signal_div', 'signal_ds', 'signal_cor', 'Multiplier', 'Signal','Price']
query = f"SELECT * FROM {table_name} ORDER BY {correct_column_name} DESC LIMIT 1"
    
    # Execute the query and store the result in a DataFrame
df = pd.read_sql(query, engine)
# Initialize an empty list to store the DataFrames
portfolio = []

# Iterate over all tables
for table_name in table_names:
    # Get column names for the table
    columns = inspector.get_columns(table_name)
    column_names = [column['name'] for column in columns]
df.reset_index(inplace=True)
df.set_index(['Ticker', 'Datetime'], inplace=True)

    # Check if 'id' is in column names, if not replace it with the correct column name
    if 'id' not in column_names:
        # Replace 'id' with your correct column name
        correct_column_name = 'symbol'
    else:
        correct_column_name = 'id'
df
df.to_csv(f"{final_path}/main_data.csv")
# portfolio = pd.DataFrame(data)

    # Query to get the last row from the table
    query = f"SELECT * FROM {table_name} ORDER BY {correct_column_name} DESC LIMIT 1"
    
    # Execute the query and store the result in a DataFrame
    df = pd.read_sql(query, engine)
    
    # Append the DataFrame to the list
    portfolio.append(df)

# Concatenate all the DataFrames in the list into a single DataFrame
portfolio = pd.concat(portfolio)

#%%
# portfolio.sort_values(by='Multiplier', ascending=False, inplace=True)
# #%%
# portfolio.set_index('Symbols', inplace = True)
# #%%
# html_table = portfolio.to_html('C:/Users/joech/OneDrive/Documents/Buddha23-RGB/FINAL_QI_2025/templates/tables/user_portfolio.html')

portfolio
#%%
# Attach the portfolio to the user info
user_info['portfolio'] = portfolio
# user_info['portfolio'] = portfolio
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

#%%
# Initialize an empty DataFrame to store the portfolio
# Initialize an empty list to store the DataFrames
portfolio = []

# Iterate over all tables
for table_name in table_names:
    # Get column names for the table
    columns = inspector.get_columns(table_name)
    column_names = [column['name'] for column in columns]

    # Check if 'id' is in column names, if not replace it with the correct column name
    if 'id' not in column_names:
        # Replace 'id' with your correct column name
        correct_column_name = 'symbol'
    else:
        correct_column_name = 'id'

    # Query to get the last row from the table
    query = f"SELECT * FROM {table_name} ORDER BY {correct_column_name} DESC LIMIT 1"
    
    # Execute the query and store the result in a DataFrame
    df = pd.read_sql(query, engine)
    
    # Append the DataFrame to the list
    portfolio.append(df)

# Concatenate all the DataFrames in the list into a single DataFrame
portfolio = pd.concat(portfolio)
portfolio
# Attach the portfolio to the user info


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
user_name = 'unknown'
    user_name = input("What is your name? ")
    user_password = input("Enter your password: ")
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

if __name__ == "__main__":
    main()

//TODO you are here 

#%%
for symbol in symbols:
    start_date = datetime(2020, 1, 6).date()
    end_date_range = datetime(2020, 11, 20).date()

    while start_date < end_date_range:
        end_date = start_date + timedelta(days=4)

        print(
            f"== Fetching minute bars for {symbol} {start_date} - {end_date} ==")
        minutes = api.polygon.historic_agg_v2(
            symbol, 1, 'minute', _from=start_date, to=end_date).df
        minutes = minutes.resample('1min').ffill()

        for index, row in minutes.iterrows():
            cursor.execute("""
                INSERT INTO stock_price_minute (stock_id, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (stock_ids[symbol], index.tz_localize(None).isoformat(), row['open'], row['high'], row['low'], row['close'], row['volume']))

        start_date = start_date + timedelta(days=7)

connection.commit()


with open('C:\\Users\\joech\\OneDrive\\Documents\\Buddha23-RGB\\FINAL_QI_2025\\db\\tables\\qqq.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        symbols.append(line[1])



for symbol in symbols:
    start_date = datetime(2020, 1, 6).date()
    end_date_range = datetime(2020, 11, 20).date()

    while start_date < end_date_range:
        end_date = start_date + timedelta(days=4)

        print(
            f"== Fetching minute bars for {symbol} {start_date} - {end_date} ==")
        minutes = api.polygon.historic_agg_v2(
            symbol, 1, 'minute', _from=start_date, to=end_date).df
        minutes = minutes.resample('1min').ffill()

        for index, row in minutes.iterrows():
            cursor.execute("""
                INSERT INTO stock_price_minute (stock_id, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (stock_ids[symbol], index.tz_localize(None).isoformat(), row['open'], row['high'], row['low'], row['close'], row['volume']))

        start_date = start_date + timedelta(days=7)

connection.commit()
#%%
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
    table.to_csv(table_file_path)

    plot_price_and_signals(table)

    # Save the last 20 rows to the SQL database
    with engine.connect() as connection:
        table.tail(20).to_sql(symbol, connection, if_exists='replace')
    

# %%


symbol = "QQQ"


# Ensure the DataFrame is not empty
if df is None or df.empty:
    print("DataFrame is empty. No data to plot.")
else:
    plot_multiplier(df, symbol)
    
# %%


table_path = os.path.join(
    "C:/workspaces/Congestion/hourly/final_tables", "QQQ.csv")
df = pd.read_csv(table_path, index_col=0, parse_dates=True)

symbol = "QQQ"
table_path = f"C:/workspaces/Congestion/hourly/signals/{symbol}.csv"

# Use a context manager to handle the file open/close operations
with open(table_path, 'r') as file:
    df = pd.read_csv(file)
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


//Figure out how you want to structure you GUI

data = []
for symbol in symbols:
    table = final_table(symbol)

    table.replace(0, pd.NA, inplace=True)
    table.fillna(method='ffill', inplace=True)
    print(table)
    table.to_csv(
        f"C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/{symbol}.csv")

    df = table[['Price', 'Multiplier']][-14:]
    

    df.to_html(f"templates/tables/{symbol}.html")
    df.to_sql(symbol, engine, if_exists="replace")
    warnings.filterwarnings('ignore')

    data.append({
        'Symbols': symbol,
        'Price': table.Price[-1],
        'Signal_div': table['signal_div'][-1],
        'Signal_ds': table['signal_ds'][-1],
        'Signal_cor': table['signal_cor'][-1],
        'Signal_ci': table['signal_ci'][-1],
        'Signal_idx': table['signal_idx'][-1],
        'Multiplier': int(table['Multiplier'][-1]),
        'Mult_change': int(table['Mult_change'][-1])
    })

df = pd.DataFrame(data)
df.set_index('Symbols', inplace=True)
df.sort_values(by='Multiplier', ascending=False, inplace=True)
df.to_csv('C:/Users/joech/source/repos/quantinvests_2025/app/db/multiplier_data.csv')
mult_bull = df[df.Multiplier > 0]
mult_bear = df[df.Multiplier < 0]


#%%
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

#%%
weights
# %%
from datetime import datetime
df['weightings'] = df.Multiplier/summ

df['tot_value'] = (df.weightings * cash)

df['tot_value'] = df['tot_value'].astype(int)
df['shares'] = np.floor(df.tot_value/df.Price)
df['shares'] = df['shares'].astype(int)
df.weightings=(df.weightings * 100).astype(int)

now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
df.to_csv(f'C:/Users/joech/source/repos/quantinvests_2025/app/data/user_portfolio/user_portfolio_{now}.csv')
#%%
columns_to_drop = ['Signal_div', 'Signal_ds',
                   'Signal_cor', 'Signal_ci', 'Signal_idx']
df.drop(columns=columns_to_drop, inplace=True)

# Apply the color_negative_red function to style the DataFrame
styled_df = df.reset_index(drop=False).style.applymap(
    color_negative_red, subset=['Multiplier', 'Mult_change', 'shares'])
styled_df.to_html(
    "C:/Users/joech/source/repos/quantinvests_2025/app/templates/tables/table_css.html")
styled_df
#%%

fig = go.Figure(data=[go.Pie(labels=mult_bull.index, values=mult_bull.Multiplier,
                textinfo='label+percent', insidetextorientation='radial')])
fig.update_layout(title_text='Bullish Portfolio',
                  template='plotly_dark', font=dict(size=20))
pyo.plot(fig, filename='C:/Users/joech/source/repos/quantinvests_2025/app/static/pie_table_long.html', auto_open=True)
fig.show()
#%%
fig2 = go.Figure(data=[go.Pie(labels=mult_bear.index, values=mult_bear.Multiplier,
                 textinfo='label+percent', insidetextorientation='radial')])
fig2.update_layout(title_text='Bearish Portfolio',
                   template='plotly_dark', font=dict(size=20))
pyo.plot(fig2, filename='C:/Users/joech/source/repos/quantinvests_2025/app/static/pie_table_short.html', auto_open=True)
fig2.show()
# ///continue from here
#%%

print(f"Bullish Portfolio: {wb}%")
print(f"Bearish Portfolio: {wp}%")

# %%
dfs = {}

symbols = ss.short_list
for symbol in symbols:

    df = pd.read_csv(
        f"C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/{symbol}.csv", index_col='Datetime')
    dfs[symbol] = pd.concat([df])

# %%
portfolio = pd.DataFrame()
for symbol in symbols:
    portfolio[symbol] = dfs[symbol]["Multiplier"].replace(
        0, pd.NA).fillna(method='ffill')

# %%
portfolio['total_weightings'] = portfolio.sum(axis=1)

total_weight = abs(portfolio)
total_weight
# %%
portfolio['abs_total_weightings'] = total_weight[::-1].sum(axis=1)

# %%
portfolio.to_csv(
    "C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/total_weightings.csv")
portfolio = portfolio[-400:]
ax = portfolio['total_weightings'].plot(figsize=(14, 6))
ax.axhline(65, color='r', linestyle='--')  # Add horizontal line at y=70
ax.axhline(0, color='w', linestyle='-')  # Add horizontal line at y=70
ax.axhline(-65, color='g', linestyle='--')  # Add horizontal line at y=-70
fig2 = portfolio['total_weightings'].plot(
    ax=ax, secondary_y='Multiplier', style='--', title="QI Custom Hourly Weightings Indicator")
ax.figure.savefig(
    "C:/Users/joech/source/repos/quantinvests_2025/app/static/total_weightings_indicator.jpg")
# portfolio = pd.read_csv("/workspaces/quantinvests_2025/app/db/multiplier_data.csv")
#%%

def get_daily_data(tickers):

    def data(ticker):
        return pd.read_csv(f"C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/{ticker}.csv", index_col='Datetime')
    datas = map(data, tickers)
    return (pd.concat(datas, keys=tickers, names=['Ticker', 'Datetime']))


symbols = ss.short_list
db = get_daily_data(symbols)
db.round(2)
db.to_csv(
    "C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/final_signal_table.csv")

db.replace(0, pd.NA, inplace=True)
db.fillna(method='ffill', inplace=True)
db.to_sql("signal_table", engine, if_exists="replace")

#%%
# Create a line plot of total_weightings
fig = go.Figure()
fig.update_layout(
    autosize=False,
    width=1200,
    height=600,
)
fig.add_trace(go.Scatter(x=portfolio.index,
              y=portfolio['total_weightings'], mode='lines', name='total_weightings'))

# Add horizontal lines
fig.add_shape(type="line", x0=portfolio.index.min(), x1=portfolio.index.max(
), y0=70, y1=70, line=dict(color="Red", width=1, dash="dash"))
fig.add_shape(type="line", x0=portfolio.index.min(
), x1=portfolio.index.max(), y0=0, y1=0, line=dict(color="White", width=1))
fig.add_shape(type="line", x0=portfolio.index.min(), x1=portfolio.index.max(
), y0=-70, y1=-70, line=dict(color="Green", width=1, dash="dash"))

# Apply the custom dark theme
fig.update_layout(template='custom_dark',
                  title="QI Custom Hourly Weightings Indicator")

# Save the figure as an HTML file
pio.write_html(
    fig, 'C:/Users/joech/source/repos/quantinvests_2025/app/templates/charts/total_weightings_indicator.html')

fig.show()
# %%
portfolio
# Display the last 7 rows of the table

# ///continue from here
# %%
dfs = {}

symbols = ss.short_list
for symbol in symbols:

    df = pd.read_csv(
        f"C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/{symbol}.csv", index_col='Datetime')
    dfs[symbol] = pd.concat([df])

# %%
portfolio = pd.DataFrame()
for symbol in symbols:
    portfolio[symbol] = dfs[symbol]["Multiplier"].replace(
        0, pd.NA).fillna(method='ffill')

# %%
portfolio['total_weightings'] = portfolio.sum(axis=1)

total_weight = abs(portfolio)
total_weight
# %%
portfolio['abs_total_weightings'] = total_weight[::-1].sum(axis=1)

# %%
portfolio.to_csv(
    "C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/total_weightings.csv")
# portfolio = pd.read_csv("/workspaces/quantinvests_2025/app/db/multiplier_data.csv")
#%%

def get_daily_data(tickers):

    def data(ticker):
        return pd.read_csv(f"C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/{ticker}.csv", index_col='Datetime')
    datas = map(data, tickers)
    return (pd.concat(datas, keys=tickers, names=['Ticker', 'Datetime']))


symbols = ss.short_list
db = get_daily_data(symbols)
db.round(2)
db.to_csv(
    "C:/Users/joech/source/repos/quantinvests_2025/app/db/portfolio_tables/final_signal_table.csv")

db.replace(0, pd.NA, inplace=True)
db.fillna(method='ffill', inplace=True)
db.to_sql("signal_table", engine, if_exists="replace")

#%%
# Create a line plot of total_weightings
fig = go.Figure()
fig.update_layout(
    autosize=False,
    width=1200,
    height=600,
)
fig.add_trace(go.Scatter(x=portfolio.index,
              y=portfolio['total_weightings'], mode='lines', name='total_weightings'))

# Add horizontal lines
fig.add_shape(type="line", x0=portfolio.index.min(), x1=portfolio.index.max(
), y0=70, y1=70, line=dict(color="Red", width=1, dash="dash"))
fig.add_shape(type="line", x0=portfolio.index.min(
), x1=portfolio.index.max(), y0=0, y1=0, line=dict(color="White", width=1))
fig.add_shape(type="line", x0=portfolio.index.min(), x1=portfolio.index.max(
), y0=-70, y1=-70, line=dict(color="Green", width=1, dash="dash"))

# Apply the custom dark theme
fig.update_layout(template='custom_dark',
                  title="QI Custom Hourly Weightings Indicator")

# Save the figure as an HTML file
pio.write_html(
    fig, 'C:/Users/joech/source/repos/quantinvests_2025/app/templates/charts/total_weightings_indicator.html')

fig.show()
# %%
portfolio
# Display the last 7 rows of the table
