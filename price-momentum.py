import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import yfinance as yf
import math

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Import data
def getData(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    
    # Extract adjusted closing prices for each stock
    adjClose = stockData['Adj Close']

    # Handle missing values before calculating returns
    adjClose = adjClose.dropna()

    # Calculate cumulative returns
    initial_price = adjClose.iloc[0]
    final_price = adjClose.iloc[-1]
    cumulative_returns = ((final_price - initial_price) / initial_price) * 100

    # Create a DataFrame directly with cumulative returns
    cumulative_returns_df = pd.DataFrame({'Cumulative_Return': cumulative_returns})

    return adjClose, cumulative_returns_df

# Specify stocks and date range
stocks = ['AAPL', 'MSFT', 'GOOGL', 'BAC', 'TSLA', 'NVDA', 'NFLX', 'E', 'UCG.MI', 'FBK']
start_date = '2022-01-01'
# Set current date
end_date = dt.datetime.now().strftime('%Y-%m-%d')

# Get data
adj_close_prices, cumulative_returns_df = getData(stocks, start_date, end_date)

# Determine deciles
quartile_10 = cumulative_returns_df['Cumulative_Return'].quantile(0.25)
quartile_90 = cumulative_returns_df['Cumulative_Return'].quantile(0.75)

# Create a column indicating the color for each stock
cumulative_returns_df['Color'] = np.where(cumulative_returns_df['Cumulative_Return'] <= quartile_10, 'red',
                                          np.where(cumulative_returns_df['Cumulative_Return'] >= quartile_90, 'green', 'blue'))

# Sort DataFrame by cumulative returns in descending order
cumulative_returns_df = cumulative_returns_df.sort_values(by='Cumulative_Return', ascending=False)

# Display a bar chart with colored columns
plt.figure(figsize=(12, 6))
plt.bar(cumulative_returns_df.index, cumulative_returns_df['Cumulative_Return'], color=cumulative_returns_df['Color'])
plt.title('Cumulative Return per Stock (Sorted by Cumulative Returns)')
plt.xlabel('Stock')
plt.ylabel('Cumulative Return (%)')
plt.xticks(rotation=45, ha='right')
legend_labels = ['Low Returns', 'High Returns']
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)]
plt.legend(legend_handles, legend_labels)

plt.show()


# Calculate historical volatility
def calculate_volatility(returns):
    # Calculate daily percentage returns
    returns = returns.pct_change().dropna()
    mean_returns = returns.mean()
    variance = (returns - mean_returns).pow(2).sum() / len(returns)
    volatility = math.sqrt(variance)

    return volatility

# Specify stocks and date range
stocks = ['AAPL', 'MSFT', 'GOOGL', 'BAC', 'TSLA', 'NVDA', 'NFLX', 'E', 'UCG.MI', 'FBK']
start_date = (dt.datetime.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')  # Six months ago
end_date = dt.datetime.now().strftime('%Y-%m-%d')  # Set current date

# Calculate volatility for each stock
volatility_data = adj_close_prices[stocks].apply(calculate_volatility)

# Create a DataFrame for volatility and color
volatility_df = pd.DataFrame({'Volatility': volatility_data})

# Sort DataFrame by volatility in ascending order
volatility_df = volatility_df.sort_values(by='Volatility', ascending=True)

# Determine quartiles
quartile_10_vol = volatility_df['Volatility'].quantile(0.25)
quartile_90_vol = volatility_df['Volatility'].quantile(0.75)

# Create a column indicating the color for each stock
volatility_df['Color'] = np.where(volatility_df['Volatility'] <= quartile_10_vol, 'red',
                                  np.where(volatility_df['Volatility'] >= quartile_90_vol, 'green', 'blue'))

# Display a bar chart with sorted volatilities and colored deciles
plt.figure(figsize=(10, 6))
bars = plt.bar(volatility_df.index, volatility_df['Volatility'], color=volatility_df['Color'])
plt.title('Volatility (last six months) for Each Stock (Sorted)')
plt.xlabel('Stock')
plt.ylabel('Volatility')
plt.xticks(rotation=45, ha='right')

# Create a custom legend for colors
legend_labels = ['Low Volatility', 'High Volatility']
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                  plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)]
plt.legend(legend_handles, legend_labels)

plt.show()
