import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download historical data for Amazon stock
amazon_data = yf.download('AMZN', start='2022-01-01', end='2024-01-01')

# Calculate the 50-day and 200-day moving averages
short_window = 50
long_window = 200

# Utilizing function rolling in order to compute all the avarages
amazon_data['50_MA'] = amazon_data['Close'].rolling(window=short_window, min_periods=1).mean()
amazon_data['200_MA'] = amazon_data['Close'].rolling(window=long_window, min_periods=1).mean()

# Plot the data with moving averages
plt.figure(figsize=(10, 6))
plt.plot(amazon_data['Close'], label='Close Price')
plt.plot(amazon_data['50_MA'], label=f'50-day MA')
plt.plot(amazon_data['200_MA'], label=f'200-day MA')

plt.title('Amazon Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
