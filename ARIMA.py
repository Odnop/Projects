# Import necessary libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

# Download historical data for Amazon
ticker = 'AMZN'
start_date = '2021-01-01'
end_date = '2024-01-01'
amazon_data = yf.download(ticker, start=start_date, end=end_date)

# Extract closing prices
amazon_close = amazon_data['Close']
amazon_close.index = pd.DatetimeIndex(amazon_close.index)

# Analyze the time series
plot_acf(amazon_close, lags=50, zero=False)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(amazon_close, lags=50, zero=False)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Make the time series stationary through differencing
amazon_diff = amazon_close.diff().dropna()

# Display the differenced time series
amazon_diff.plot(figsize=(12, 6))
plt.title('Differenced Amazon Stock Prices')
plt.show()

# fit stepwise auto-ARIMA using the differenced series
stepwise_fit = pm.auto_arima(amazon_diff, start_p=1, start_q=1,
                             max_p=3, max_q=3, m=12,
                             start_P=0, seasonal=True,
                             d=1, D=1, trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)


# Train the ARIMA model
p, d, q = 3, 1, 0
P, D, Q, S = 2, 1, 0, 12 
model = ARIMA(amazon_close, order=(p, d, q), seasonal_order=(P, D, Q, S))
results = model.fit()

# Print model summary
print(results.summary())

# Plot residual errors
residuals = pd.DataFrame(results.resid)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
residuals.plot(title="Residuals",figsize=(12,5), ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Make predictions with the trained model
forecast_steps = 30
forecast_index = pd.date_range(start=amazon_close.index[-1], periods=forecast_steps + 1, freq='B')[1:]
forecast_index = pd.DatetimeIndex(forecast_index)
forecast = results.get_forecast(steps=forecast_steps, index=forecast_index)
forecast_mean = forecast.predicted_mean

# Visualize the forecasts
plt.figure(figsize=(12, 6))
plt.plot(amazon_close.index, amazon_close, label='Actual Prices')
plt.plot(forecast_index, forecast_mean, color='red', label='Forecast')
plt.title('Amazon Stock Price Forecast')
plt.legend()
plt.show()




