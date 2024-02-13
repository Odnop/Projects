# Import necessary libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Download historical data for S&P 500
ticker = '^GSPC'
start_date = '2021-01-01'
end_date = '2024-09-02'
sp500_data = yf.download(ticker, start=start_date, end=end_date)

# Extract closing prices
sp500_close = sp500_data['Close']
sp500_close.index = pd.DatetimeIndex(sp500_close.index)
returns = sp500_data['Close'].pct_change().dropna()

# Analyze the time series
plot_acf(sp500_close, lags=50, zero=False)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(sp500_close, lags=50, zero=False)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Make the time series stationary through differencing
sp500_diff = sp500_close.diff().dropna()

# Display the differenced time series
sp500_diff.plot(figsize=(12, 6))
plt.title('Differenced Series')
plt.show()

# fit stepwise auto-ARIMA using the differenced series
stepwise_fit = pm.auto_arima(sp500_diff, start_p=1, start_q=1,
                             max_p=3, max_q=3, m=12,
                             start_P=0, seasonal=True,
                             d=1, D=1, trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)

# Train the ARIMA model
p, d, q = 3, 1, 0
P, D, Q, S = 2, 1, 0, 12

# Fit the ARIMA model to the S&P 500 closing prices
arima_model = ARIMA(sp500_close, order=(p, d, q), seasonal_order=(P, D, Q, S))
results = arima_model.fit()

# Print model summary
print(results.summary())

# Plot residual errors
residuals = pd.DataFrame(results.resid)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
residuals.plot(title="Residuals", figsize=(12,5), ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Make predictions with the trained model
forecast_steps = 30
forecast_index = pd.date_range(start=sp500_close.index[-1], periods=forecast_steps + 1, freq='B')[1:]
forecast_index = pd.DatetimeIndex(forecast_index)
forecast = results.get_forecast(steps=forecast_steps, index=forecast_index)
forecast_mean = forecast.predicted_mean

# Fit the GARCH model
best_aic = np.inf
best_model = None

for p in range(1, 4):
    for q in range(1, 4):
        model = arch_model(returns, p=p, q=q)
        res = model.fit(disp='off')
        if res.aic < best_aic:
            best_aic = res.aic
            best_model = res

garch_model = best_model
print(garch_model.summary())

# Calculate the forecasted volatility
forecast = garch_model.forecast(horizon=1, start=None)
volatility_scalar = forecast.variance.iloc[-1].iloc[0]**0.5
print(volatility_scalar)

# Calculate the volatility bands
last_price = sp500_close.iloc[-1]
days_forward = 30
future_dates = pd.date_range(start=sp500_close.index[-1], periods=days_forward + 1, freq='B')[1:]
upper_band_future = [last_price * np.exp(volatility_scalar * 1.64 * np.sqrt(i)) for i in range(1, days_forward + 1)]
lower_band_future = [last_price * np.exp(-volatility_scalar * 1.64 * np.sqrt(i)) for i in range(1, days_forward + 1)]

# Visualize the forecasts
plt.figure(figsize=(12, 6))
plt.plot(sp500_close.index, sp500_close, label='Actual Prices')
plt.plot(forecast_index, forecast_mean, color='red', label='Forecast')
plt.fill_between(future_dates, lower_band_future, upper_band_future, color='gray', alpha=0.3, label='Volatility Cone')
plt.title('S&P 500 Stock Price and Volatility Forecast')
plt.legend()
plt.show()
