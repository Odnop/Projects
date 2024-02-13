import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Retrieve data from Yahoo Finance
amazon_data = yf.download('AMZN', start='2022-01-01', end='2024-01-01')

# Use closing price as a feature
features = amazon_data[['Close']].values

# Label: 1 if the price increases, 0 otherwise
labels = (amazon_data['Close'].shift(-1) > amazon_data['Close']).astype(int).values

# Drop missing values
amazon_data.dropna(inplace=True)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)

# Data normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
k = 4
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train_scaled, y_train)

# Predictions on the test data
predictions = knn_model.predict(X_test_scaled)

# Create an array of dates corresponding to the test data
dates = amazon_data.index[-len(predictions):]

# Plot the real prices
plt.figure(figsize=(12, 6))
plt.plot(dates, amazon_data['Close'][-len(predictions):], label='Real Price', linewidth=2)

# Plot the predicted prices
plt.plot(dates, amazon_data['Close'][-len(predictions)-1:-1].values, label='Predicted Price', linestyle='dashed', linewidth=2)

plt.title('Comparison of Real Prices and Predicted Prices')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Accuracy Score
accuracy_test = accuracy_score(y_test, knn_model.predict(X_test))
print ('Test_data Accuracy: %.2f' %accuracy_test)

