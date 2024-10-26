import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Download stock data from Yahoo Finance
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2015-01-01', end='2023-01-01')

# Feature Engineering: Add moving averages and RSI
data['7_day_MA'] = data['Close'].rolling(window=7).mean()
data['30_day_MA'] = data['Close'].rolling(window=30).mean()
data['Daily_Return'] = data['Close'].pct_change()

def compute_RSI(data, window):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_RSI(data['Close'], 14)
data = data.dropna()

# Split data into features (X) and target (y)
X = data[['Open', 'High', 'Low', 'Volume', '7_day_MA', '30_day_MA', 'RSI']]
y = data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Prices', color='b')
plt.plot(y_test.index, y_pred, label='Predicted Prices', color='r')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'Actual vs Predicted Stock Prices for {stock_symbol}')
plt.legend()
plt.show()
