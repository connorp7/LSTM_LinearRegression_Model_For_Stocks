import pandas as pd # Data manipulation library
import ta as t # Technical Analysis library
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/SPY_Trading_Project/data/SPY.csv')

data.dropna(inplace=True)

# Calculate the 50-day simple moving average (SMA)
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Calculate the 200-day simple moving average (SMA)
data['SMA_200'] = data['Close'].rolling(window=200).mean()

data['RSI'] = t.momentum.rsi(data['Close'], window=14)

# Initialize the scaler
scaler = MinMaxScaler()

# Scale the features
scaled_features = scaler.fit_transform(data[['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI']])
scaled_data = pd.DataFrame(scaled_features, columns=['Close_scaled', 'Volume_scaled', 'SMA_50_scaled', 'SMA_200_scaled', 'RSI_scaled'])

# Add the scaled features back to the original dataframe
data = pd.concat([data, scaled_data], axis=1)


# Split the data into training and testing sets
train_data = data[data.index < '2015-01-01']
test_data = data[data.index >= '2015-01-01']
