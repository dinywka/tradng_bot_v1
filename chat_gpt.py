import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import yfinance as yf

# Function to get historical stock data
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to create a simple neural network model
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to prepare data for training the model
def prepare_data(data, target_col, look_back=5):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back)][target_col].values
        X.append(a)
        y.append(data.iloc[i + look_back][target_col])
    return np.array(X), np.array(y)

# Stock symbol and date range
stock_symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2020-01-01'

# Download stock data
stock_data = get_stock_data(stock_symbol, start_date, end_date)

# Selecting only 'Close' prices for simplicity
data = stock_data[['Close']]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Prepare data for training
look_back = 5  # Number of previous days to consider
X, y = prepare_data(pd.DataFrame(data_scaled, columns=['Close']), 'Close', look_back)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = create_model(look_back)
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Set: {mse}')

# Visualize the predictions
import matplotlib.pyplot as plt

y_test_inverse = scaler.inverse_transform(np.array([y_test]).reshape(-1, 1))
y_pred_inverse = scaler.inverse_transform(y_pred)

plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse, label='Actual Prices')
plt.plot(y_pred_inverse, label='Predicted Prices')
plt.title('Yahoo Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

import gym
from stable_baselines3 import PPO

# Create and wrap the CartPole environment
env = gym.make('CartPole-v1')

# Define the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model for 100,000 steps
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_cartpole")

# Evaluate the trained model
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    env.render()

# Close the environment
env.close()
