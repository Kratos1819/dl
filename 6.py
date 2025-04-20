"""
LSTM and GRU
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential #type:ignore 
from tensorflow.keras.layers import LSTM, Dense #type:ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load and clean dataset
df = pd.read_csv("time_series_data.csv", parse_dates=[0], index_col=0)
df = df.iloc[1:].apply(pd.to_numeric, errors='coerce').dropna()

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Create sequences
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(y.shape[1])
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

def forecast_next(data, model, steps=10):
    current_input = data[-steps:].reshape(1, steps, X.shape[2])
    forecasted = [model.predict(current_input).flatten() for _ in range(steps)]
    forecasted = scaler.inverse_transform(forecasted)

    # Print formatted output
    print("\n📊 Future Predictions 📊")
    print("Day | Close Price | High Price | Low Price  | Open Price  | Volume")
    print("----|-------------|------------|------------|--------------|-----------")
    for i, values in enumerate(forecasted, start=1):
        print(f" {i:2d} | {values[0]:10.2f} | {values[1]:9.2f} | {values[2]:9.2f} | {values[3]:9.2f} | {values[4]:9.0f}")

    return forecasted

# Call function and store results
future_forecast = forecast_next(data_scaled, model)
print(future_forecast)