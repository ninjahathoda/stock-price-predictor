import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    return scaled, scaler

def inverse_scale(scaled_value, scaler):
    return scaler.inverse_transform([[scaled_value]])[0][0]

def prepare_lstm_data(df, feature_col='Close', window=30):
    data = df[feature_col].values
    scaled_data, scaler = scale_data(data)
    X, y = [], []
    for i in range(len(scaled_data) - window):
        X.append(scaled_data[i:i+window])
        y.append(scaled_data[i+window])
    return np.array(X), np.array(y), scaler

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
