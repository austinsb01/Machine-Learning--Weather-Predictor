from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import pandas as pd
import numpy as np

from utils import split_data, load_data

# Load and preprocess the data
X, y = load_data('modified_weather_data.csv')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_scaled, y)

# Define the model
Model = Sequential([
    Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(units=32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(units=16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(units=1, activation='linear', kernel_regularizer=l2(0.01))
])

Model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
Model.fit(X_train, y_train, epochs=100, batch_size=1000, validation_data=(X_val, y_val))

# Predict the temperatures
predictions = Model.predict(X_test)
mae = Model.evaluate(X_test, y_test, verbose=0)[1]

print("Predicted Temperatures:")
print(predictions)
print("\nMean Absolute Error:")
print(mae)
