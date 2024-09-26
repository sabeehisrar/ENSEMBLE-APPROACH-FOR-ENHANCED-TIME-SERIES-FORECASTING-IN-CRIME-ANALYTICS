import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load the CSV file into a pandas DataFrame
csv_file = 'CLEAN_DATASET_IPC.csv'
df = pd.read_csv(csv_file)

# Convert 'Month-Year' column to datetime format with specified format
df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')

# Assuming 'Murder' is the target variable to be predicted
target_variable = 'Murder'

# Feature selection (you can modify this based on your dataset)
features = ['Month-Year', 'Murder']

# Create feature matrix X and target vector y
X = df[features].values
y = df[target_variable].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract the numerical features for scaling (excluding the timestamp column)
num_features_indices = [i for i, col in enumerate(features) if col != 'Month-Year']
X_train_numerical = X_train[:, num_features_indices]
X_test_numerical = X_test[:, num_features_indices]

# Standardize the numerical features (excluding timestamp)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numerical)
X_test_scaled = scaler.transform(X_test_numerical)

# Build the ANN model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.2))  # Optional dropout layer for regularization
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer with 1 neuron for regression

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Evaluate the model on the test set
loss = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {loss}')

# Make predictions for all data points
y_pred_all = model.predict(scaler.transform(X[:, num_features_indices]))

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y, y_pred_all.flatten()))
mae = mean_absolute_error(y, y_pred_all.flatten())
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')

# Plotting actual vs predicted values for all data points
plt.figure(figsize=(10, 6))
plt.plot(df['Month-Year'], y, label='Actual', marker='o')
plt.plot(df['Month-Year'], y_pred_all, label='Predicted', marker='x')
plt.title('ANN Prediction vs Actual (All Data Points)')
plt.xlabel('Month-Year')
plt.ylabel('Murder Count')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# Forecasting for the next 2 months
future_dates = pd.date_range(df['Month-Year'].max(), periods=3, freq='M')[1:]
future_data = np.zeros((len(future_dates), len(features) - 1))  # Exclude timestamp column
future_data_scaled = scaler.transform(future_data)
future_predictions = model.predict(future_data_scaled)

# Print forecasted values
for date, prediction in zip(future_dates, future_predictions):
    print(f'Forecast for {date.strftime("%b-%y")}: {prediction[0]:.2f}')

# Plotting forecasted values
plt.plot(future_dates, future_predictions, label='Forecast', marker='s')
plt.legend()
plt.tight_layout()
plt.show()

# Define the function to calculate MAPE
def calculate_mape(y_true, y_pred):
    non_zero_indices = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    return mape

mape = calculate_mape(y, y_pred_all.flatten())
print(f'MAPE: {mape:.2f}%')
