import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # New import for Random Forest Regression

# Define the function to calculate MAPE
def calculate_mape(y_true, y_pred):
    non_zero_indices = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    return mape

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load the CSV file into a pandas DataFrame
csv_file = 'CLEAN_DATASET_IPC.csv'
df = pd.read_csv(csv_file)

# Convert 'Month-Year' column to datetime format with specified format
df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')

# Assuming 'Prep.for Dacoity' is the target variable to be predicted
target_variable = 'Other I.P.C'

# Feature selection (you can modify this based on your dataset)
features = ['Month-Year', 'Other I.P.C']

# Create feature matrix X and target vector y
X = df[features].values
y = df[target_variable].values

# Replace NaN values in y with the mean of non-NaN values
y = np.nan_to_num(y, nan=np.nanmean(y))

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Extract the numerical features for scaling (excluding the timestamp column)
num_features_indices = [i for i, col in enumerate(features) if col != 'Month-Year']
X_train_numerical = X_train[:, num_features_indices].astype(float)  # Convert to float
X_val_numerical = X_val[:, num_features_indices].astype(float)  # Convert to float
X_test_numerical = X_test[:, num_features_indices].astype(float)  # Convert to float

# Standardize the numerical features (excluding timestamp)
scaler = MinMaxScaler()  # Use MinMaxScaler instead of StandardScaler
X_train_scaled = scaler.fit_transform(X_train_numerical)
X_val_scaled = scaler.transform(X_val_numerical)
X_test_scaled = scaler.transform(X_test_numerical)

# Build the ANN model with improved architecture and hyperparameters
ann_model = Sequential()
ann_model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
ann_model.add(Dropout(0.3))  # Increase dropout for regularization
ann_model.add(Dense(64, activation='relu'))
ann_model.add(Dense(1, activation='relu'))

# Compile the ANN model with optimized optimizer and learning rate
optimizer = Adam(learning_rate=0.0005)  # Lower learning rate for stability
ann_model.compile(optimizer=optimizer, loss='mean_squared_error')

# Early stopping callback for ANN training
early_stop_ann = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the ANN model on the training set with validation
ann_history = ann_model.fit(X_train_scaled, y_train, epochs=500, batch_size=64,
                            validation_data=(X_val_scaled, y_val), verbose=1,
                            callbacks=[early_stop_ann])

# Evaluate the ANN model on the validation set
mape_ann_val = calculate_mape(y_val, ann_model.predict(X_val_scaled).flatten())
print(f'MAPE for ANN on Validation Set: {mape_ann_val:.2f}%')

# Make predictions using the ANN model
X_numerical = X[:, num_features_indices].astype(float)  # Convert to float
y_pred_ann = ann_model.predict(scaler.transform(X_numerical))

# Fit an ARIMA model on the residuals (actual - predicted by ANN)
residuals = y - y_pred_ann.flatten()
residuals = np.nan_to_num(residuals, nan=0.0)  # Replace NaN with 0.0

# Use auto_arima to find optimal ARIMA order parameters
arima_model = auto_arima(residuals, seasonal=False, trace=True)
arima_results = arima_model.fit(residuals)

# Forecast using the ARIMA model for the next 2 months (adjust as needed)
forecast_periods = 2
arima_forecast = arima_results.predict(n_periods=forecast_periods)

# Train a meta-model (Random Forest Regression) on the predictions of ANN and ARIMA
meta_model_input = np.column_stack((y_pred_ann.flatten(), np.concatenate((arima_forecast, np.zeros(len(y) - len(arima_forecast))))))

meta_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust parameters as needed
meta_model.fit(meta_model_input, y)

# Combine ANN and ARIMA predictions using the Random Forest meta-model
y_pred_stacked_rf = meta_model.predict(meta_model_input)

# Calculate MAPE for the hybrid model using Random Forest
mape_stacked_rf = calculate_mape(y, y_pred_stacked_rf[:len(y)])  
print(f'MAPE for Stacked ANN-ARIMA-RF Model: {mape_stacked_rf:.2f}%')

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate RMSE and MAE for the hybrid model using Random Forest
rmse_stacked_rf = np.sqrt(mean_squared_error(y, y_pred_stacked_rf[:len(y)]))
mae_stacked_rf = mean_absolute_error(y, y_pred_stacked_rf[:len(y)])

print(f'RMSE for Stacked ANN-ARIMA-RF Model: {rmse_stacked_rf:.2f}')
print(f'MAE for Stacked ANN-ARIMA-RF Model: {mae_stacked_rf:.2f}')


# Plotting actual vs predicted values for all data points including forecast with Random Forest
plt.figure(figsize=(10, 6))
plt.plot(df['Month-Year'], y, label='Actual', marker='o')
plt.plot(df['Month-Year'], y_pred_stacked_rf, label='Stacked Predicted (RF)', marker='x')
plt.title('Stacked ANN-ARIMA-RF Prediction vs Actual (Including Forecast)')
plt.xlabel('Month-Year')
plt.ylabel(target_variable)  # Use target variable name for y-axis label
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()

# Print the forecasted values using Random Forest
print(f'Forecasted {target_variable} for Next {forecast_periods} Months (RF):')
for i in range(forecast_periods):
    print(f'Month {i+1}: {y_pred_stacked_rf[-forecast_periods+i]:.2f}')
