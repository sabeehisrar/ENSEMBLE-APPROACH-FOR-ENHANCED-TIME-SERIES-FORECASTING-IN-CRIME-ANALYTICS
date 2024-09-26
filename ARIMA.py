import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the CSV file into a pandas DataFrame
csv_file = 'CLEAN_DATASET_IPC.csv'
df = pd.read_csv(csv_file)

# Convert 'Month-Year' column to datetime format with specified format
df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b-%y')

# Set 'Month-Year' column as the index (assuming it represents the time index)
df.set_index('Month-Year', inplace=True)

# Ensure no missing values in the target column
df['Other I.P.C'].dropna(inplace=True)

# Print the first few rows of the DataFrame to verify the data
print("First few rows of the DataFrame:")
print(df.head())

# Fit auto ARIMA model
model = auto_arima(df['Other I.P.C'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)

# Get in-sample predicted values
predicted_values = model.predict_in_sample()

# Forecast future values
n_periods = 12  # Example: forecast next 12 periods
forecast_values = model.predict(n_periods=n_periods)

# Calculate error metrics
def calculate_mape(y_true, y_pred):
    non_zero_indices = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    return mape

actual_values = df['Other I.P.C'].values
mape = calculate_mape(actual_values, predicted_values)
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
mae = mean_absolute_error(actual_values, predicted_values)

print(f'MAPE for Predicted Values: {mape:.2f}%')
print(f'RMSE for Predicted Values: {rmse:.2f}')
print(f'MAE for Predicted Values: {mae:.2f}')

# Print forecasted values
print("Forecasted Values:")
print(forecast_values)

# Plot observed data, predicted values, and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Other I.P.C'], label='Observed', marker='o')
plt.plot(df.index, predicted_values, label='Predicted', linestyle='--')
plt.plot(pd.date_range(start=df.index[-1], periods=n_periods+1, freq='M')[1:], forecast_values, label='Forecast', linestyle='--')
plt.title('ARIMA Model Prediction and Forecast vs Observed Data')
plt.xlabel('Month-Year')
plt.ylabel('Other I.P.C')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
