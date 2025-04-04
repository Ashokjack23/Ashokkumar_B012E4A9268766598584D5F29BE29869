import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "air_quality_dataset.csv"
df = pd.read_csv("/content/air_quality_dataset.csv")

# Select features and target variable
features = ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (ppb)", "CO (ppm)", "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Traffic Density (vehicles/hour)"]
target = "AQI"

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Plot actual vs predicted AQI values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("SVR Model: Predicted vs Actual AQI")
plt.legend()
plt.grid(True)
plt.show()

# Plot residual errors
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='dashed', linewidth=2, label="Zero Error Line")
plt.xlabel("Residual (Actual AQI - Predicted AQI)")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.legend()
plt.grid(True)
plt.show()
