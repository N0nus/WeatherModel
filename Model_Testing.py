import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import random
import os

# Define the WeatherModel class
class WeatherModel(nn.Module):
    def __init__(self, input_size):
        super(WeatherModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#################################
DATASET = "2000-2019_weather_data.csv"   #Change this to change what dataset is being tested
#################################

# Load the dataset
hourly_dataframe = pd.read_csv(DATASET)

# Prepare features and labels
features = hourly_dataframe.drop(columns=['temperature_2m']).values
labels = hourly_dataframe['temperature_2m'].values

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

# Load the best model for evaluation
input_size = X_tensor.shape[1]  # Number of input features
model = WeatherModel(input_size).to(device)

# Load the model weights
model.load_state_dict(torch.load('new_weather_model.pth'))
model.eval()

# Function to make predictions using a random data point from the dataset
def predict_random_weather(model, features, labels):
    random_index = random.randint(0, len(features) - 1)
    selected_features = features[random_index]
    actual_temperature = labels[random_index]

    # Prepare the feature tensor for the model
    input_data = torch.tensor(selected_features, dtype=torch.float32).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        predicted_temperature = model(input_data).item()

    print(f"Index: {random_index} | "
          f"Predicted: {predicted_temperature:.2f} °C | "
          f"Actual: {actual_temperature:.2f} °C | "
          f"Difference: {predicted_temperature - actual_temperature:.2f} °C")

# Call the random prediction function multiple times
for _ in range(10):
    predict_random_weather(model, features_scaled, labels)

# Evaluate the model
with torch.no_grad():
    outputs = model(X_tensor).view(-1).cpu().numpy()  # Get predictions
    y_numpy = y_tensor.cpu().numpy()  # Get actual labels

# Calculate and print performance metrics
mse = np.mean((outputs - y_numpy) ** 2)
mae = np.mean(np.abs(outputs - y_numpy))
print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')

# Plot predicted vs actual temperatures
plt.figure(figsize=(12, 6))
plt.scatter(y_numpy, outputs, alpha=0.5, color='blue', label='Predicted vs Actual')
plt.plot([y_numpy.min(), y_numpy.max()], [y_numpy.min(), y_numpy.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Predicted vs Actual Temperatures')
plt.legend()
plt.grid()
plt.show()

def predict_next_day_weather(model, features, labels, current_index):
    # Get the timestamp of the current index
    current_time = hourly_dataframe.index[current_index]  # Get the index or datetime from the DataFrame

    # Check if the next day's data exists in the DataFrame
    next_day_index = current_index + 24  # Assuming hourly data
    if next_day_index < len(features):
        selected_features = features[next_day_index]
        actual_temperature = labels[next_day_index]

        # Prepare the feature tensor for the model
        input_data = torch.tensor(selected_features, dtype=torch.float32).unsqueeze(0).to(device)

        # Make prediction
        model.eval()
        with torch.no_grad():
            predicted_temperature = model(input_data).item()

        print(f"Predicted for next day at index {next_day_index}: "
              f"{predicted_temperature:.2f} °C | Actual: {actual_temperature:.2f} °C | "
              f"Difference: {predicted_temperature - actual_temperature:.2f} °C")
    else:
        print("Next day's data not available in the dataset.")

# Call the next day prediction function for a random index
for _ in range(5):
    random_index = random.randint(0, len(features) - 25)
    predict_next_day_weather(model, features_scaled, labels, random_index)