import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
import os

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
openmeteo = openmeteo_requests.Client(session=cache_session)

# Define the API URL and parameters
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 27.9475,
    "longitude": -82.4584,
    "start_date": "1950-01-01",
    "end_date": "2000-12-31",
    "hourly": [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "precipitation", "pressure_msl", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "soil_temperature_0_to_7cm"
    ]
}

# Get the weather data
response = openmeteo.weather_api(url, params=params)[0]
hourly = response.Hourly()

# Process hourly data into variables
hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
    "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
    "dew_point_2m": hourly.Variables(2).ValuesAsNumpy(),
    "precipitation": hourly.Variables(3).ValuesAsNumpy(),
    "pressure_msl": hourly.Variables(4).ValuesAsNumpy(),
    "cloud_cover": hourly.Variables(5).ValuesAsNumpy(),
    "wind_speed_10m": hourly.Variables(6).ValuesAsNumpy(),
    "wind_direction_10m": hourly.Variables(7).ValuesAsNumpy(),
    "wind_gusts_10m": hourly.Variables(8).ValuesAsNumpy(),
    "soil_temperature_0_to_7cm": hourly.Variables(9).ValuesAsNumpy()
}

# Convert the DataFrame to a Pandas DataFrame
hourly_dataframe = pd.DataFrame(data=hourly_data)

# Add time features
hourly_dataframe['hour'] = hourly_dataframe['date'].dt.hour
hourly_dataframe['day_of_year'] = hourly_dataframe['date'].dt.dayofyear

# Cyclical encoding for hour and day of year
hourly_dataframe['hour_sin'] = np.sin(2 * np.pi * hourly_dataframe['hour'] / 24)
hourly_dataframe['hour_cos'] = np.cos(2 * np.pi * hourly_dataframe['hour'] / 24)
hourly_dataframe['day_of_year_sin'] = np.sin(2 * np.pi * hourly_dataframe['day_of_year'] / 365)
hourly_dataframe['day_of_year_cos'] = np.cos(2 * np.pi * hourly_dataframe['day_of_year'] / 365)

# Prepare features and labels
features = hourly_dataframe.drop(columns=['date', 'temperature_2m', 'hour', 'day_of_year']).values
labels = hourly_dataframe['temperature_2m'].values

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and temporary sets
X_train, X_temp, y_train, y_temp = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert the splits into PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define a more complex feedforward neural network
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

# Instantiate the model and move it to the GPU
input_size = X_train_tensor.shape[1]
model = WeatherModel(input_size).to(device)

# Check if the model file exists and load it
model_filename = 'new_weather_model.pth'
if os.path.exists(model_filename):
    model.load_state_dict(torch.load(model_filename, weights_only=True))
    print("Loaded existing model.")
else:
    print("No existing model found. Training a new model.")

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the last epoch number
last_epoch = 0
if os.path.exists('training_state.txt'):
    with open('training_state.txt', 'r') as f:
        content = f.read()
        if content.strip():  # Check if the content is not empty
            last_epoch = int(content)
        else:
            print("Warning: training_state.txt is empty. Starting from epoch 0.")

# Training loop with early stopping

###########################
num_epochs = 1000000             #Change this to change how many times it trains
############################
best_val_loss = float('inf')
patience = 1000
wait = 0
epoch = 0

try:
    for epoch in range(last_epoch, num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.view(-1), y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs.view(-1), y_val_tensor)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), model_filename)  # Save the best model
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    # Save the last epoch number after training loop
    with open('training_state.txt', 'w') as f:
        f.write(str(epoch + 1))

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Ensure the last epoch number is saved even if an error occurs
    with open('training_state.txt', 'w') as f:
        f.write(str(epoch + 1))


# Load the best model for evaluation
model.load_state_dict(torch.load(model_filename, weights_only=True))
model.eval()

# Evaluate the model on the validation set and store predictions
with torch.no_grad():
    val_outputs = model(X_val_tensor).view(-1).cpu().numpy()  # Convert to NumPy array for plotting
    y_val_numpy = y_val_tensor.cpu().numpy()  # Convert actual values to NumPy array