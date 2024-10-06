import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
openmeteo = openmeteo_requests.Client(session=cache_session)

# Define the API URL and parameters
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 27.9475,
    "longitude": -82.4584,
    ##########################
    "start_date": "1950-01-01", #Change this to create a csv file with data between these dates
    "end_date": "1950-12-31",
    ##########################
    "hourly": [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "precipitation", "pressure_msl", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        "soil_temperature_0_to_7cm"
    ]
}

# Get the weather data
responses = openmeteo.weather_api(url, params=params)

# Process the first location
response = responses[0]
hourly = response.Hourly()

# Process hourly data into variables
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(4).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(5).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(6).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_soil_temperature_0_to_7cm = hourly.Variables(9).ValuesAsNumpy()

# Create a DataFrame with the collected data
hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ),
    "temperature_2m": hourly_temperature_2m,
    "relative_humidity_2m": hourly_relative_humidity_2m,
    "dew_point_2m": hourly_dew_point_2m,
    "precipitation": hourly_precipitation,
    "pressure_msl": hourly_pressure_msl,
    "cloud_cover": hourly_cloud_cover,
    "wind_speed_10m": hourly_wind_speed_10m,
    "wind_direction_10m": hourly_wind_direction_10m,
    "wind_gusts_10m": hourly_wind_gusts_10m,
    "soil_temperature_0_to_7cm": hourly_soil_temperature_0_to_7cm
}

# Convert the DataFrame to a Pandas DataFrame
hourly_dataframe = pd.DataFrame(data=hourly_data)

# Extract hour and day of year from the date column
hourly_dataframe['hour'] = hourly_dataframe['date'].dt.hour
hourly_dataframe['day_of_year'] = hourly_dataframe['date'].dt.dayofyear

# Add sine and cosine transformations for hour and day of year
hourly_dataframe['hour_sin'] = np.sin(2 * np.pi * hourly_dataframe['hour'] / 24)
hourly_dataframe['hour_cos'] = np.cos(2 * np.pi * hourly_dataframe['hour'] / 24)
hourly_dataframe['day_of_year_sin'] = np.sin(2 * np.pi * hourly_dataframe['day_of_year'] / 365)
hourly_dataframe['day_of_year_cos'] = np.cos(2 * np.pi * hourly_dataframe['day_of_year'] / 365)

# Prepare the final DataFrame with only 13 features (drop the date and original hour/day columns)
final_columns = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
    'precipitation', 'pressure_msl', 'cloud_cover', 
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m', 
    'soil_temperature_0_to_7cm', 'hour_sin', 'hour_cos', 
    'day_of_year_sin', 'day_of_year_cos'
]
final_dataframe = hourly_dataframe[final_columns]

# Save the DataFrame to a CSV file
final_dataframe.to_csv('1950_weather_data.csv', index=False)

print("Raw weather data with correct features has been saved to '1950_weather_data.csv'.")
