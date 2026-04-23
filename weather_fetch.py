import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
	"latitude": 48.2904,
	"longitude": 25.9324,
	"start_date": "2025-01-01",
	"end_date": "2025-12-31",
	"hourly": ["temperature_2m", "shortwave_radiation"],
	"timezone": "auto",
}
responses = openmeteo.weather_api(url, params = params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_shortwave_radiation = hourly.Variables(1).ValuesAsNumpy()

hourly_data = {"datetime": pd.date_range(
	start = pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m.round(1)
hourly_data["shortwave_radiation"] = hourly_shortwave_radiation

hourly_dataframe = pd.DataFrame(data = hourly_data)
# Keep local time as plain date + HH:MM without timezone info.
hourly_dataframe["datetime"] = hourly_dataframe["datetime"].dt.tz_localize(None)
hourly_dataframe["date"] = hourly_dataframe["datetime"].dt.strftime("%Y-%m-%d")
hourly_dataframe["hour"] = hourly_dataframe["datetime"].dt.strftime("%H:%M")
hourly_dataframe = hourly_dataframe[["date", "hour", "temperature_2m", "shortwave_radiation"]]

print(hourly_dataframe.head(24))
hourly_dataframe.to_csv("weather.csv", index = False)
