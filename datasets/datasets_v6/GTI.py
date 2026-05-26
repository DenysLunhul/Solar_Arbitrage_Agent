import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 48.29,
	"longitude": 25.93,
	"start_date": "2025-01-01",
	"end_date": "2025-12-31",
	"hourly": ["global_tilted_irradiance_instant", "temperature_2m"],
	"timezone": "auto",
	"tilt": 35,
}
responses = openmeteo.weather_api(url, params = params)
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
hourly = response.Hourly()
hourly_global_tilted_irradiance_instant = hourly.Variables(0).ValuesAsNumpy()
hourly_temperature_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["global_tilted_irradiance_instant"] = hourly_global_tilted_irradiance_instant
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe = hourly_dataframe.loc[hourly_dataframe.index.repeat(4)].reset_index(drop=True)
hourly_dataframe.to_csv("GTI.csv", index=False)
