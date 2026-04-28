from datetime import date, timedelta

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


def fetch_weather(today: date) -> pd.DataFrame:
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	tomorrow = today + timedelta(days=1)
	target_date = tomorrow.strftime("%Y-%m-%d")
	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": 48.2904,
		"longitude": 25.9324,
		"hourly": ["temperature_2m", "shortwave_radiation", "global_tilted_irradiance_instant"],
		"timezone": "auto",
		"tilt": 35,
		"start_date": target_date,
		"end_date": target_date,
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
	hourly_global_tilted_irradiance_instant = hourly.Variables(2).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["Temperature_2m"] = hourly_temperature_2m
	hourly_data["Shortwave_radiation"] = hourly_shortwave_radiation
	hourly_data["Global_tilted_irradiance_instant"] = hourly_global_tilted_irradiance_instant

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	hourly_dataframe.drop(columns=["date"], inplace=True)
	hourly_dataframe = hourly_dataframe.loc[hourly_dataframe.index.repeat(4)].reset_index(drop=True)
	return hourly_dataframe