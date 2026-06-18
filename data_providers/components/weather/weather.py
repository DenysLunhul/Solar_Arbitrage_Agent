from datetime import date, timedelta
import openmeteo_requests
import requests
import pandas as pd
from retry_requests import retry


def fetch_weather(today: date, tilt: float = 35, azimuth: float = 0) -> pd.DataFrame:
	retry_session = retry(requests.Session(), retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)
	tomorrow = today + timedelta(days=1)
	target_date = tomorrow.strftime("%Y-%m-%d")
	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": 48.2904,
		"longitude": 25.9324,
		"hourly": ["temperature_2m", "shortwave_radiation", "global_tilted_irradiance_instant"],
		"timezone": "auto",
		"tilt": tilt,
		"azimuth": azimuth,
		"start_date": target_date,
		"end_date": target_date,
	}
	responses = openmeteo.weather_api(url, params = params)
	response = responses[0]
	print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Elevation: {response.Elevation()} m asl")
	print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
	print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
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
