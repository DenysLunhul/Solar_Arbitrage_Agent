import pandas as pd
from datetime import datetime

from data_providers.grid.synthetic_grid import fetch_grid
from data_providers.load.synthetic_load import fetch_load
from data_providers.market_manager.IDM_DAM_features import fetch_DAM
from data_providers.weather.weather import fetch_weather
from data_providers.time.time_features import fetch_time

def combine():
    today = datetime.today()
    time = fetch_time(today)
    grid = fetch_grid(today)
    load = fetch_load(today)
    DAM = fetch_DAM(today)
    weather = fetch_weather(today, tilt, azimuth)
    dataset = pd.concat([time, grid, load, weather, DAM], axis=1)
    return dataset

df = combine()
df.to_csv("combined.csv", index=False)