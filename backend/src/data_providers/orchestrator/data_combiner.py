import pandas as pd
from datetime import datetime

from src.data_providers.grid.synthetic_grid import fetch_grid
from src.data_providers.load.synthetic_load import fetch_load
from src.data_providers.market_manager.IDM_DAM_features import fetch_DAM
from src.data_providers.weather.weather import fetch_weather


def combine():
    today = datetime.today()
    grid = fetch_grid(today)
    load = fetch_load(today)
    DAM = fetch_DAM(today)
    weather = fetch_weather(today)
    dataset = pd.concat([grid, load, DAM, weather], axis=1)
    return dataset

df = combine()
df.to_csv("combined.csv", index=False)