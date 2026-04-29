import pandas as pd
from datetime import datetime

from data_providers.grid.synthetic_grid import fetch_grid
from data_providers.load.synthetic_load import fetch_load
from data_providers.market_manager.IDM_DAM_features import fetch_DAM
from data_providers.weather.weather import fetch_weather
from data_providers.time.time_features import fetch_time


from backend.core.database import SessionLocal
from backend.models.site import SystemConfig


def get_solar_parameters():
    db = SessionLocal()
    config_record = db.query(SystemConfig).order_by(SystemConfig.id.desc()).first()
    db.close()

    if config_record and config_record.settings:
        solar = config_record.settings.get("solar", {})
        return solar.get("tilt", 35), solar.get("azimuth", 0)

    return 35, 0


def combine(tilt = None, azimuth = None):
    if tilt is None or azimuth is None:
        tilt, azimuth = get_solar_parameters()

    today = datetime.today()
    time = fetch_time(today)
    grid = fetch_grid(today)
    load = fetch_load(today)
    DAM = fetch_DAM(today)
    weather = fetch_weather(today, tilt, azimuth)
    dataset = pd.concat([time, grid, load, weather, DAM], axis=1)
    return dataset


if __name__ == "__main__":
    t, a = get_solar_parameters()
    df = combine(t, a)
    df.to_csv("combined.csv", index=False)