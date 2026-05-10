import pandas as pd
from datetime import datetime

from data_providers.components.grid.synthetic_grid import fetch_grid
from data_providers.components.load.synthetic_load import fetch_load
from data_providers.components.market_manager.IDM_DAM_features import fetch_DAM
from data_providers.components.weather.weather import fetch_weather
from data_providers.components.time.time_features import fetch_time


from backend.core.database import SessionLocal
from backend.models.site import SystemConfig


def get_solar_parameters(config_id):
    db = SessionLocal()
    config_record = db.query(SystemConfig).filter(SystemConfig.id == config_id).first()
    db.close()

    if config_record and config_record.settings:
        solar = config_record.settings.get("solar", {})
        return solar.get("solar_tilt", 35), solar.get("solar_azimuth", 0)

    return 35, 0


def combine(config_id, tilt=None, azimuth=None):
    if tilt is None or azimuth is None:
        tilt, azimuth = get_solar_parameters(config_id)

    today = datetime.today()
    time = fetch_time(today)
    grid = fetch_grid(today)
    load = fetch_load(today)
    DAM = fetch_DAM(today)
    if DAM is None:
        return None
    weather = fetch_weather(today, tilt, azimuth)
    dataset = pd.concat([time, grid, load, weather, DAM], axis=1)

    for col in ('DAM_Price', 'DAM_Vol_Sale', 'DAM_Vol_Buy'):
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(
                dataset[col].astype(str)
                .str.replace('\xa0', '', regex=False)
                .str.replace(' ',    '', regex=False)
                .str.replace(',',    '.', regex=False),
                errors='coerce',
            )

    current_year = datetime.now().year
    dataset['timestamp'] = dataset.apply(
        lambda r: datetime(current_year, int(r['Month']), int(r['Day']), int(r['Hour']), int(r['Minute'])),
        axis=1
    )
    cols = dataset.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    dataset = dataset[cols]
    return dataset


if __name__ == "__main__":
    # t, a = get_solar_parameters()
    t, a = 35, 0
    df = combine(0, t, a)
    df.to_csv("combined.csv", index=False)