import pandas as pd
from datetime import date, timedelta

from data_providers.components.load.load_profiles import (
    DEFAULT_PEAK_KW, DEFAULT_PROFILE, generate_day,
)


def fetch_load(today: date, peak_kw: float = DEFAULT_PEAK_KW,
               profile: str = DEFAULT_PROFILE) -> pd.DataFrame:
    tomorrow = today + timedelta(days=1)
    load = generate_day(profile, peak_kw, tomorrow)
    return pd.DataFrame({"Load": load.round(2)})
