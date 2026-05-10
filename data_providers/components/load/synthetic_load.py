import numpy as np
import pandas as pd
from datetime import date, timedelta


def fetch_load(today: date) -> pd.DataFrame:
    tomorrow = today + timedelta(days=1)
    day_of_week = tomorrow.isoweekday()

    hours = range(0, 24)
    minutes = [0, 15, 30, 45]
    rows = [
        {"Hour": h, "Minute": m}
        for h in hours
        for m in minutes
    ]
    df = pd.DataFrame(rows)

    base = np.where((df["Hour"] >= 8) & (df["Hour"] <= 18), 170.0, 50.0)
    if day_of_week >= 6:
        base = base * 0.4

    noise = np.random.uniform(0.85, 1.15, size=len(df))
    spikes = np.where(np.random.rand(len(df)) > 0.98, np.random.uniform(100, 200, size=len(df)), 0.0)

    df["Load"] = (base * noise + spikes).round(2)

    return df[["Load"]]