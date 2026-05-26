import pandas as pd
import numpy as np
from datetime import timedelta, date


def fetch_time(today: date) -> pd.DataFrame:
    tomorrow = today + timedelta(days=1)
    hours = range(0, 24)
    minutes = [0, 15, 30, 45]
    rows = [
        {"Month": tomorrow.month, "Day": tomorrow.day, "Hour": h, "Minute": m}
        for h in hours
        for m in minutes
    ]
    df = pd.DataFrame(rows)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Minute_sin"] = np.sin(2 * np.pi * df["Minute"] / 60)
    df["Minute_cos"] = np.cos(2 * np.pi * df["Minute"] / 60)
    df["Day_of_week"] = tomorrow.isoweekday()
    df["Day_of_week_sin"] = np.sin(2 * np.pi * df["Day_of_week"] / 7)
    df["Day_of_week_cos"] = np.cos(2 * np.pi * df["Day_of_week"] / 7)
    df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 31)
    return df
