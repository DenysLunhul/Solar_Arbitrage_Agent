"""
change_load.py
==============
Regenerates the Load column in dataset_final.csv for the full year.

Profile (kW):
    Weekday business hours (8-18): base ~60 kW
    Weekday night / early morning: base ~20 kW
    Weekend:                       base * 0.4

Each day gets its own random multiplier (0.92-1.08) so no two days
are identical. Each 15-min step also gets independent noise (0.85-1.15)
plus a 2% chance of a short-duration spike (+5-15 kW).

Usage:
    python change_load.py

After running:
    python normalize.py --input dataset_final.csv \
                        --output dataset_normalized.csv \
                        --scalers models/scalers.pkl
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

DATA_PATH    = Path('../environment/dataset_final.csv')
ALSO_UPDATE  = Path('../datasets/dataset_v10/dataset_final.csv')

NIGHT_BASE    = 20.0   # kW  (00:00-07:45 and 19:00-23:45)
DAY_BASE      = 60.0   # kW  (08:00-18:45)
WEEKEND_SCALE = 0.40   # factory mostly idle on weekends
DAY_START     = 8
DAY_END       = 18
SEED          = 42


def generate_load(df: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(SEED)

    ts   = pd.to_datetime(df['timestamp'])
    hour = ts.dt.hour.values
    dow  = ts.dt.dayofweek.values   # 0=Mon … 6=Sun

    # Per-day multiplier so each day looks slightly different
    dates        = ts.dt.date.values
    unique_dates = list(dict.fromkeys(dates))
    day_mult_map = {d: rng.uniform(0.92, 1.08) for d in unique_dates}
    day_mult     = np.array([day_mult_map[d] for d in dates])

    is_business = (hour >= DAY_START) & (hour <= DAY_END)
    is_weekend  = dow >= 5   # Sat=5, Sun=6

    base = np.where(is_business, DAY_BASE, NIGHT_BASE)
    base = np.where(is_weekend,  base * WEEKEND_SCALE, base)

    step_noise = rng.uniform(0.85, 1.15, size=len(df))
    spikes     = np.where(
        rng.random(size=len(df)) > 0.98,
        rng.uniform(5.0, 15.0, size=len(df)),
        0.0,
    )

    load = (base * day_mult * step_noise + spikes).round(2)
    return pd.Series(load, index=df.index, name='Load')


if __name__ == '__main__':
    print(f"Reading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    print(f"  Before — min: {df['Load'].min():.2f}  max: {df['Load'].max():.2f}"
          f"  mean: {df['Load'].mean():.2f}  unique: {df['Load'].nunique()}")

    df['Load'] = generate_load(df)

    print(f"  After  — min: {df['Load'].min():.2f}  max: {df['Load'].max():.2f}"
          f"  mean: {df['Load'].mean():.2f}  unique: {df['Load'].nunique()}")

    print("\nMean Load by hour:")
    by_hour = df.groupby(pd.to_datetime(df['timestamp']).dt.hour)['Load'].mean().round(1)
    for h, v in by_hour.items():
        bar = '█' * int(v / 3)
        print(f"  {h:02d}h  {v:5.1f} kW  {bar}")

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df.to_csv(DATA_PATH, index=False)
    print(f"\nSaved → {DATA_PATH}")

    if ALSO_UPDATE.exists():
        df.to_csv(ALSO_UPDATE, index=False)
        print(f"Saved → {ALSO_UPDATE}")

    print("\nNext: python normalize.py --input dataset_final.csv"
          " --output dataset_normalized.csv --scalers models/scalers.pkl")
