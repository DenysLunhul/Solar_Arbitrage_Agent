import pandas as pd
import numpy as np
from datetime import date, timedelta

df = pd.read_csv("../datasets_v8/dataset.csv")
gti = pd.read_csv("../datasets_v6/GTI.csv")

# ── 1. Fix Hour=24 → Hour=0 of the next day ──────────────────────────────────
mask_24 = df["Hour"] == 24
df.loc[mask_24, "Hour"] = 0

for idx in df[mask_24].index:
    d = date(2025, int(df.loc[idx, "Month"]), int(df.loc[idx, "Day"])) + timedelta(days=1)
    df.loc[idx, "Month"] = d.month
    df.loc[idx, "Day"] = d.day
    df.loc[idx, "Day_of_week"] = d.isoweekday()
    df.loc[idx, "Date"] = f"{d.month} - {d.day}"

# ── 2. Shift GTI / Temperature by +1 hour (4 rows of 15-min intervals) ───────
# Currently dataset row i uses GTI row i (wrong: GTI starts at 00:00, dataset at 01:00).
# Correct: dataset row i should use GTI row i+4 (same clock hour as the dataset row).
gti_vals = gti[["global_tilted_irradiance_instant", "temperature_2m"]].copy()
gti_shifted = gti_vals.shift(-4).ffill()  # last 4 rows repeat the last available GTI hour

df["global_tilted_irradiance_instant"] = gti_shifted["global_tilted_irradiance_instant"].values
df["Temperature_2m"] = gti_shifted["temperature_2m"].values

# ── 3. Recalculate all cyclical (sin/cos) features ───────────────────────────
# Formulas:  sin(2π * value / period),  cos(2π * value / period)
def make_cyclical(df, col, max_val):
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df

df = make_cyclical(df, "Hour", 24)
df = make_cyclical(df, "Minute", 60)
df = make_cyclical(df, "Day_of_week", 7)

df.to_csv("dataset.csv", index=False)
print(f"Saved {len(df)} rows to dataset.csv")
