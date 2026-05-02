import pandas as pd

df = pd.read_csv("dataset_no_idm.csv")

df = df.rename(columns={
    "global_tilted_irradiance_instant": "Global_tilted_irradiance_instant",
    "Shortwave_Radiation": "Shortwave_radiation",
    "hours_till_outage": "hours_until_outage",
})

df["timestamp"] = pd.to_datetime({
    "year": 2025,
    "month": df["Month"],
    "day": df["Day"],
    "hour": df["Hour"],
    "minute": df["Minute"],
})

df = df[[
    "timestamp", "Month", "Day", "Hour", "Minute",
    "Hour_sin", "Hour_cos", "Minute_sin", "Minute_cos",
    "Day_of_week", "Day_of_week_sin", "Day_of_week_cos", "Day_sin", "Day_cos",
    "Grid", "next_outage_duration", "outage_remaining_h", "hours_until_outage",
    "Load", "Temperature_2m", "Shortwave_radiation", "Global_tilted_irradiance_instant",
    "DAM_Price", "DAM_Vol_Buy", "DAM_Vol_Sale"
]]

df.to_csv("dataset_final.csv", index=False)