import pandas as pd

KEY_COLS = ["Date", "Month", "Day", "Day_of_week", "Hour"]

df1 = pd.read_csv("../datasets_v2/DAM.csv")
df2 = pd.read_csv("../datasets_v2/IDM.csv")
df3 = pd.read_csv("../datasets_v2/switched_0_to_24.csv")

for frame in (df1, df2, df3):
	frame["Hour"] = pd.to_numeric(frame["Hour"], errors="coerce").astype("Int64")

base = pd.merge(df1, df2, on=KEY_COLS, how="inner")
df = pd.merge(base, df3, on=KEY_COLS, how="left")

weather_cols = [col for col in df3.columns if col not in KEY_COLS]
missing_weather = df[weather_cols].isna().all(axis=1)

if missing_weather.any():
	# DST fallback: DAM/IDM can contain Hour=25, while weather has only up to Hour=24.
	dst_missing = missing_weather & (df["Hour"] == 25)
	if dst_missing.any():
		day_key = ["Date", "Month", "Day", "Day_of_week"]
		fallback = df3[df3["Hour"] == 24][day_key + weather_cols].rename(
			columns={col: f"{col}__fallback" for col in weather_cols}
		)
		df = pd.merge(df, fallback, on=day_key, how="left")
		for col in weather_cols:
			df.loc[dst_missing, col] = df.loc[dst_missing, col].fillna(df.loc[dst_missing, f"{col}__fallback"])
		df.drop(columns=[f"{col}__fallback" for col in weather_cols], inplace=True)
		missing_weather = df[weather_cols].isna().all(axis=1)

if missing_weather.any():
	missing_keys = df.loc[missing_weather, KEY_COLS].head(5).to_dict(orient="records")
	raise ValueError(f"Missing weather rows after alignment: {missing_weather.sum()}, sample keys: {missing_keys}")

df.to_csv("dataset_temp.csv", index=False)