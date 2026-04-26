import pandas as pd

df = pd.read_csv("../datasets_v7/dataset_final.csv")
df1 = pd.read_csv("../datasets_v6/dataset.csv")

df.drop(columns=["Temperature_2m", "global_tilted_irradiance_instant", "Shortwave_Radiation"], inplace=True)
df["global_tilted_irradiance_instant"] = df1["global_tilted_irradiance_instant"]
df["Temperature_2m"] = df1["Temperature_2m"]
df["Shortwave_Radiation"] = df1["Shortwave_Radiation"]
df.to_csv("dataset.csv", index=False)