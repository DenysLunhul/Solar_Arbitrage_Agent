import pandas as pd

df = pd.read_csv("../datasets_v5/dataset_with_grid.csv")
gti = pd.read_csv("GTI.csv")
df["Temperature_2m"] = gti["temperature_2m"]
df["global_tilted_irradiance_instant"] = gti["global_tilted_irradiance_instant"]

df.to_csv("dataset.csv", index=False)