import pandas as pd

df = pd.read_csv("sin_cos_dataset.csv")
col = pd.read_csv("grid.csv")
df["Grid"] = col
df.to_csv("dataset_with_grid.csv", index=False)