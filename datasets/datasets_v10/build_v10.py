import numpy as np
import pandas as pd

df = pd.read_csv("../datasets_v9/sorted.csv")

df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 31)

# Insert after Day_of_week_cos
insert_at = df.columns.get_loc("Day_of_week_cos") + 1
cols = list(df.columns)
cols.remove("Day_sin")
cols.remove("Day_cos")
cols.insert(insert_at, "Day_sin")
cols.insert(insert_at + 1, "Day_cos")
df = df[cols]

df.to_csv("dataset.csv", index=False)