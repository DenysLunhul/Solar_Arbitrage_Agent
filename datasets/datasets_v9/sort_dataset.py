import pandas as pd

df = pd.read_csv("dataset.csv")
df = df.sort_values(["Month", "Day", "Hour", "Minute"], ascending=True)
df.to_csv("sorted.csv", index=False)