import pandas as pd

df1 = pd.read_csv("../datasets_v2/DAM.csv")
df2 = pd.read_csv("../datasets_v2/IDM.csv")
df3 = pd.read_csv("../datasets_v2/switched.csv")

df = pd.merge(df1, df2, on=["Date", "Month", "Day", "Day_of_week", "Hour"], how="inner")
df = pd.merge(df, df3, on=["Date", "Month", "Day", "Day_of_week", "Hour"], how="inner")

df.to_csv("dataset.csv", index=False)