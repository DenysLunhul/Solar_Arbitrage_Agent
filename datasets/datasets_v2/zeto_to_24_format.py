import pandas as pd

df1 = pd.read_csv("DAM.csv")
df2 = pd.read_csv("IDM.csv")
df3 = pd.read_csv("switched.csv")

df3["Hour"] = df3["Hour"].replace(0, 24)
df3.to_csv("switched_0_to_24.csv", index=False)