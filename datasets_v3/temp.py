import pandas as pd

df = pd.read_csv("dataset.csv")
print(df["Hour"].value_counts())