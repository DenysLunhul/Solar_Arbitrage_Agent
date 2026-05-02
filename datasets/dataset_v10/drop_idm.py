import pandas as pd

df = pd.read_csv("dataset.csv")

idm_cols = [c for c in df.columns if "IDM" in c]
df = df.drop(columns=idm_cols)

df.to_csv("dataset_no_idm.csv", index=False)