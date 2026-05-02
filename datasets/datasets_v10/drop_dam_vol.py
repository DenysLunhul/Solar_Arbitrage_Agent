import pandas as pd

df = pd.read_csv("dataset_no_idm.csv")

df = df.drop(columns=["DAM_Vol_Sale", "DAM_Vol_Buy"])

df.to_csv("dataset_final.csv", index=False)