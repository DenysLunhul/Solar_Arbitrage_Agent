import pandas as pd

df = pd.read_csv("../datasets_v1/DAM.csv")
df.drop(columns=["Decl_Sale", "Decl_Buy"], inplace=True)
df.to_csv("DAM.csv", index=False)