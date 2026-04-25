import pandas as pd

df = pd.read_csv("../datasets_v1/IDM.csv")
df.drop(columns=["Decl_Sale", "Decl_Buy", "Min_Price", "Max_Price"], inplace=True)
df.to_csv("IDM.csv")