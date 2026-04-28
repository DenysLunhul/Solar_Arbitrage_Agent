import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")

def make_cyclical(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

df = make_cyclical(df, "Month", 12)
df = make_cyclical(df, "Day", 31)

df.to_csv("dataset_final.csv", index=False)