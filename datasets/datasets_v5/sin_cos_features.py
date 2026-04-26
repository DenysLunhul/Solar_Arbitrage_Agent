import pandas as pd
import numpy as np

df = pd.read_csv("expanded.csv")

def make_cyclical(df, col, max_val):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    return df

df = make_cyclical(df, "Hour", 24)
df = make_cyclical(df, "Minute", 60)
df = make_cyclical(df, "Day_of_week", 7)


df.to_csv("sin_cos_dataset.csv", index=False)