import pandas as pd
from datetime import datetime

input_file = "kwh_coef.csv"
output_file = "kwh_coef.csv"

final_cols = [
    'Date', 'Month', 'Day', 'Hour', 'global_tilted_irradiance_instant'
]

df = pd.read_csv(input_file)

df['date'] = pd.to_datetime(df['date'])

df['Month'] = df['date'].dt.month

df['Day'] = df['date'].dt.day

df['Hour'] = df['date'].dt.hour

df['Date'] = df['Month'].astype(str) + " - " + df['Day'].astype(str)

df_final = df[final_cols].copy()

df_final.to_csv(output_file, index=False)



