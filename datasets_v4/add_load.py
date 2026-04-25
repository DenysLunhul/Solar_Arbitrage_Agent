import pandas as pd
import numpy as np

df = pd.read_csv("../datasets_v3/dataset.csv")

def generate_load_column(hour_col, day_of_week_col, base_day=350, base_night=60):
    hr = hour_col.values
    dw = day_of_week_col.values
    load = np.where((hr >= 8) & (hr <= 18), base_day, base_night)
    load = np.where(dw >= 6, load * 0.4, load)
    noise = np.random.uniform(0.85, 1.15, size=len(hr))
    load = load * noise
    spikes = np.where(np.random.rand(len(hr)) > 0.98, np.random.uniform(100, 200), 0)
    load = load + spikes
    return load.round(2)

df['Load'] = generate_load_column(df['Hour'], df['Day_of_week'])
df.to_csv("dataset.csv", index=False)