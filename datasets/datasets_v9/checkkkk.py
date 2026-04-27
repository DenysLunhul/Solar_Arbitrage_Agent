import pandas as pd
import numpy as np

df = pd.read_csv('sorted.csv')

LOOKAHEAD_LIMIT = 24 * 4 

hours_till = []
durations = []

for i in range(len(df)):
    future_grid = df['Grid'].iloc[i : i + LOOKAHEAD_LIMIT].values
    
    zero_indices = np.where(future_grid == 0)[0]
    if len(zero_indices) > 0:
        steps_to_zero = zero_indices[0]
        hours_till.append(steps_to_zero * 0.25)
    else:
        hours_till.append(24.0)
        
    if len(zero_indices) > 0:
        start_idx = zero_indices[0]

        count = 0
        for j in range(i + start_idx, len(df)):
            if df['Grid'].iloc[j] == 0:
                count += 1
            else:
                break
        durations.append(count * 0.25)
    else:
        durations.append(0.0)

df['hours_till_outage'] = hours_till
df['next_outage_duration'] = durations

df.to_csv('sorted.csv', index=False)