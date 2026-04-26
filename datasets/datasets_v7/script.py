import pandas as pd
import numpy as np


input_file = "dataset_with_grid.csv"
output_file = "dataset_with_grid_and_outgages.csv"


df = pd.read_csv(input_file)


risk_map = df.groupby('Month')['Grid'].apply(lambda x: (x == 0).mean())
df['Outage_Risk'] = df['Month'].map(risk_map)

df['block_id'] = (df['Grid'] != df['Grid'].shift()).cumsum()

df['outage_remaining_h'] = 0.0

for block_id, block in df[df['Grid'] == 0].groupby('block_id'):
    
    n = len(block)
    
    values = [(n - i) * 0.25 for i in range(n)]
    
    df.loc[block.index, 'outage_remaining_h'] = values

df.to_csv(output_file, index=False)
