import pandas as pd
 
df = pd.read_csv('dataset_final.csv')
 
grid_zeros = (df['Grid'] == 0).sum()
outage_nonzero = (df['outage_remaining_h'] != 0).sum()
 
print(f"Grid == 0:                  {grid_zeros}")
print(f"outage_remaining_h != 0:   {outage_nonzero}")
print(f"Різниця:                   {grid_zeros - outage_nonzero}")
 
if grid_zeros == outage_nonzero:
    print("\nОК — значення співпадають")
else:
    print("\nПОМИЛКА — значення не співпадають")
 