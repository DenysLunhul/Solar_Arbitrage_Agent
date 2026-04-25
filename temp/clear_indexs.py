import pandas as pd

input_file = "FULL_DAM_indexs_and_avg_weighted_prices.csv"
final_file = "../datasets/datasets_v0/Prices.csv"

df = pd.read_csv(input_file)

df['Date_dt'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')

df['Month'] = df['Date_dt'].dt.month
df['Day'] = df['Date_dt'].dt.day
df['Dayofweek'] = df['Date_dt'].dt.dayofweek + 1

cols_map = {
    'Base, грн/МВт.год': 'Base',
    'Peak, грн/МВт.год': 'Peak',
    'OffPeak, грн/МВт.год': 'Offpeak',
    'Мінімальна ціна, грн/МВт.год': 'Min_Price',
    'Максимальна ціна, грн/МВт.год': 'Max_Price',
    'Середньозважена ціна, грн/МВт.год': 'Avg_Weighted_Price'
}

# 3. Чистка та конвертація
for old_name, new_name in cols_map.items():
    df[new_name] = (df[old_name].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float))

finals_cols = [
    'Month', 'Day', 'Dayofweek', 'Base', 'Peak', 'Offpeak', 
    'Min_Price', 'Max_Price', 'Avg_Weighted_Price'
]

df_final = df[finals_cols]

df_final.to_csv(final_file, index=False)

print(df_final.dtypes)
