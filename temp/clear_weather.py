import pandas as pd

input_file = "weather.csv"
output_file = "cleared_weather.csv"

df = pd.read_csv(input_file)



df['date'] = pd.to_datetime(df['date'], format= '%Y-%m-%d')

df['Month'] = df['date'].dt.month
df['Day'] = df['date'].dt.day
df['Hour'] = df['hour'].str.split(':').str[0].astype(int)
df['Temperature_2m'] = df['temperature_2m']
df['Shortwave_Radiation'] = df['shortwave_radiation']

cols_to_keep = [
    'Month', 'Day', 'Hour', 'Temperature_2m', 'Shortwave_Radiation'
]

df_final = df[cols_to_keep].copy()

print(df_final.dtypes)

df_final.to_csv(output_file, index=False)


