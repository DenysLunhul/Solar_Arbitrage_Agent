import pandas as pd

input_file = "final_full_dataset_IDM.csv" 
output_file = "../datasets/IDM.csv"

df = pd.read_csv(input_file)

df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Day_of_week'] = df['Date'].dt.dayofweek + 1
df['Hour'] = df['Година'].str.split(':').str[0].astype(int)

cols_map = {
    'Ціна, грн/МВт.год': 'Price',
    'Мінімальна ціна, грн/МВт.год' : 'Min_Price',
    'Максимальна ціна, грн/МВт.год' : 'Max_Price',
    'Остання ціна, грн/МВт.год' : 'Last_Price',
    'Обсяг продажу, МВт.год': 'Vol_Sale',
    'Обсяг купівлі, МВт.год': 'Vol_Buy',
    'Заявлений обсяг продажу, МВт.год': 'Decl_Sale',
    'Заявлений обсяг купівлі, МВт.год': 'Decl_Buy'
}

for old_name, new_name in cols_map.items():
    df[new_name] = df[old_name].str.replace(' ', '').str.replace(',', '.').astype(float)

final_columns = [
    'Month', 'Day', 'Day_of_week', 'Hour', 
    'Price', 'Min_Price', 'Max_Price', 'Last_Price', 'Vol_Sale', 'Vol_Buy', 'Decl_Sale', 'Decl_Buy'
]

df_final = df[final_columns]


df_final.to_csv(output_file, index=False)

print(df_final.head()) 

print(df_final.dtypes)
