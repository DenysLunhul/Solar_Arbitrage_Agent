import pandas as pd
import os

folder_name = "energy_dataset_csv_IDM"  #папка з якоъ файли будуть конкатенуватися
output_file = "final_full_dataset_IDM.csv"

files = [f for f in os.listdir(folder_name) if f.endswith('.csv')]
files.sort()

all_data = []

print(f"Знайдено файлів: {len(files)}")

for filename in files:
    file_path = os.path.join(folder_name, filename)

    date_part = filename.split('_')[0]
    
    try:
        df = pd.read_csv(file_path)

        df.insert(0, 'Date', date_part)
        
        all_data.append(df)
    except Exception as e:
        print(f"Помилка у файлі {filename}: {e}")

#Конкатенація
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    final_df.to_csv(output_file, index=False)
