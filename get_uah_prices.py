import os
import xlrd
import pandas as pd
from io import BytesIO
import time
import requests
from datetime import datetime, timedelta


#межі скачування
start_date = datetime(2025, 1, 1)

end_date = datetime(2025, 12, 31)

folder_name = "energy_dataset_csv_final"

os.makedirs(folder_name)

current_date = start_date

while current_date <= end_date:

    date_structure = current_date.strftime("%d.%m.%Y")

    file_date_str = current_date.strftime("%Y-%m-%d")

    url = f"https://www.oree.com.ua/index.php/PXS/downloadxlsx/{date_structure}/DAM/2"

    file_name = f"{file_date_str}_DAM.xlsx"
    file_path = os.path.join(folder_name, file_name)

    try:
        response = requests.get(url)

        if response.status_code == 200:
                 df = pd.read_excel(BytesIO(response.content), engine='calamine')
                 csv_file_name = f"{file_date_str}_DAM.csv"
                 csv_file_path = os.path.join(folder_name, csv_file_name)
                 df.to_csv(csv_file_path, index=False)
                 print(f"Всьо чисто з кайфом неспешка с легкой іроніей скачалось: {csv_file_name}")
        else:
            print(f"Якщо язик йде по пізді то це добре, а якшо скрипт то це погано і якраз таки скрипт пішов по пизді")
    except Exception as e:
        print(f"Помилка для дати {date_structure}: {e}, ну і всьо пішло по пиздьонці такій немитій небритій вонючій прям фууууу")
    
    current_date += timedelta(days = 1)

    time.sleep(2)




