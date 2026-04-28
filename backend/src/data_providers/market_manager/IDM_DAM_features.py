import time

import pandas as pd
from io import BytesIO
import requests
from datetime import datetime, timedelta, date


def fetch_DAM(today: date) -> pd.DataFrame:
    # IDM_RENAME = {
    #     "Година": "Hour",
    #     "Ціна, грн/МВт.год": "IDM_Price",
    #     "Остання ціна, грн/МВт.год": "IDM_Last_Price",
    #     "Обсяг продажу, МВт.год": "IDM_Vol_Sale",
    #     "Обсяг купівлі, МВт.год": "IDM_Vol_Buy"
    # }

    DAM_RENAME = {
        "Година": "Hour",
        "Ціна, грн/МВт.год": "DAM_Price",
        "Обсяг продажу, МВт.год": "DAM_Vol_Sale",
        "Обсяг купівлі, МВт.год": "DAM_Vol_Buy"
    }


    current_date = today + timedelta(days=1)
    date_structure = current_date.strftime("%d.%m.%Y")
    file_date_str = current_date.strftime("%Y-%m-%d")
    # url = f"https://www.oree.com.ua/index.php/PXS/downloadxlsx/{date_structure}/IDM/2"
    # print(f"Downloading IDM data for {file_date_str}...")
    #
    # IDM_drop_columns = ["Мінімальна ціна, грн/МВт.год","Максимальна ціна, грн/МВт.год",
    #                     "Заявлений обсяг продажу, МВт.год","Заявлений обсяг купівлі, МВт.год"]
    DAM_drop_columns = ["Заявлений обсяг продажу, МВт.год","Заявлений обсяг купівлі, МВт.год"]

    # # try:
    # #     response = requests.get(url, timeout=30)
    # #     if response.status_code == 200:
    # #         df = pd.read_excel(BytesIO(response.content), engine="calamine", header=0)
    # #         csv_file_name = "IDM.csv"
    # #         df = df.drop(columns=IDM_drop_columns)
    # #         df.rename(columns=IDM_RENAME)
    # #         df.to_csv(csv_file_name, index=False, encoding='utf-8-sig')
    # #         print(f"Saved file: {csv_file_name}")
    # # except Exception:
    # #     print("Something went wrong!")
    # #
    # # time.sleep(2)

    url = f"https://www.oree.com.ua/index.php/PXS/downloadxlsx/{date_structure}/DAM/2"
    print(f"Downloading DAM data for {file_date_str}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_excel(BytesIO(response.content), engine="calamine", header=0)
            df = df.drop(columns=DAM_drop_columns)
            df.rename(columns=DAM_RENAME)
            return df
    except Exception:
        return None