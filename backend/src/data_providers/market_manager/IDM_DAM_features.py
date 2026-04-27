import pandas as pd
from io import BytesIO
import time
import requests
from datetime import date, timedelta
from pathlib import Path




def get_tomorrow_operator_market_data(today):
    tomorrow = today + timedelta(days=1)
    date_structure = tomorrow.strftime("%d.%m.%Y")
    file_date_str = tomorrow.strftime("%Y-%m-%d")
    url = f"https://www.oree.com.ua/index.php/PXS/downloadxlsx/{date_structure}/DAM/2"
    print(f"Downloading IDM data for {file_date_str}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_excel(BytesIO(response.content), engine="calamine", header=0)
            columns_to_drop = [
                "Заявлений обсяг продажу, МВт.год", "Заявлений обсяг купівлі, МВт.год",
            ]
            df = df.drop(columns=columns_to_drop)
            df.to_csv("IDM.csv", index=False)
            print(f"Saved file: IDM.csv")
    except Exception:
        pass


if __name__ == "__main__":
    today = date.today()
    df = get_tomorrow_operator_market_data(today)
    df.to_csv("schedule.csv", index=False)