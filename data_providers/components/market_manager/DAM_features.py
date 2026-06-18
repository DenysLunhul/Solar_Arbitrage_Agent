import pandas as pd
from io import BytesIO
import requests
from datetime import timedelta, date


def fetch_DAM(today: date) -> pd.DataFrame:
    DAM_RENAME = {
        "Ціна, грн/МВт.год": "DAM_Price",
        "Обсяг купівлі, МВт.год": "DAM_Vol_Buy",
        "Обсяг продажу, МВт.год": "DAM_Vol_Sale"
    }
    current_date = today + timedelta(days=1)
    date_structure = current_date.strftime("%d.%m.%Y")
    file_date_str = current_date.strftime("%Y-%m-%d")
    DAM_drop_columns = ["Заявлений обсяг продажу, МВт.год","Заявлений обсяг купівлі, МВт.год",
                        "Година"]
    url = f"https://www.oree.com.ua/index.php/PXS/downloadxlsx/{date_structure}/DAM/2"
    print(f"Downloading DAM data for {file_date_str}...")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_excel(BytesIO(response.content), engine="calamine", header=0)
            df = df.drop(columns=DAM_drop_columns)
            df = df.loc[df.index.repeat(4)].reset_index(drop=True)
            df = df.rename(columns=DAM_RENAME)
            for col in DAM_RENAME.values():
                df[col] = (
                    df[col].astype(str)
                    .str.replace('\xa0', '', regex=False)
                    .str.replace(' ', '', regex=False)
                    .str.replace(',', '.', regex=False)
                    .astype(float)
                )
            return df
    except Exception:
        return None