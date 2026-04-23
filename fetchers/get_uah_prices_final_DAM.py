import pandas as pd
from io import BytesIO
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path


start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)

output_dir = Path(__file__).resolve().parent.parent / "datasets"
output_dir.mkdir(parents=True, exist_ok=True)

current_date = start_date

while current_date <= end_date:
    date_structure = current_date.strftime("%d.%m.%Y")
    file_date_str = current_date.strftime("%Y-%m-%d")

    url = f"https://www.oree.com.ua/index.php/PXS/downloadxlsx/{date_structure}/DAM/2"

    print(f"Downloading DAM data for {file_date_str}...")

    try:
        response = requests.get(url, timeout=30)

        response.raise_for_status()

        df = pd.read_excel(BytesIO(response.content), engine="calamine")
        csv_file_name = f"{file_date_str}_DAM.csv"
        csv_file_path = output_dir / csv_file_name
        df.to_csv(csv_file_path, index=False)
        print(f"Saved file: {csv_file_name}")
    except Exception:
        pass

    current_date += timedelta(days = 1)

    time.sleep(2)
