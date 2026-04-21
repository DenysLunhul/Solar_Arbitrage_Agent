import requests
from bs4 import BeautifulSoup

url = "https://www.oree.com.ua/index.php/main/get_amount"

payload = { 
    'day' : '22.04.2026',
    'day_from' : '22.04.2026',
    'day_to' : '22.04.2026',
    'month' : '04.2026',
    'type' : 'day'
}

headers = {
    'User-Agent': 'Mozilla/5.0',
    'X-Requested-With': 'XMLHttpRequest',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
}

response = requests.post(url, data = payload, headers = headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    raw_text = soup.get_text(separator=' --- ', strip=True)

    parts = raw_text.split(' --- ')
    
    prices = []
    for item in parts:
        try:
            price = float(item)
            prices.append(price)
        except ValueError:
            continue
            
    print(f"ціни: {prices}")
else:
    print(f"Помилка: {response.status_code}")

#ну тут я ваше не одупляю що ці циферки значать тому бб