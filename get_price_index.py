import requests
from bs4 import BeautifulSoup

url = "https://www.oree.com.ua/index.php/main/get_price_indexes"

payload = { 
    'date' : '20.04.2026',
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

# там кароче перше це ціна вдень від 8 - 23 години 
# друге ціна за ніч 1 - 7 години 
# третє це пікові години зазвичай вечірні 
# четверте то напівпік? що би це не значило, 
# а ще є якісь проценти які в душі не єбу за що відповідають але я їх поки не спарсив