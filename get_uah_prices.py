import requests
from bs4 import BeautifulSoup

url = "https://www.oree.com.ua/index.php/main/get_uah_prices"

# Параметри у вкладці Network -> Payload щоб сервер прийняв тіпа за свого і видав норм інфу а не хуйню якусь з якою анріл працювати
payload = {
    'day': '20.04.2026',
    'month': '04.2026',
    'type': 'day'
}

# Прикинувся браузером мозілла фаєрфокс і тіпа роблю дефолтний запит 
headers = {
    'User-Agent': 'Mozilla/5.0',
    'X-Requested-With': 'XMLHttpRequest',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
}

#Це по факту сам дефолтний запит 
response = requests.post(url, data=payload, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    raw_text = soup.get_text(separator=' --- ', strip=True)
    
    # Розбив текст на список
    parts = raw_text.split(' --- ')
    
    prices = []
    for item in parts:
        try:
            # кастим ціну до флоата
            price = float(item)
            prices.append(price)
        except ValueError:
            continue
            
    print(f"ціни: {prices}")
else:
    print(f"Помилка: {response.status_code}")

n = len(prices)

#матриця цін
prices_structured = [[0.0 for i in range(3)] for i in range(n//3)]

k = 0

#заповнення цін по дням
#Base - 0 index Peak - 1 index Offpeak - 2 index
for i in range(n//3):
    for j in range(3):
        prices_structured[i][j] = prices[k]
        k += 1


for row in prices_structured:
    print(row)


