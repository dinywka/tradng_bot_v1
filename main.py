import pandas as pd
import requests

# https://www.coingecko.com/api/documentation

# url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"
# r = requests.get(url)
# data = r.json()
# print(data)

url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=CNY&apikey=YUXUQAYZGSDRUN85'
r = requests.get(url)
data = r.json()

print(data)