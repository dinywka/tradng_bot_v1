# import pandas as pd
# import requests
#
# # https://www.coingecko.com/api/documentation
#
# # url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"
# # r = requests.get(url)
# # data = r.json()
# # print(data)
#
# url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=CNY&apikey=YUXUQAYZGSDRUN85'
# data = pd.read_csv(url)

import pandas as pd
import requests

url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=CNY&apikey=YUXUQAYZGSDRUN85'
response = requests.get(url)
data = response.json()

# Extract the time series data from the JSON
time_series_data = data['Time Series (Digital Currency Daily)']

# Create a DataFrame from the time series data
df = pd.DataFrame(time_series_data).T

# Display the DataFrame
print(df)
