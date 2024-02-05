

import alpaca_trade_api as tradeapi
import datetime
from sklearn.preprocessing import MinMaxScaler

# Replace 'your_api_key' and 'your_api_secret' with your actual Alpaca API key and secret
api_key = 'PKERXMVY0AVJ8CVIDN2N'
api_secret = 'VyJp01gStMc7DA1fYAcXbLMCd4NBfNl0MXIQQDSg'
base_url = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading

# Instantiate the Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

# Define the stock symbol and time range
symbol = 'YHOO'
start_date = datetime.datetime.now() - datetime.timedelta(days=365)  # 1 year ago
end_date = datetime.datetime.now()

# Get historical stock data
try:
    historical_data = api.get_barset(symbol, 'day', limit=252, start=start_date, end=end_date).df[symbol]
except Exception as e:
    print(f"Error getting historical data for {symbol}: {e}")
    historical_data = None

# Check if historical data is available
if historical_data is not None and not historical_data.empty:
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(historical_data[['close']])

    # Continue with the rest of your code, for example:
    # X, y = prepare_data(pd.DataFrame(data_scaled, columns=['Close']), 'Close', look_back)
    # ... rest of your code
else:
    print(f"No valid historical data for {symbol}")



