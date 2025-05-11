import yfinance as yf
import pandas as pd

# Download QQQ data
ticker = "QQQ"
data = yf.download(ticker, start="2010-01-01", interval="1h")

# Ensure NY Time (yfinance usually provides this, but double-check and convert if necessary)
# data.index = data.index.tz_convert('America/New_York') # If conversion needed

# Save to CSV
data.to_csv("nasdaq_h1_2010_present.csv")
print("Data downloaded and saved.")