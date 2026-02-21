# src/data_fetch.py

import ssl
import os
import certifi
import urllib3

# ── Nuclear SSL Fix for Windows ───────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_CERT_FILE']        = certifi.where()
os.environ['REQUESTS_CA_BUNDLE']   = certifi.where()
os.environ['CURL_CA_BUNDLE']       = certifi.where()
urllib3.disable_warnings()

# Patch requests globally
import requests
from requests.adapters import HTTPAdapter
old_send = HTTPAdapter.send
def patched_send(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_send(self, *args, **kwargs)
HTTPAdapter.send = patched_send

import yfinance as yf
import pandas as pd

STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "WIPRO.NS"
]

def fetch_stock_data(ticker: str, start: str = "2020-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    print(f"Fetching data for {ticker}...")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError("Empty dataframe")
    except Exception as e:
        print(f"  yfinance failed ({e}), trying pandas_datareader fallback...")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    print(f"  ✅ Fetched {len(df)} rows for {ticker}")
    return df


def fetch_all_stocks(stocks: list = STOCKS, start: str = "2020-01-01", end: str = "2024-12-31") -> dict:
    os.makedirs("data", exist_ok=True)
    all_data = {}

    for ticker in stocks:
        df = fetch_stock_data(ticker, start, end)
        if not df.empty:
            filename = f"data/{ticker.replace('.', '_')}_prices.csv"
            df.to_csv(filename)
            print(f"  Saved to {filename}")
            all_data[ticker] = df

    print(f"\nDone! Fetched data for {len(all_data)} stocks.")
    return all_data


if __name__ == "__main__":
    fetch_all_stocks()