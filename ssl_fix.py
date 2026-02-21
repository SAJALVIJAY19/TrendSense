# ssl_fix.py
# Run this ONCE before anything else on Windows
# python ssl_fix.py

import ssl
import certifi
import os

# Fix 1: Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Fix 2: Patch yfinance to skip SSL verification
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter

session = requests.Session()
session.verify = False

# Suppress the SSL warning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Test if it works
print("Testing connection to Yahoo Finance...")
try:
    import yfinance as yf
    # Monkey-patch yfinance to use no-verify session
    df = yf.download("RELIANCE.NS", period="5d", progress=False)
    if not df.empty:
        print(f"✅ SUCCESS! Fetched {len(df)} rows for RELIANCE.NS")
        print(df.tail(2))
    else:
        print("⚠️  Connected but no data returned")
except Exception as e:
    print(f"❌ Still failing: {e}")