# src/scraper.py
# Step 3: Scrape financial news headlines using BeautifulSoup

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import os

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Map of ticker → search keyword for news
TICKER_KEYWORDS = {
    "RELIANCE.NS" : "Reliance Industries",
    "TCS.NS"      : "TCS Tata Consultancy",
    "INFY.NS"     : "Infosys",
    "HDFCBANK.NS" : "HDFC Bank",
    "WIPRO.NS"    : "Wipro"
}


def scrape_moneycontrol(keyword: str, max_articles: int = 30) -> list:
    """
    Scrape news headlines from MoneyControl search results.
    
    Args:
        keyword:      Company name to search
        max_articles: Max number of headlines to fetch
    
    Returns:
        List of headline strings
    """
    headlines = []
    try:
        search_url = f"https://www.moneycontrol.com/news/tags/{keyword.replace(' ', '-').lower()}.html"
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # MoneyControl news list items
        articles = soup.find_all("li", class_="clearfix")
        for article in articles[:max_articles]:
            h2 = article.find("h2")
            if h2 and h2.text.strip():
                headlines.append(h2.text.strip())

    except Exception as e:
        print(f"  MoneyControl scrape failed for '{keyword}': {e}")

    return headlines


def scrape_economic_times(keyword: str, max_articles: int = 30) -> list:
    """
    Scrape news headlines from Economic Times search.
    
    Args:
        keyword:      Company name to search
        max_articles: Max headlines to return
    
    Returns:
        List of headline strings
    """
    headlines = []
    try:
        search_url = f"https://economictimes.indiatimes.com/topic/{keyword.replace(' ', '-').lower()}"
        response = requests.get(search_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # ET uses these tags for story listings
        for tag in ["h3", "h2"]:
            items = soup.find_all(tag)
            for item in items[:max_articles]:
                text = item.get_text(strip=True)
                if len(text) > 20:   # filter out short nav items
                    headlines.append(text)
            if headlines:
                break

    except Exception as e:
        print(f"  Economic Times scrape failed for '{keyword}': {e}")

    return headlines


def get_headlines_for_ticker(ticker: str, max_articles: int = 50) -> list:
    """
    Get combined headlines from both sources for a ticker.
    
    Args:
        ticker:       Stock ticker e.g. 'RELIANCE.NS'
        max_articles: Total max headlines
    
    Returns:
        Deduplicated list of headlines
    """
    keyword = TICKER_KEYWORDS.get(ticker, ticker.replace(".NS", ""))
    print(f"  Scraping news for {ticker} ({keyword})...")

    mc_headlines = scrape_moneycontrol(keyword, max_articles // 2)
    time.sleep(1)  # be polite to the server
    et_headlines = scrape_economic_times(keyword, max_articles // 2)

    combined = list(set(mc_headlines + et_headlines))  # deduplicate
    print(f"  Got {len(combined)} headlines for {ticker}")
    return combined


def scrape_all_tickers(tickers: list = None) -> dict:
    """
    Scrape headlines for all tickers and save to CSV.
    
    Returns:
        Dict of {ticker: [headlines]}
    """
    if tickers is None:
        tickers = list(TICKER_KEYWORDS.keys())

    os.makedirs("data", exist_ok=True)
    all_headlines = {}

    for ticker in tickers:
        headlines = get_headlines_for_ticker(ticker)
        all_headlines[ticker] = headlines

        # Save to CSV
        df = pd.DataFrame({
            "ticker"   : ticker,
            "headline" : headlines,
            "scraped_at": datetime.now().strftime("%Y-%m-%d")
        })
        filename = f"data/{ticker.replace('.', '_')}_headlines.csv"
        df.to_csv(filename, index=False)
        print(f"  Saved {len(headlines)} headlines to {filename}")
        time.sleep(2)  # avoid rate limiting

    return all_headlines


if __name__ == "__main__":
    scrape_all_tickers()
