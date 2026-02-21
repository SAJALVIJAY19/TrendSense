# src/sentiment.py
# Step 4: Score news headlines using VADER sentiment analyzer

import pandas as pd
import numpy as np
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER once (reuse across calls)
analyzer = SentimentIntensityAnalyzer()


def score_headline(headline: str) -> float:
    """
    Score a single headline using VADER.
    
    Returns:
        Compound score between -1.0 (very negative) and +1.0 (very positive)
        -1.0  = very bearish news
         0.0  = neutral
        +1.0  = very bullish news
    """
    scores = analyzer.polarity_scores(headline)
    return scores["compound"]


def score_headlines_list(headlines: list) -> float:
    """
    Score a list of headlines and return the average sentiment.
    
    Args:
        headlines: List of news headline strings
    
    Returns:
        Average compound score for the batch
    """
    if not headlines:
        return 0.0

    scores = [score_headline(h) for h in headlines]
    return float(np.mean(scores))


def load_and_score_headlines(ticker: str) -> pd.DataFrame:
    """
    Load scraped headlines CSV for a ticker, score each headline,
    and return a DataFrame with date + daily_sentiment columns.
    
    Args:
        ticker: e.g. 'RELIANCE.NS'
    
    Returns:
        DataFrame with columns: [scraped_at, daily_sentiment]
    """
    filename = f"data/{ticker.replace('.', '_')}_headlines.csv"

    if not os.path.exists(filename):
        print(f"  No headlines file found for {ticker}. Run scraper.py first.")
        return pd.DataFrame()

    df = pd.read_csv(filename)
    df["sentiment_score"] = df["headline"].apply(score_headline)

    # Aggregate: average sentiment per date
    daily = df.groupby("scraped_at")["sentiment_score"].mean().reset_index()
    daily.rename(columns={"scraped_at": "Date", "sentiment_score": "Sentiment"}, inplace=True)
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily.set_index("Date", inplace=True)

    print(f"  Sentiment for {ticker}: mean={daily['Sentiment'].mean():.3f}, "
          f"min={daily['Sentiment'].min():.3f}, max={daily['Sentiment'].max():.3f}")
    return daily


def merge_sentiment_with_features(features_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge daily sentiment scores into the feature matrix.
    Missing sentiment days are filled with 0.0 (neutral).
    
    Args:
        features_df: DataFrame indexed by Date with technical features
        ticker:      Stock ticker
    
    Returns:
        Feature DataFrame with added 'Sentiment' column
    """
    sentiment_df = load_and_score_headlines(ticker)

    if sentiment_df.empty:
        # No sentiment data — fill with neutral
        features_df["Sentiment"] = 0.0
        return features_df

    # Left join: keep all trading days, fill missing sentiment with 0
    merged = features_df.join(sentiment_df, how="left")
    merged["Sentiment"] = merged["Sentiment"].fillna(0.0)

    print(f"  Merged sentiment into features. Shape: {merged.shape}")
    return merged


def get_live_sentiment(ticker: str, headlines: list) -> float:
    """
    Score a fresh list of headlines for live prediction (used by FastAPI).
    
    Args:
        ticker:    Stock ticker (for logging)
        headlines: List of recent headlines
    
    Returns:
        Average sentiment score
    """
    score = score_headlines_list(headlines)
    print(f"  Live sentiment for {ticker}: {score:.3f}")
    return score


if __name__ == "__main__":
    # Quick demo
    test_headlines = [
        "Reliance Industries reports record quarterly profit, beats estimates",
        "RIL shares surge 5% after strong earnings announcement",
        "Reliance faces regulatory probe over pricing practices",
        "Market remains cautious ahead of RIL annual general meeting",
    ]
    for h in test_headlines:
        score = score_headline(h)
        sentiment = "BULLISH 📈" if score > 0.05 else ("BEARISH 📉" if score < -0.05 else "NEUTRAL ➡️")
        print(f"  [{score:+.3f}] {sentiment}  |  {h[:60]}")
