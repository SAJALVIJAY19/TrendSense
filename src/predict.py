# src/predict.py
# Step 6: Inference pipeline — get live prediction for any stock ticker

import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf

from features import add_technical_indicators
from sentiment import get_live_sentiment
from scraper import get_headlines_for_ticker

LABEL_MAP   = {1: "UP", 0: "NEUTRAL", -1: "DOWN"}
EMOJI_MAP   = {"UP": "📈", "NEUTRAL": "➡️", "DOWN": "📉"}
COLOR_MAP   = {"UP": "green", "NEUTRAL": "gray", "DOWN": "red"}


def load_artifacts():
    """Load saved model, scaler, and feature names."""
    model_path   = "models/best_model.pkl"
    scaler_path  = "models/scaler.pkl"
    features_path = "models/feature_names.pkl"
    results_path  = "models/results.json"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        raise FileNotFoundError(
            "Model files not found! Run src/train.py first."
        )

    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)

    return model, scaler, feature_names


def get_latest_features(ticker: str, feature_names: list) -> pd.Series:
    """
    Fetch the most recent trading day's features for a ticker.
    
    Args:
        ticker:        e.g. 'RELIANCE.NS'
        feature_names: list of feature column names from training
    
    Returns:
        Series of feature values for the latest available day
    """
    # Fetch last 100 days to have enough for indicator warmup
    df = yf.download(ticker, period="100d", progress=False)
    if df.empty:
        raise ValueError(f"Could not fetch data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = add_technical_indicators(df)

    # Add sentiment (scrape fresh headlines)
    headlines = get_headlines_for_ticker(ticker, max_articles=20)
    sentiment = get_live_sentiment(ticker, headlines)
    df["Sentiment"] = sentiment

    # Drop OHLCV cols
    drop_cols = ["Open", "High", "Low", "Close", "Volume",
                 "Adj Close", "Forward_Return", "Label", "Daily_Return"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(f"Not enough data after feature engineering for {ticker}")

    # Get the latest row
    latest = df.iloc[-1]

    # Align to training feature names — fill missing with 0
    aligned = pd.Series(0.0, index=feature_names)
    for col in feature_names:
        if col in latest.index:
            aligned[col] = latest[col]

    return aligned, sentiment


def predict_ticker(ticker: str, model_name: str = "best") -> dict:
    """
    Generate a prediction for a given stock ticker.
    
    Args:
        ticker:     Stock symbol e.g. 'RELIANCE.NS'
        model_name: 'best' or specific model name
    
    Returns:
        Dictionary with prediction details
    """
    # Load artifacts
    if model_name == "best":
        model, scaler, feature_names = load_artifacts()
    else:
        import json
        scaler        = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        model_file    = f"models/{model_name.replace(' ', '_').lower()}.pkl"
        model         = joblib.load(model_file)

    # Get features
    print(f"\nFetching latest features for {ticker}...")
    features, sentiment = get_latest_features(ticker, feature_names)

    # Scale (most models need this)
    features_scaled = scaler.transform([features.values])

    # Predict
    pred_label = model.predict(features_scaled)[0]
    pred_text  = LABEL_MAP[pred_label]

    # Confidence
    confidence = None
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(features_scaled)[0]
        classes    = model.classes_
        conf_dict  = {LABEL_MAP[c]: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
        confidence = max(proba) * 100
    else:
        conf_dict  = {}
        confidence = None

    result = {
        "ticker"          : ticker,
        "prediction"      : pred_text,
        "emoji"           : EMOJI_MAP[pred_text],
        "confidence_pct"  : round(confidence, 1) if confidence else None,
        "class_probs"     : conf_dict,
        "sentiment_score" : round(sentiment, 3),
        "sentiment_label" : "Positive" if sentiment > 0.05 else ("Negative" if sentiment < -0.05 else "Neutral"),
        "top_features"    : dict(zip(feature_names[:5], features.values[:5].round(4).tolist())),
    }

    print(f"\n{'='*45}")
    print(f"  Ticker     : {ticker}")
    print(f"  Prediction : {pred_text} {EMOJI_MAP[pred_text]}")
    if confidence:
        print(f"  Confidence : {confidence:.1f}%")
    print(f"  Sentiment  : {sentiment:.3f} ({result['sentiment_label']})")
    print(f"{'='*45}")

    return result


if __name__ == "__main__":
    # Test prediction
    result = predict_ticker("RELIANCE.NS")
    print("\nFull result:", result)
