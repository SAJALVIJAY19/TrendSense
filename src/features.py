# src/features.py
# Step 2: Engineer technical indicator features from OHLCV data

import pandas as pd
import numpy as np
import ta  # Technical Analysis library


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10+ technical indicators to the OHLCV DataFrame.
    
    Features added:
    - RSI (14)            : Momentum oscillator
    - MACD + Signal       : Trend following momentum
    - Bollinger Bands     : Volatility bands
    - EMA 9 & 21          : Short/Long term trend
    - ATR                 : Volatility measure
    - OBV                 : Volume momentum
    - Stochastic %K, %D   : Momentum oscillator
    - ROC                 : Rate of change / momentum
    """
    df = df.copy()

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # --- RSI ---
    df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # --- MACD ---
    macd_obj = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Diff"]   = macd_obj.macd_diff()   # histogram

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["BB_High"]  = bb.bollinger_hband()
    df["BB_Low"]   = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()  # band width = volatility proxy
    df["BB_Pct"]   = bb.bollinger_pband()  # where price sits inside bands (0-1)

    # --- EMA ---
    df["EMA_9"]  = ta.trend.EMAIndicator(close=close, window=9).ema_indicator()
    df["EMA_21"] = ta.trend.EMAIndicator(close=close, window=21).ema_indicator()
    df["EMA_Cross"] = df["EMA_9"] - df["EMA_21"]   # positive = bullish crossover

    # --- ATR (Average True Range) ---
    df["ATR"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # --- OBV (On Balance Volume) ---
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()

    # --- Stochastic Oscillator ---
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # --- Rate of Change ---
    df["ROC"] = ta.momentum.ROCIndicator(close=close, window=10).roc()

    # --- Price-based features ---
    df["Daily_Return"]   = close.pct_change()          # today's % return
    df["Price_Range"]    = (high - low) / close        # intraday volatility
    df["Close_to_Open"]  = (close - df["Open"]) / df["Open"]

    return df


def create_labels(df: pd.DataFrame, n_days: int = 5, up_thresh: float = 0.01, down_thresh: float = -0.01) -> pd.DataFrame:
    """
    Create target labels based on N-day forward returns.
    
    Labels:
        1  = UP   (forward return > +1%)
        0  = NEUTRAL
       -1  = DOWN (forward return < -1%)
    
    Args:
        df:           DataFrame with Close prices
        n_days:       How many days ahead to predict
        up_thresh:    Minimum return to label as UP
        down_thresh:  Maximum return to label as DOWN
    """
    df = df.copy()

    # Calculate forward return: what will price be N days from now?
    df["Forward_Return"] = df["Close"].shift(-n_days) / df["Close"] - 1

    def label_row(ret):
        if pd.isna(ret):
            return np.nan
        elif ret > up_thresh:
            return 1     # UP
        elif ret < down_thresh:
            return -1    # DOWN
        else:
            return 0     # NEUTRAL

    df["Label"] = df["Forward_Return"].apply(label_row)
    return df


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup: drop NaN rows (from indicator warmup period),
    drop raw OHLCV and forward return columns, keep only features + label.
    """
    df = df.copy()

    # Columns to drop (not features)
    drop_cols = ["Open", "High", "Low", "Close", "Volume", 
                 "Adj Close", "Forward_Return"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Drop rows with any NaN (indicator warmup period = first ~30 rows)
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    # Quick test
    import yfinance as yf
    raw = yf.download("RELIANCE.NS", start="2020-01-01", end="2024-12-31", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = add_technical_indicators(raw)
    df = create_labels(df, n_days=5)
    df = prepare_feature_matrix(df)

    print(f"Feature matrix shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLabel distribution:\n{df['Label'].value_counts()}")
