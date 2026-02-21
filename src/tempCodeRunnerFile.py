# src/train.py
import pandas as pd
import numpy as np
import os, sys, json, joblib, warnings
import ssl, urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report, confusion_matrix)
from xgboost import XGBClassifier
from features import add_technical_indicators, create_labels, prepare_feature_matrix
from sentiment import merge_sentiment_with_features
import yfinance as yf

warnings.filterwarnings("ignore")

STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS"]


def load_and_prepare(ticker):
    filename = f"data/{ticker.replace('.', '_')}_prices.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    else:
        df = yf.download(ticker, start="2020-01-01", end="2024-12-31", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    df = add_technical_indicators(df)
    df = create_labels(df, n_days=5)
    df = merge_sentiment_with_features(df, ticker)
    df = prepare_feature_matrix(df)
    df.dropna(subset=["Label"], inplace=True)
    df["Label"] = df["Label"].astype(int)
    return df


def build_combined_dataset():
    frames = []
    for ticker in STOCKS:
        print(f"\nPreparing {ticker}...")
        try:
            df = load_and_prepare(ticker)
            df["Ticker"] = ticker
            frames.append(df)
            print(f"  {ticker}: {len(df)} samples")
        except Exception as e:
            print(f"  ERROR on {ticker}: {e}")
    combined = pd.concat(frames, axis=0).sort_index()
    print(f"\nCombined dataset: {combined.shape}")
    print(f"Label distribution:\n{combined['Label'].value_counts()}")
    return combined


def time_split(df, test_ratio=0.2):
    df = df.drop(columns=["Ticker"], errors="ignore")
    split_idx = int(len(df) * (1 - test_ratio))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train = train.drop(columns=["Label"])
    y_train = train["Label"]
    X_test  = test.drop(columns=["Label"])
    y_test  = test["Label"]
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    os.makedirs("models", exist_ok=True)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    # ── XGBoost requires 0-indexed labels ──────────────────────────
    # Original: -1, 0, 1  →  Mapped: 0, 1, 2
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)   # -1→0, 0→1, 1→2
    y_test_enc  = le.transform(y_test)
    joblib.dump(le, "models/label_encoder.pkl")

    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, C=1.0, random_state=42), True),
        "SVM"                : (SVC(kernel="rbf", probability=True, random_state=42),       True),
        "KNN"                : (KNeighborsClassifier(n_neighbors=7, weights="distance"),    True),
        "Random Forest"      : (RandomForestClassifier(n_estimators=200, max_depth=10,
                                    min_samples_leaf=5, random_state=42, n_jobs=-1),        False),
        "XGBoost"            : (XGBClassifier(n_estimators=200, max_depth=6,
                                    learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, random_state=42,
                                    verbosity=0, eval_metric="mlogloss",
                                    num_class=3, objective="multi:softmax"),                False),
    }

    results   = {}
    best_f1   = -1
    best_name = ""

    print("\n" + "="*65)
    print(f"{'Model':<22} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("="*65)

    for name, (model, use_scaled) in models.items():
        Xtr = X_train_s if use_scaled else X_train.values
        Xte = X_test_s  if use_scaled else X_test.values

        # XGBoost uses encoded labels; others use original
        is_xgb  = (name == "XGBoost")
        ytr_use = y_train_enc if is_xgb else y_train
        yte_use = y_test_enc  if is_xgb else y_test

        model.fit(Xtr, ytr_use)
        raw_preds = model.predict(Xte)

        # Decode XGBoost predictions back to -1, 0, 1
        if is_xgb:
            preds = le.inverse_transform(raw_preds.astype(int))
        else:
            preds = raw_preds

        acc  = accuracy_score(y_test, preds)
        f1   = f1_score(y_test, preds, average="weighted", zero_division=0)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec  = recall_score(y_test, preds, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": round(acc, 4), "f1_score": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "use_scaled": use_scaled,
            "report": classification_report(y_test, preds,
                          target_names=["DOWN","NEUTRAL","UP"], zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }

        joblib.dump(model, f"models/{name.replace(' ','_').lower()}.pkl")

        if f1 > best_f1:
            best_f1, best_name = f1, name

        print(f"{name:<22} {acc:>10.2%} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f}")

    print("="*65)
    print(f"\n🏆  Best model: {best_name}  (F1 = {best_f1:.4f})")

    joblib.dump(joblib.load(f"models/{best_name.replace(' ','_').lower()}.pkl"),
                "models/best_model.pkl")

    feature_names = list(X_train.columns)
    joblib.dump(feature_names, "models/feature_names.pkl")

    results["best_model"]    = best_name
    results["feature_names"] = feature_names
    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ All models saved to models/")
    return results, best_name


def print_feature_importance(feature_names):
    try:
        rf = joblib.load("models/random_forest.pkl")
        feat_imp = sorted(zip(feature_names, rf.feature_importances_),
                          key=lambda x: x[1], reverse=True)
        print("\n📊 Top 10 Feature Importances (Random Forest):")
        for feat, imp in feat_imp[:10]:
            print(f"  {feat:<20} {imp:.4f}  {'█' * int(imp*100)}")
    except Exception as e:
        print(f"  Feature importance skipped: {e}")


if __name__ == "__main__":
    print("🚀 Starting Training Pipeline\n")
    combined_df = build_combined_dataset()
    X_train, X_test, y_train, y_test = time_split(combined_df)
    results, best = train_and_evaluate(X_train, X_test, y_train, y_test)
    print_feature_importance(results.get("feature_names", []))
    print("\n✅ Training complete!")