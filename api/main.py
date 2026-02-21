# api/main.py
# FastAPI backend — serves predictions via REST API

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from predict import predict_ticker, load_artifacts, LABEL_MAP

# ─── APP SETUP ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Stock Trend Predictor API",
    description = "Predicts stock movement (UP/DOWN/NEUTRAL) using ML + sentiment analysis",
    version     = "1.0.0"
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Supported tickers
SUPPORTED_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HDFCBANK.NS", "WIPRO.NS",
    "TATAMOTORS.NS", "BAJFINANCE.NS",
    "ICICIBANK.NS", "SBIN.NS", "ADANIENT.NS"
]

MODEL_NAMES = [
    "best",
    "logistic_regression",
    "svm",
    "knn",
    "random_forest",
    "xgboost"
]


# ─── REQUEST / RESPONSE MODELS ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker    : str
    model_name: Optional[str] = "best"


class PredictResponse(BaseModel):
    ticker          : str
    prediction      : str
    emoji           : str
    confidence_pct  : Optional[float]
    class_probs     : dict
    sentiment_score : float
    sentiment_label : str
    model_used      : str


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message"   : "Stock Trend Predictor API is running!",
        "docs"      : "/docs",
        "endpoints" : ["/predict", "/tickers", "/models", "/results", "/health"]
    }


@app.get("/health")
def health():
    """Check if model files are loaded correctly."""
    try:
        load_artifacts()
        return {"status": "healthy", "models_loaded": True}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/tickers")
def get_tickers():
    """Returns list of supported stock tickers."""
    return {"tickers": SUPPORTED_TICKERS}


@app.get("/models")
def get_models():
    """Returns list of available model names."""
    return {"models": MODEL_NAMES}


@app.get("/results")
def get_results():
    """Return model comparison results from training."""
    results_path = "models/results.json"
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404,
            detail="Results not found. Run src/train.py first."
        )
    with open(results_path) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict stock trend for a given ticker.
    
    - **ticker**: Stock symbol e.g. RELIANCE.NS
    - **model_name**: Which model to use (default = best)
    """
    ticker = req.ticker.upper()
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = ticker + ".NS"   # default to NSE

    try:
        result = predict_ticker(ticker, req.model_name)
        result["model_used"] = req.model_name
        return PredictResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict/{ticker}")
def predict_get(ticker: str, model_name: str = "best"):
    """
    GET version of predict endpoint (easier to test in browser).
    
    - **ticker**: Stock symbol e.g. RELIANCE.NS
    - **model_name**: Query param for model selection
    """
    return predict(PredictRequest(ticker=ticker, model_name=model_name))


# ─── RUN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# HOW TO RUN:
# cd api
# uvicorn main:app --reload --port 8000
# Then open: http://localhost:8000/docs
