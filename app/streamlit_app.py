# app/streamlit_app.py
# Streamlit frontend dashboard

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import json
import joblib
import requests
from datetime import datetime, timedelta

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Stock Trend Predictor",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─── STYLING ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(90deg, #1B3A6B, #2563EB);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #F0F4FF; border-radius: 12px;
        padding: 1rem 1.5rem; border-left: 4px solid #2563EB;
    }
    .prediction-up     { color: #059669; font-size: 2rem; font-weight: 700; }
    .prediction-down   { color: #DC2626; font-size: 2rem; font-weight: 700; }
    .prediction-neutral{ color: #D97706; font-size: 2rem; font-weight: 700; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────

TICKERS = {
    "Reliance Industries" : "RELIANCE.NS",
    "TCS"                 : "TCS.NS",
    "Infosys"             : "INFY.NS",
    "HDFC Bank"           : "HDFCBANK.NS",
    "Wipro"               : "WIPRO.NS",
    "Tata Motors"         : "TATAMOTORS.NS",
    "Bajaj Finance"       : "BAJFINANCE.NS",
    "ICICI Bank"          : "ICICIBANK.NS",
    "State Bank of India" : "SBIN.NS",
}

MODELS = [
    "best",
    "logistic_regression",
    "svm",
    "knn",
    "random_forest",
    "xgboost"
]

COLOR_MAP = {"UP": "#059669", "NEUTRAL": "#D97706", "DOWN": "#DC2626"}
API_BASE  = "http://localhost:8000"

# ─── HELPERS ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)   # cache for 5 minutes
def fetch_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def call_predict_api(ticker: str, model_name: str) -> dict:
    """Call FastAPI predict endpoint."""
    try:
        resp = requests.get(
            f"{API_BASE}/predict/{ticker}",
            params={"model_name": model_name},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend. Start it with: `uvicorn api.main:app --reload`")
        return None


def load_results_local() -> dict:
    """Load model comparison results from local JSON."""
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ─── CHARTS ──────────────────────────────────────────────────────────────────

def plot_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Interactive candlestick chart with volume."""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x     = df.index,
        open  = df["Open"],
        high  = df["High"],
        low   = df["Low"],
        close = df["Close"],
        name  = ticker,
        increasing_line_color = "#059669",
        decreasing_line_color = "#DC2626"
    ))

    # EMA lines
    close = df["Close"].squeeze()
    ema9  = close.ewm(span=9).mean()
    ema21 = close.ewm(span=21).mean()

    fig.add_trace(go.Scatter(x=df.index, y=ema9,  name="EMA 9",
                             line=dict(color="#2563EB", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=ema21, name="EMA 21",
                             line=dict(color="#F59E0B", width=1.5)))

    fig.update_layout(
        title          = f"{ticker} — Candlestick Chart",
        xaxis_title    = "Date",
        yaxis_title    = "Price (₹)",
        template       = "plotly_white",
        height         = 420,
        xaxis_rangeslider_visible = False,
        legend         = dict(orientation="h", y=1.02, x=0),
        margin         = dict(l=40, r=20, t=60, b=40)
    )
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    """RSI chart with overbought/oversold zones."""
    close = df["Close"].squeeze()
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI",
                             line=dict(color="#7C3AED", width=2)))
    fig.add_hline(y=70, line_dash="dash", line_color="#DC2626",
                  annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#059669",
                  annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="#9CA3AF")

    fig.update_layout(
        title    = "RSI (14)",
        height   = 220,
        template = "plotly_white",
        yaxis    = dict(range=[0, 100]),
        margin   = dict(l=40, r=20, t=40, b=20)
    )
    return fig


def plot_model_comparison(results: dict) -> go.Figure:
    """Bar chart comparing all 5 models."""
    model_names = [k for k in results.keys()
                   if k not in ("best_model", "feature_names")]
    accuracies  = [results[m]["accuracy"]  * 100 for m in model_names]
    f1_scores   = [results[m]["f1_score"]  * 100 for m in model_names]
    best        = results.get("best_model", "")

    colors = ["#2563EB" if m == best else "#93C5FD" for m in model_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy (%)",  x=model_names, y=accuracies,
                         marker_color=colors, opacity=0.85))
    fig.add_trace(go.Bar(name="F1 Score (%)", x=model_names, y=f1_scores,
                         marker_color="#059669", opacity=0.75))

    fig.update_layout(
        title    = "Model Comparison — Accuracy vs F1 Score",
        barmode  = "group",
        height   = 300,
        template = "plotly_white",
        yaxis    = dict(title="%", range=[0, 100]),
        legend   = dict(orientation="h", y=1.1),
        margin   = dict(l=40, r=20, t=60, b=40)
    )
    return fig


def plot_confidence_donut(class_probs: dict) -> go.Figure:
    """Donut chart showing prediction confidence breakdown."""
    colors = [COLOR_MAP.get(k, "#9CA3AF") for k in class_probs.keys()]
    fig = go.Figure(go.Pie(
        labels    = list(class_probs.keys()),
        values    = list(class_probs.values()),
        hole      = 0.55,
        marker    = dict(colors=colors),
        textinfo  = "label+percent"
    ))
    fig.update_layout(
        height      = 250,
        showlegend  = False,
        margin      = dict(l=20, r=20, t=20, b=20),
        annotations = [dict(text="Confidence", x=0.5, y=0.5,
                            font_size=13, showarrow=False)]
    )
    return fig


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-share.png", width=64)
    st.title("⚙️ Settings")
    st.divider()

    selected_company = st.selectbox("📌 Select Stock", list(TICKERS.keys()))
    selected_ticker  = TICKERS[selected_company]

    selected_model = st.selectbox("🤖 Select Model", MODELS,
                                  help="'best' uses the highest F1-score model from training")

    chart_period = st.selectbox("📅 Chart Period",
                                ["1mo", "3mo", "6mo", "1y", "2y"],
                                index=3)

    st.divider()
    predict_btn = st.button("🔮 Predict Now", type="primary", use_container_width=True)

    st.divider()
    st.caption("⚠️ Educational purposes only. Not financial advice.")
    st.caption("Made with ❤️ using scikit-learn + Streamlit")


# ─── MAIN CONTENT ────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">📈 Stock Trend Predictor</p>', unsafe_allow_html=True)
st.markdown("*ML-powered stock direction prediction using technical indicators + news sentiment*")
st.divider()

# ── Tab Layout ──
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Charts & Analysis", "🏆 Model Comparison"])


# ═══════════════════════════════════════════════════════
# TAB 1: PREDICTION
# ═══════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"🏢 {selected_company}")
        st.caption(f"Ticker: `{selected_ticker}`  |  Model: `{selected_model}`")

        # Live price
        try:
            ticker_obj = yf.Ticker(selected_ticker)
            info = ticker_obj.fast_info
            curr_price = info.last_price
            prev_close = info.previous_close
            change     = ((curr_price - prev_close) / prev_close) * 100
            direction  = "🔺" if change >= 0 else "🔻"

            st.metric(
                label  = "Current Price",
                value  = f"₹{curr_price:,.2f}",
                delta  = f"{direction} {change:+.2f}% today"
            )
        except:
            st.info("Could not fetch live price. Market may be closed.")

    with col2:
        if predict_btn:
            with st.spinner("Running ML prediction + scraping news..."):
                result = call_predict_api(selected_ticker, selected_model)

            if result:
                pred    = result["prediction"]
                conf    = result.get("confidence_pct")
                sent    = result.get("sentiment_score", 0)
                sent_lb = result.get("sentiment_label", "Neutral")

                # Big prediction display
                color = COLOR_MAP[pred]
                st.markdown(
                    f'<div style="text-align:center; padding:1rem; '
                    f'background:#F8FAFF; border-radius:12px; '
                    f'border: 2px solid {color};">'
                    f'<div style="font-size:1rem; color:#6B7280;">5-Day Prediction</div>'
                    f'<div style="font-size:3rem; font-weight:800; color:{color};">'
                    f'{result["emoji"]} {pred}</div>'
                    f'{"<div style=font-size:1rem;color:#374151;>Confidence: " + str(conf) + "%</div>" if conf else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.divider()

                # Sentiment display
                sent_color = "#059669" if sent > 0.05 else ("#DC2626" if sent < -0.05 else "#D97706")
                st.markdown(
                    f'<div style="padding:0.75rem; background:#F9FAFB; border-radius:8px;">'
                    f'📰 News Sentiment: <b style="color:{sent_color};">{sent_lb}</b> '
                    f'(score: {sent:+.3f})</div>',
                    unsafe_allow_html=True
                )

                # Confidence donut
                if result.get("class_probs"):
                    st.plotly_chart(
                        plot_confidence_donut(result["class_probs"]),
                        use_container_width=True
                    )
        else:
            st.info("👈 Select a stock and click **Predict Now** to get started!")
            st.markdown("""
            **How it works:**
            1. Fetches last 100 days of price data
            2. Computes 10+ technical indicators
            3. Scrapes latest news headlines
            4. Scores news sentiment using VADER
            5. Runs your selected ML model
            6. Returns UP / DOWN / NEUTRAL prediction
            """)


# ═══════════════════════════════════════════════════════
# TAB 2: CHARTS
# ═══════════════════════════════════════════════════════
with tab2:
    st.subheader(f"📊 Technical Analysis — {selected_company}")

    with st.spinner("Loading chart data..."):
        df = fetch_price_data(selected_ticker, chart_period)

    if not df.empty:
        # Candlestick chart
        st.plotly_chart(plot_candlestick(df, selected_ticker), use_container_width=True)

        # RSI chart
        st.plotly_chart(plot_rsi(df), use_container_width=True)

        # Quick stats
        st.subheader("📋 Quick Stats")
        close = df["Close"].squeeze()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{close.iloc[-1]:,.2f}")
        col2.metric("52W High",      f"₹{close.max():,.2f}")
        col3.metric("52W Low",       f"₹{close.min():,.2f}")
        col4.metric("Avg Volume",    f"{df['Volume'].squeeze().mean():,.0f}")

        # Raw data expander
        with st.expander("📄 View Raw Data"):
            st.dataframe(df.tail(30).sort_index(ascending=False))
    else:
        st.error(f"Could not load data for {selected_ticker}")


# ═══════════════════════════════════════════════════════
# TAB 3: MODEL COMPARISON
# ═══════════════════════════════════════════════════════
with tab3:
    st.subheader("🏆 Model Comparison Results")

    results = load_results_local()

    if not results:
        st.warning("No training results found. Run `python src/train.py` first!")
    else:
        best_model = results.get("best_model", "Unknown")
        st.success(f"🥇 Best Model: **{best_model}**")

        # Bar chart
        st.plotly_chart(plot_model_comparison(results), use_container_width=True)

        # Results table
        model_names = [k for k in results.keys()
                       if k not in ("best_model", "feature_names")]
        table_data  = []
        for m in model_names:
            r = results[m]
            table_data.append({
                "Model"     : f"🥇 {m}" if m == best_model else m,
                "Accuracy"  : f"{r['accuracy']*100:.1f}%",
                "F1 Score"  : f"{r['f1_score']:.4f}",
                "Precision" : f"{r['precision']:.4f}",
                "Recall"    : f"{r['recall']:.4f}",
            })

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Detailed report expander
        for m in model_names:
            with st.expander(f"📋 Full Classification Report — {m}"):
                st.code(results[m].get("report", "Not available"))

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("📌 Stock Trend Predictor | Built with Python, scikit-learn, XGBoost, FastAPI & Streamlit | ⚠️ Not financial advice")
