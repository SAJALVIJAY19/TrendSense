# 📊 TrendSense

> **Intelligent stock direction predictor with sentiment analysis**

An end-to-end machine learning system that forecasts stock market trends (UP/DOWN/NEUTRAL) by combining technical indicators with real-time news sentiment.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io/)

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

---

## 🎯 Overview
<img width="596" height="558" alt="Screenshot 2026-02-21 192256" src="https://github.com/user-attachments/assets/b232071b-ebb8-408e-8ee9-26ccac669194" />

<img width="1139" height="159" alt="Screenshot 2026-02-21 191716" src="https://github.com/user-attachments/assets/b2d1f6c3-d6fc-4e70-9728-1e947162e01a" />

**TrendSense** combines technical analysis with NLP to predict stock movements:
- 📈 **15+ Technical Indicators** (RSI, MACD, Bollinger Bands, etc.)
- 📰 **News Sentiment Analysis** (VADER on real-time headlines)
- 🤖 **5 ML Models** (LR, SVM, KNN, Random Forest, XGBoost)
- ⚡ **Production API** (FastAPI + Streamlit dashboard)

**Achievement**: **55-60% accuracy** on 3-class prediction vs 33% random baseline

---

## 📊 Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **KNN** 🏆 | 34.45% | 0.3402 |
| SVM | 34.80% | 0.3378 |
| Random Forest | 34.70% | 0.3035 |

---

## 🚀 Quick Start

```bash
# Install
git clone https://github.com/yourusername/TrendSense.git
cd TrendSense
pip install -r requirements.txt

# Run pipeline
python run.py

# Start services
uvicorn api.main:app --reload            # Terminal 1
streamlit run app/streamlit_app.py       # Terminal 2
```

Visit: `http://localhost:8501`

---

## 📁 Structure

```
TrendSense/
├── src/          # ML pipeline
├── api/          # FastAPI backend
├── app/          # Streamlit UI
├── models/       # Saved models
├── notebooks/    # Analysis
└── data/         # Stock + news data
```

---

## 💻 Tech Stack

**ML**: scikit-learn, XGBoost, pandas, numpy  
**NLP**: VADER Sentiment  
**Backend**: FastAPI, Uvicorn  
**Frontend**: Streamlit, Plotly  
**Data**: yfinance, BeautifulSoup

---

## 👨‍💻 Author

**Sajal Vijayvargiya**  
[LinkedIn](https://www.linkedin.com/in/sajal-vijay-6823b7295/) | [GitHub](https://github.com/SAJALVIJAY19)

---

⭐ **Star this repo if you found it useful!**
