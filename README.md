Islamabad AQI Predictor

**A fully serverless, automated ML system that predicts Islamabad's Air Quality Index for the next 24, 48, and 72 hours — running 24/7 with zero manual intervention.**

[![Live App] (https://share.streamlit.io/user/maham-bhatti)](https://aqi-predictor-natgjjfhdmgmvne6pkd9op.streamlit.app)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions)](https://github.com/maham-bhatti/aqi-predictor/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)

---

## Dashboard Preview

The live dashboard shows real-time AQI, 3-day forecasts, pollutant levels, 30-day history, and model performance — all updating automatically every hour.

---

## System Architecture

```
Weather & Pollution APIs
        │
        ▼
┌─────────────────────┐     Every hour (GitHub Actions)
│  Feature Pipeline   │ ──────────────────────────────────┐
│  feature_pipeline.py│                                   │
└─────────────────────┘                                   ▼
                                                  ┌───────────────┐
                                                  │   Supabase    │
                                                  │  PostgreSQL   │
                                                  │ (Feature Store│
                                                  │  + Targets)   │
                                                  └───────┬───────┘
                                                          │
┌─────────────────────┐     Every day 02:00 UTC           │
│  Training Pipeline  │ ◄────────────────────────────────┘
│  training_pipeline.py                           
└──────────┬──────────┘                           
           │                                      
           ▼                                      
     models/*.pkl                                 
     models/metrics.json                          
           │                                      
           ▼                                      
┌─────────────────────┐     Always-on (Streamlit Cloud)
│  Streamlit Dashboard│
│  streamlit_app.py   │
└─────────────────────┘
```

---

## Repository Structure

```
aqi-predictor/
│
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
│
├── feature_pipeline/
│   └── feature_pipeline.py       # Hourly data collection
│
├── training_pipeline/
│   ├── training_pipeline.py      # Daily model training
│   ├── backfill.py               # One-time historical backfill
│   └── notebooks/
│       └── eda_analysis.ipynb    # Exploratory Data Analysis
│
├── models/
│   ├── model_24h.pkl             # Ridge Regression (24h forecast)
│   ├── model_48h.pkl             # Ridge Regression (48h forecast)
│   ├── model_72h.pkl             # Ridge Regression (72h forecast)
│   ├── scaler.pkl                # StandardScaler
│   ├── features.pkl              # Feature name list
│   └── metrics.json             # Model performance metrics
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml  # Runs feature_pipeline.py every hour
│       └── training_pipeline.yml # Runs training_pipeline.py every day
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/maham-bhatti/aqi-predictor.git
cd aqi-predictor
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# or
source venv/bin/activate       # Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables
Create a `.env` file in the project root:
```env
AQICN_TOKEN=56bd3e6bf4e43f63b9ee23ba692a43bf49da9767
SUPABASE_URL=[https://your-project.supabase.co](https://zejndpznhjrjejvubsnq.supabase.co)
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inplam5kcHpuaGpyamVqdnVic25xIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3OTg1MTgzNCwiZXhwIjoyMDk1NDI3ODM0fQ.PUPugZqPODnVrdCmfiLOMSrFQ6T3xi146IBaKGMdKdM
```

| Variable | Where to Get |
|----------|-------------|
| `AQICN_TOKEN` | [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token) |
| `SUPABASE_URL` | Supabase Dashboard → Settings → API |
| `SUPABASE_KEY` | Supabase Dashboard → Settings → API |

### 5. Run the Dashboard Locally
```bash
streamlit run app/streamlit_app.py
```

---

## Pipeline Details

### Feature Pipeline — Runs Every Hour
**File:** `feature_pipeline/feature_pipeline.py`
**Schedule:** `0 * * * *` (GitHub Actions)

Collects and stores one hourly record:
- Fetches AQI + pollutants from AQICN API
- Fetches weather from Open-Meteo API
- Computes lag features (1h, 3h, 6h, 24h) from Supabase history
- Computes rolling averages (3h, 6h, 24h)
- Upserts record to Supabase (no duplicates)

### Training Pipeline — Runs Every Day at 02:00 UTC
**File:** `training_pipeline/training_pipeline.py`
**Schedule:** `0 2 * * *` (GitHub Actions)

Trains and saves models:
- Fetches all historical records from Supabase
- Trains 5 models: Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost
- Evaluates on 80/20 chronological split
- Saves best model (by RMSE) to `models/`
- Commits updated models back to GitHub

### Backfill — One-Time Historical Data
**File:** `training_pipeline/backfill.py`

Run once to populate 365 days of historical data:
```bash
python training_pipeline/backfill.py
```

---

##  Features Used (24 Total)

| Category | Features |
|----------|----------|
| **Pollutants** | `aqi`, `pm25`, `pm10`, `no2`, `o3`, `co`, `so2` |
| **Weather** | `temperature`, `humidity`, `windspeed`, `pressure`, `precipitation` |
| **Time** | `hour`, `day_of_week`, `month`, `is_weekend` |
| **Lag** | `aqi_lag_1h`, `aqi_lag_3h`, `aqi_lag_6h`, `aqi_lag_24h` |
| **Rolling** | `aqi_rolling_3h`, `aqi_rolling_6h`, `aqi_rolling_24h` |
| **Derived** | `aqi_change_rate` |

---

##  Model Performance

| Horizon | Model | RMSE | MAE | R² |
|---------|-------|------|-----|-----|
| **24h** | Ridge Regression | 10.94 | 8.09 | **0.363** |
| 48h | Ridge Regression | 13.68 | 10.24 | 0.005 |
| 72h | Ridge Regression | 16.90 | 11.86 | -0.098 |

> **Training data:** 6,975 hourly records · **Test data:** 1,744 records · **Period:** June 2025 – June 2026

Performance improves automatically as more data accumulates through the daily pipeline.

### All Models Evaluated (24h Horizon)

| Model | RMSE | R² |
|-------|------|----|
| Linear Regression | 10.94 | 0.363 |
| **Ridge Regression** ✅ | **10.94** | **0.363** |
| XGBoost | 12.10 | 0.220 |
| Gradient Boosting | 12.16 | 0.212 |
| Random Forest | 12.39 | 0.182 |

---

##  Dashboard Features

| Section | Description |
|---------|-------------|
|  **Current AQI Gauge** | Real-time AQI with delta vs 24h forecast |
|  **Weather Panel** | Temperature, humidity, wind speed, pressure |
|  **Pollutant Levels** | PM2.5, PM10, NO₂, O₃, CO, SO₂ bar chart |
|  **3-Day Forecast** | 24h / 48h / 72h AQI with category and health advice |
|  **30-Day History** | AQI time series with threshold lines |
|  **Weather Forecast** | 24h temperature + rain probability |
|  **AQI Heatmap** | Hour-of-day vs date heatmap |
|  **Model Metrics** | Live R² and RMSE for all horizons |
|  **Health Alerts** | Dynamic banners based on AQI severity |

---

##  GitHub Actions Setup

Add these secrets to your repository (**Settings → Secrets → Actions**):

| Secret | Value |
|--------|-------|
| `AQICN_TOKEN` | Your AQICN API token |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase service key |

The workflows run automatically once secrets are set.

---

##  Deployment (Streamlit Cloud)

1. Push repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) (https://aqi-predictor-natgjjfhdmgmvne6pkd9op.streamlit.app/)
3. Connect your GitHub repo
4. Set **Main file path** to `app/streamlit_app.py`
5. Add environment variables (AQICN_TOKEN, SUPABASE_URL, SUPABASE_KEY) in **Advanced settings**
6. Deploy

---

## Dependencies

```
requests       # API calls
pandas         # Data manipulation
numpy          # Numerical operations
scikit-learn   # ML models + scaler
xgboost        # XGBoost model
streamlit      # Web dashboard
python-dotenv  # Environment variables
plotly         # Interactive charts
shap           # Feature importance (planned)
supabase       # Feature store client
joblib         # Model serialization
```

---

## AQI Scale Reference

| AQI Range | Category | Health Implication |
|-----------|----------|--------------------|
| 0–50 | 😊 Good | No health risk |
| 51–100 | 😐 Moderate | Sensitive groups should take caution |
| 101–150 | 😷 Unhealthy (Sensitive) | Sensitive groups reduce outdoor activity |
| 151–200 | 🤢 Unhealthy | Everyone may experience health effects |
| 201–300 | 🚨 Very Unhealthy | Health alert — avoid outdoor exertion |
| 300+ | ☠️ Hazardous | Emergency — stay indoors |

---

## Project Submitted by:

Maham Shahid
📧 GitHub: [@maham-bhatti](https://github.com/maham-bhatti) For Project: (https://github.com/maham-bhatti/aqi-predictor)
🌐 Live App: [aqi-predictor-natgjjfhdmgmvne6pkd9op.streamlit.app](https://aqi-predictor-natgjjfhdmgmvne6pkd9op.streamlit.app)

--

##  License

This project is built for academic submission. Data is sourced from AQICN (CC BY) and Open-Meteo (CC BY 4.0).
