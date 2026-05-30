# Streamlit App
# ============================================================
#  AQI Predictor — Islamabad Air Quality Dashboard
#  app/streamlit_app.py
# ============================================================

import os, joblib, json, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Islamabad AQI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg:       #0a0f1e;
    --bg2:      #111827;
    --bg3:      #1a2235;
    --accent:   #00d4aa;
    --accent2:  #0099ff;
    --danger:   #ff4757;
    --warn:     #ffa502;
    --good:     #2ed573;
    --text:     #e8f4f0;
    --muted:    #8899aa;
    --border:   rgba(0,212,170,0.2);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%) !important;
}

h1,h2,h3,h4 { font-family: 'Space Mono', monospace !important; color: var(--text) !important; }

[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 2.2rem !important;
    color: var(--accent) !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.85rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

div[data-testid="stHorizontalBlock"] > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    backdrop-filter: blur(10px);
}

.aqi-card {
    background: linear-gradient(135deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.forecast-card {
    background: linear-gradient(160deg, #111827, #1a2235);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s;
}

.alert-danger {
    background: linear-gradient(135deg, rgba(255,71,87,0.15), rgba(255,71,87,0.05));
    border: 1px solid rgba(255,71,87,0.5);
    border-left: 4px solid var(--danger);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    animation: pulse 2s infinite;
}

.alert-warn {
    background: linear-gradient(135deg, rgba(255,165,2,0.15), rgba(255,165,2,0.05));
    border: 1px solid rgba(255,165,2,0.5);
    border-left: 4px solid var(--warn);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

.alert-good {
    background: linear-gradient(135deg, rgba(46,213,115,0.15), rgba(46,213,115,0.05));
    border: 1px solid rgba(46,213,115,0.5);
    border-left: 4px solid var(--good);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}

@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(255,71,87,0.3); }
    50%      { box-shadow: 0 0 0 8px rgba(255,71,87,0); }
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    color: #0a0f1e !important;
    font-weight: 700 !important;
}

div[data-testid="stPlotlyChart"] {
    background: transparent !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ── Env & clients ─────────────────────────────────────────────
AQICN_TOKEN  = os.getenv("AQICN_TOKEN", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
LAT, LON     = 33.6844, 73.0479

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    try:
        m24  = joblib.load(f"{base}/model_24h.pkl")
        m48  = joblib.load(f"{base}/model_48h.pkl")
        m72  = joblib.load(f"{base}/model_72h.pkl")
        sc   = joblib.load(f"{base}/scaler.pkl")
        feat = joblib.load(f"{base}/features.pkl")
        with open(f"{base}/metrics.json") as f:
            met = json.load(f)
        return m24, m48, m72, sc, feat, met
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None, None, None, {}

# ── AQI helpers ───────────────────────────────────────────────
def aqi_category(aqi):
    if aqi <= 50:   return "Good",            "#2ed573", "😊"
    if aqi <= 100:  return "Moderate",         "#ffa502", "😐"
    if aqi <= 150:  return "Unhealthy (Sensitive)", "#ff6b35", "😷"
    if aqi <= 200:  return "Unhealthy",        "#ff4757", "🤢"
    if aqi <= 300:  return "Very Unhealthy",   "#c44569", "🚨"
    return              "Hazardous",           "#8b0000", "☠️"

def aqi_advice(aqi):
    if aqi <= 50:   return "Air quality is satisfactory. Enjoy outdoor activities!"
    if aqi <= 100:  return "Acceptable quality. Unusually sensitive people should limit prolonged outdoor exertion."
    if aqi <= 150:  return "Sensitive groups should reduce outdoor activity. General public is fine."
    if aqi <= 200:  return "Everyone may begin to experience health effects. Limit outdoor activities."
    if aqi <= 300:  return "Health alert! Everyone should avoid outdoor exertion."
    return "Emergency conditions. Everyone should avoid all outdoor activity."

# ── Fetch live AQI ────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_live_aqi():
    try:
        url = f"https://api.waqi.info/feed/islamabad/?token={AQICN_TOKEN}"
        r   = requests.get(url, timeout=10)
        d   = r.json()["data"]
        iaqi = d.get("iaqi", {})
        return {
            "aqi":   float(d.get("aqi", 0) or 0),
            "pm25":  float(iaqi.get("pm25", {}).get("v", 0) or 0),
            "pm10":  float(iaqi.get("pm10", {}).get("v", 0) or 0),
            "no2":   float(iaqi.get("no2",  {}).get("v", 0) or 0),
            "o3":    float(iaqi.get("o3",   {}).get("v", 0) or 0),
            "co":    float(iaqi.get("co",   {}).get("v", 0) or 0),
            "so2":   float(iaqi.get("so2",  {}).get("v", 0) or 0),
        }
    except:
        return {"aqi":75,"pm25":35,"pm10":50,"no2":15,"o3":40,"co":0.5,"so2":5}

# ── Fetch live weather ────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_live_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LAT, "longitude": LON,
            "current": ["temperature_2m","relative_humidity_2m",
                        "wind_speed_10m","surface_pressure","precipitation",
                        "weather_code"],
            "hourly": ["temperature_2m","relative_humidity_2m",
                       "wind_speed_10m","precipitation_probability"],
            "forecast_days": 4,
            "timezone": "Asia/Karachi"
        }
        r = requests.get(url, params=params, timeout=10)
        d = r.json()
        c = d["current"]
        return {
            "temperature":   float(c.get("temperature_2m", 25)),
            "humidity":      float(c.get("relative_humidity_2m", 60)),
            "windspeed":     float(c.get("wind_speed_10m", 5)),
            "pressure":      float(c.get("surface_pressure", 950)),
            "precipitation": float(c.get("precipitation", 0)),
            "weather_code":  int(c.get("weather_code", 0)),
            "hourly":        d.get("hourly", {}),
        }
    except:
        return {"temperature":28,"humidity":55,"windspeed":8,
                "pressure":950,"precipitation":0,"weather_code":0,"hourly":{}}

def weather_icon(code):
    if code == 0:             return "☀️"
    if code in [1,2,3]:       return "⛅"
    if code in range(45,50):  return "🌫️"
    if code in range(51,68):  return "🌧️"
    if code in range(71,78):  return "❄️"
    if code in range(80,83):  return "🌦️"
    if code in range(95,100): return "⛈️"
    return "🌤️"

# ── Load historical data ──────────────────────────────────────
@st.cache_data(ttl=3600)
def load_history(days=30):
    try:
        supabase = get_supabase()
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        res = (supabase.table("aqi_features")
               .select("timestamp,aqi,pm25,pm10,temperature,humidity,windspeed")
               .gte("timestamp", since)
               .order("timestamp").execute())
        df = pd.DataFrame(res.data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except:
        return pd.DataFrame()

# ── Make prediction ───────────────────────────────────────────
def make_prediction(aqi_data, weather_data, models_tuple, history_df):
    m24, m48, m72, scaler, features, metrics = models_tuple
    if m24 is None:
        return aqi_data["aqi"], aqi_data["aqi"], aqi_data["aqi"]

    now = datetime.now(timezone.utc)

    # Build lag features from history
    aqi_vals = []
    if not history_df.empty:
        aqi_vals = history_df["aqi"].dropna().tolist()

    def lag(n):
        return float(aqi_vals[-n]) if len(aqi_vals) >= n else aqi_data["aqi"]
    def roll(n):
        return float(np.mean(aqi_vals[-n:])) if len(aqi_vals) >= n else aqi_data["aqi"]

    row = {
        "aqi":              aqi_data["aqi"],
        "pm25":             aqi_data["pm25"],
        "pm10":             aqi_data["pm10"],
        "no2":              aqi_data["no2"],
        "o3":               aqi_data["o3"],
        "co":               aqi_data["co"] / 1000,
        "so2":              aqi_data["so2"],
        "temperature":      weather_data["temperature"],
        "humidity":         weather_data["humidity"],
        "windspeed":        weather_data["windspeed"],
        "pressure":         weather_data["pressure"],
        "precipitation":    weather_data["precipitation"],
        "hour":             now.hour,
        "day_of_week":      now.weekday(),
        "month":            now.month,
        "is_weekend":       int(now.weekday() >= 5),
        "aqi_lag_1h":       lag(1),
        "aqi_lag_3h":       lag(3),
        "aqi_lag_6h":       lag(6),
        "aqi_lag_24h":      lag(24),
        "aqi_lag_48h":      lag(48),
        "aqi_rolling_3h":   roll(3),
        "aqi_rolling_6h":   roll(6),
        "aqi_rolling_24h":  roll(24),
        "aqi_rolling_48h":  roll(48),
        "aqi_change_rate":  aqi_data["aqi"] - lag(1),
        "aqi_change_6h":    aqi_data["aqi"] - lag(6),
        "aqi_change_24h":   aqi_data["aqi"] - lag(24),
        "temp_humidity":    weather_data["temperature"] * weather_data["humidity"] / 100,
        "wind_pressure":    weather_data["windspeed"] * weather_data["pressure"] / 1000,
        "is_morning":       int(6  <= now.hour <= 9),
        "is_evening":       int(17 <= now.hour <= 21),
        "is_night":         int(now.hour >= 22 or now.hour <= 5),
        "is_winter":        int(now.month in [11,12,1,2]),
        "is_summer":        int(now.month in [5,6,7,8]),
        "is_monsoon":       int(now.month in [7,8,9]),
    }

    try:
        X = np.array([[row.get(f, 0) for f in features]])
        use_scaled = hasattr(scaler, "mean_") and "Regression" in metrics.get("best_model","")
        if use_scaled:
            X = scaler.transform(X)
        p24 = float(m24.predict(X)[0])
        p48 = float(m48.predict(X)[0])
        p72 = float(m72.predict(X)[0])
        return max(0, p24), max(0, p48), max(0, p72)
    except Exception as e:
        return aqi_data["aqi"], aqi_data["aqi"], aqi_data["aqi"]

# ════════════════════════════════════════════════════════════
#  DASHBOARD LAYOUT
# ════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style='text-align:center; padding: 1rem 0 0.5rem;'>
    <h1 style='font-size:2.5rem; margin:0; background: linear-gradient(135deg, #00d4aa, #0099ff);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        🌿 ISLAMABAD AQI PREDICTOR
    </h1>
    <p style='color:#8899aa; font-family:DM Sans; margin:0.3rem 0 0;
              font-size:0.95rem; letter-spacing:0.05em;'>
        Real-time Air Quality Index · 3-Day Forecast · Powered by ML
    </p>
</div>
""", unsafe_allow_html=True)

# Load everything
with st.spinner("🔄 Fetching live data..."):
    aqi_data    = fetch_live_aqi()
    weather     = fetch_live_weather()
    history_df  = load_history(30)
    models_tup  = load_models()

pred_24, pred_48, pred_72 = make_prediction(
    aqi_data, weather, models_tup, history_df)

cat, color, emoji = aqi_category(aqi_data["aqi"])
now_str = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")

# ── Alert Banner ──────────────────────────────────────────────
if aqi_data["aqi"] > 200:
    st.markdown(f"""
    <div class='alert-danger'>
        <b>🚨 HAZARD ALERT</b> — AQI is <b>{aqi_data['aqi']:.0f}</b>.
        Everyone should avoid all outdoor activity immediately.
    </div>""", unsafe_allow_html=True)
elif aqi_data["aqi"] > 150:
    st.markdown(f"""
    <div class='alert-danger'>
        <b>⚠️ HEALTH WARNING</b> — AQI is <b>{aqi_data['aqi']:.0f}</b> (Unhealthy).
        Sensitive groups must stay indoors.
    </div>""", unsafe_allow_html=True)
elif aqi_data["aqi"] > 100:
    st.markdown(f"""
    <div class='alert-warn'>
        <b>⚠️ CAUTION</b> — AQI is <b>{aqi_data['aqi']:.0f}</b> (Moderate-Unhealthy).
        Sensitive individuals should limit outdoor exposure.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class='alert-good'>
        <b>✅ AIR QUALITY</b> — AQI is <b>{aqi_data['aqi']:.0f}</b> ({cat}).
        {aqi_advice(aqi_data['aqi'])}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Current AQI Gauge + Weather Cards ─────────────────
col_gauge, col_weather = st.columns([1, 2], gap="large")

with col_gauge:
    st.markdown("<p class='section-title'>📍 Current AQI · Islamabad</p>",
                unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_data["aqi"],
        delta={"reference": pred_24, "valueformat": ".0f",
               "prefix": "24h forecast: "},
        title={"text": f"{emoji} {cat}", "font": {"size": 18, "color": "#e8f4f0",
                                                    "family": "Space Mono"}},
        gauge={
            "axis":      {"range": [0, 300], "tickcolor": "#8899aa",
                          "tickfont": {"color":"#8899aa","size":10}},
            "bar":       {"color": color, "thickness": 0.25},
            "bgcolor":   "#111827",
            "bordercolor": "rgba(0,212,170,0.3)",
            "steps": [
                {"range": [0,   50],  "color": "rgba(46,213,115,0.15)"},
                {"range": [50,  100], "color": "rgba(255,165,2,0.15)"},
                {"range": [100, 150], "color": "rgba(255,107,53,0.15)"},
                {"range": [150, 200], "color": "rgba(255,71,87,0.15)"},
                {"range": [200, 300], "color": "rgba(196,69,105,0.15)"},
            ],
            "threshold": {"line":{"color":"white","width":3},
                          "thickness":0.8, "value": aqi_data["aqi"]},
        },
        number={"font": {"size": 48, "color": color, "family": "Space Mono"},
                "suffix": " AQI"},
    ))
    fig_gauge.update_layout(
        height=300, margin=dict(l=20,r=20,t=40,b=20),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e8f4f0",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown(f"""
    <div style='text-align:center; color:#8899aa; font-size:0.8rem;
                font-family:Space Mono; margin-top:-0.5rem;'>
        Last updated: {now_str}
    </div>""", unsafe_allow_html=True)

with col_weather:
    st.markdown("<p class='section-title'>🌤️ Current Weather · Islamabad</p>",
                unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    wicon = weather_icon(weather["weather_code"])
    with w1:
        st.metric(f"{wicon} Temperature",
                  f"{weather['temperature']:.1f}°C")
    with w2:
        st.metric("💧 Humidity",
                  f"{weather['humidity']:.0f}%")
    with w3:
        st.metric("💨 Wind Speed",
                  f"{weather['windspeed']:.1f} km/h")
    with w4:
        st.metric("🌡️ Pressure",
                  f"{weather['pressure']:.0f} hPa")

    st.markdown("<br>", unsafe_allow_html=True)

    # Pollutants bar chart
    st.markdown("<p class='section-title'>🧪 Pollutant Levels</p>",
                unsafe_allow_html=True)
    poll_names = ["PM2.5","PM10","NO₂","O₃","CO×100","SO₂"]
    poll_vals  = [
        aqi_data["pm25"], aqi_data["pm10"],
        aqi_data["no2"],  aqi_data["o3"],
        aqi_data["co"] * 100, aqi_data["so2"]
    ]
    poll_colors = ["#00d4aa","#0099ff","#ff6b35","#ffa502","#c44569","#7bed9f"]
    fig_poll = go.Figure(go.Bar(
        x=poll_names, y=poll_vals,
        marker_color=poll_colors,
        marker_line_color="rgba(0,0,0,0)",
        text=[f"{v:.1f}" for v in poll_vals],
        textposition="outside",
        textfont={"color":"#e8f4f0","size":11},
    ))
    fig_poll.update_layout(
        height=220, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#8899aa", tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(136,153,170,0.1)",
                   color="#8899aa", title="μg/m³"),
        showlegend=False,
    )
    st.plotly_chart(fig_poll, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: 3-Day Forecast ─────────────────────────────────────
st.markdown("<p class='section-title'>📅 3-Day AQI Forecast</p>",
            unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns(3, gap="medium")
forecasts = [
    ("Tomorrow",    pred_24, "24h"),
    ("Day After",   pred_48, "48h"),
    ("In 3 Days",   pred_72, "72h"),
]

for col, (label, pred, tag) in zip([fc1, fc2, fc3], forecasts):
    fcat, fcol, femo = aqi_category(pred)
    with col:
        st.markdown(f"""
        <div class='forecast-card'>
            <p style='color:#8899aa; font-size:0.75rem; font-family:Space Mono;
                      letter-spacing:0.1em; margin:0;'>{tag.upper()} FORECAST</p>
            <h3 style='color:{fcol}; font-size:2.8rem; margin:0.3rem 0;
                       font-family:Space Mono;'>{pred:.0f}</h3>
            <p style='color:#8899aa; font-size:0.8rem; margin:0;'>AQI · {label}</p>
            <p style='font-size:1.5rem; margin:0.5rem 0;'>{femo}</p>
            <span style='background:{fcol}22; color:{fcol}; padding:0.2rem 0.8rem;
                         border-radius:20px; font-size:0.75rem;
                         font-family:Space Mono; border:1px solid {fcol}44;'>
                {fcat}
            </span>
            <p style='color:#8899aa; font-size:0.72rem; margin:0.5rem 0 0;
                      font-family:DM Sans;'>{aqi_advice(pred)[:60]}...</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 3: Historical Trend + Hourly Forecast ─────────────────
col_hist, col_hourly = st.columns([3, 2], gap="large")

with col_hist:
    st.markdown("<p class='section-title'>📈 30-Day AQI History</p>",
                unsafe_allow_html=True)
    if not history_df.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=history_df["timestamp"], y=history_df["aqi"],
            mode="lines", name="AQI",
            line=dict(color="#00d4aa", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.08)",
        ))
        # Add AQI threshold lines
        for level, lcolor, lname in [
            (50,"#2ed573","Good"),
            (100,"#ffa502","Moderate"),
            (150,"#ff4757","Unhealthy")
        ]:
            fig_hist.add_hline(
                y=level, line_dash="dot",
                line_color=lcolor, opacity=0.5,
                annotation_text=lname,
                annotation_font_color=lcolor,
                annotation_font_size=10,
            )
        fig_hist.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#8899aa"),
            yaxis=dict(showgrid=True, gridcolor="rgba(136,153,170,0.1)",
                       color="#8899aa", title="AQI"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#8899aa"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Historical data loading...")

with col_hourly:
    st.markdown("<p class='section-title'>⏱️ 24h Weather Forecast</p>",
                unsafe_allow_html=True)
    hourly = weather.get("hourly", {})
    if hourly:
        times  = hourly.get("time", [])[:24]
        temps  = hourly.get("temperature_2m", [])[:24]
        rain_p = hourly.get("precipitation_probability", [])[:24]
        winds  = hourly.get("wind_speed_10m", [])[:24]

        fig_hw = go.Figure()
        fig_hw.add_trace(go.Scatter(
            x=list(range(len(temps))), y=temps,
            name="Temp °C", line=dict(color="#ff6b35", width=2),
            yaxis="y1",
        ))
        fig_hw.add_trace(go.Bar(
            x=list(range(len(rain_p))), y=rain_p,
            name="Rain %", marker_color="rgba(0,153,255,0.3)",
            yaxis="y2",
        ))
        fig_hw.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#8899aa",
                       tickvals=list(range(0,24,4)),
                       ticktext=[f"{h}:00" for h in range(0,24,4)]),
            yaxis=dict(showgrid=True, gridcolor="rgba(136,153,170,0.1)",
                       color="#ff6b35", title="Temperature °C"),
            yaxis2=dict(overlaying="y", side="right",
                        color="#0099ff", title="Rain Prob %",
                        range=[0,100]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#8899aa",
                        orientation="h", y=1.1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_hw, use_container_width=True)
    else:
        st.info("Weather forecast loading...")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 4: Monthly heatmap + Model metrics ────────────────────
col_heat, col_metrics = st.columns([3, 2], gap="large")

with col_heat:
    st.markdown("<p class='section-title'>🗓️ Monthly AQI Heatmap</p>",
                unsafe_allow_html=True)
    if not history_df.empty and len(history_df) > 7:
        hdf = history_df.copy()
        hdf["date"] = hdf["timestamp"].dt.date
        hdf["hour"] = hdf["timestamp"].dt.hour
        pivot = hdf.pivot_table(
            values="aqi", index="hour", columns="date", aggfunc="mean")
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[f"{h:02d}:00" for h in pivot.index],
            colorscale=[
                [0.0,  "#2ed573"],
                [0.33, "#ffa502"],
                [0.66, "#ff4757"],
                [1.0,  "#8b0000"],
            ],
            colorbar=dict(
                title="AQI",
                tickfont=dict(color="#8899aa"),
                title_font=dict(color="#8899aa"),  # ← fixed
                ),
               
            hoverongaps=False,
        ))
        fig_heat.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#8899aa",
                       title="Date", nticks=10),
            yaxis=dict(showgrid=False, color="#8899aa",
                       title="Hour of Day", autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Need more history data for heatmap...")

with col_metrics:
    st.markdown("<p class='section-title'>🤖 Model Performance</p>",
                unsafe_allow_html=True)
    _, _, _, _, _, met = models_tup
    if met:
        bm = met.get("best_model", "N/A")
        st.markdown(f"""
        <div style='background:rgba(0,212,170,0.08); border:1px solid rgba(0,212,170,0.3);
                    border-radius:12px; padding:0.8rem 1rem; margin-bottom:0.8rem;'>
            <p style='color:#8899aa; font-size:0.72rem; font-family:Space Mono;
                      margin:0; letter-spacing:0.1em;'>BEST MODEL</p>
            <p style='color:#00d4aa; font-size:1.1rem; font-family:Space Mono;
                      margin:0.2rem 0 0; font-weight:700;'>{bm}</p>
        </div>
        """, unsafe_allow_html=True)

        for horizon, key in [("24h Forecast","model_24h"),
                              ("48h Forecast","model_48h"),
                              ("72h Forecast","model_72h")]:
            m = met.get(key, {})
            if m:
                r2   = m.get("r2", 0)
                rmse = m.get("rmse", 0)
                r2c  = "#2ed573" if r2 > 0.7 else "#ffa502" if r2 > 0.4 else "#ff4757"
                st.markdown(f"""
                <div style='background:rgba(17,24,39,0.8);
                            border:1px solid rgba(0,212,170,0.15);
                            border-radius:10px; padding:0.7rem 1rem; margin:0.4rem 0;
                            display:flex; justify-content:space-between;'>
                    <span style='color:#8899aa; font-size:0.8rem;
                                 font-family:Space Mono;'>{horizon}</span>
                    <span>
                        <span style='color:{r2c}; font-family:Space Mono;
                                     font-size:0.85rem; font-weight:700;'>
                            R²={r2:.3f}
                        </span>
                        <span style='color:#8899aa; font-size:0.75rem;
                                     margin-left:0.5rem;'>
                            RMSE={rmse:.1f}
                        </span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # Feature count
        n_feat = met.get("n_features", 0)
        train  = met.get("train_size", 0)
        st.markdown(f"""
        <div style='margin-top:0.8rem; color:#8899aa; font-size:0.78rem;
                    font-family:DM Sans;'>
            📊 Trained on <b style='color:#00d4aa;'>{train:,}</b> hourly records
            using <b style='color:#00d4aa;'>{n_feat}</b> features
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model metrics not available.")

st.markdown("<br>", unsafe_allow_html=True)

# ── AQI Scale Reference ───────────────────────────────────────
st.markdown("<p class='section-title'>📋 AQI Scale Reference</p>",
            unsafe_allow_html=True)

scale_cols = st.columns(6)
scale_data = [
    ("0–50",   "Good",             "#2ed573", "😊", "No health risk"),
    ("51–100", "Moderate",         "#ffa502", "😐", "Sensitive caution"),
    ("101–150","Unhealthy (SG)",   "#ff6b35", "😷", "Sensitive groups"),
    ("151–200","Unhealthy",        "#ff4757", "🤢", "Everyone affected"),
    ("201–300","Very Unhealthy",   "#c44569", "🚨", "Health emergency"),
    ("300+",   "Hazardous",        "#8b0000", "☠️", "Stay indoors"),
]
for col, (rng, cat, clr, ico, desc) in zip(scale_cols, scale_data):
    with col:
        st.markdown(f"""
        <div style='background:{clr}11; border:1px solid {clr}44;
                    border-top: 3px solid {clr};
                    border-radius:10px; padding:0.7rem; text-align:center;'>
            <p style='font-size:1.3rem; margin:0;'>{ico}</p>
            <p style='color:{clr}; font-family:Space Mono; font-size:0.75rem;
                      font-weight:700; margin:0.2rem 0;'>{rng}</p>
            <p style='color:#e8f4f0; font-size:0.72rem; margin:0;'>{cat}</p>
            <p style='color:#8899aa; font-size:0.65rem; margin:0.2rem 0 0;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#8899aa; font-size:0.75rem;
            font-family:Space Mono; padding:1rem;
            border-top:1px solid rgba(0,212,170,0.15);'>
    🌿 Islamabad AQI Predictor · ML-powered · Data: AQICN + Open-Meteo
    · Built with Streamlit · © 2026
</div>
""", unsafe_allow_html=True)