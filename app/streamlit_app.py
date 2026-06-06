# ============================================================
#  AQI Predictor — Islamabad Air Quality Dashboard
#  app/streamlit_app.py
# ============================================================

import os, joblib, json, requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

st.set_page_config(
    page_title="Islamabad AQI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
    --bg:#f0f7ff; --bg2:#ffffff; --bg3:#e8f4fd;
    --accent:#0288d1; --accent2:#00bcd4; --accent3:#26a69a;
    --danger:#f4511e; --warn:#ff9800; --good:#43a047;
    --text:#1a3a4a; --muted:#5a8fa8;
    --border:rgba(2,136,209,0.18);
    --card-bg:rgba(255,255,255,0.85);
    --grad:linear-gradient(135deg,#e8f4fd 0%,#f0f7ff 50%,#e8fdf5 100%);
}
html,body,[data-testid="stAppViewContainer"]{
    background:var(--grad) !important;
    color:var(--text) !important;
    font-family:'Inter',sans-serif !important;
}
[data-testid="stAppViewContainer"]{background:var(--grad) !important;}
h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif !important;color:var(--text) !important;}
[data-testid="stMetricValue"]{
    font-family:'Space Grotesk',monospace !important;
    font-size:1.8rem !important;color:var(--accent) !important;font-weight:700 !important;
}
[data-testid="stMetricLabel"]{color:var(--muted) !important;font-size:0.82rem !important;font-weight:500 !important;}
[data-testid="stMetricDelta"]{font-size:0.82rem !important;}
div[data-testid="stHorizontalBlock"]>div{
    background:var(--card-bg) !important;border:1px solid var(--border) !important;
    border-radius:16px !important;padding:1rem !important;
    backdrop-filter:blur(8px);box-shadow:0 4px 20px rgba(2,136,209,0.06);
}
.alert-danger{
    background:linear-gradient(90deg,#fff8f0,#fff3e8);
    border:1px solid #ffb74d;border-left:5px solid #f4511e;
    border-radius:12px;padding:0.9rem 1.4rem;margin:0.8rem 0;
    color:#bf360c;font-weight:600;font-size:0.9rem;
}
.alert-warn{
    background:linear-gradient(90deg,#fffde7,#fff8e1);
    border:1px solid #ffe082;border-left:5px solid #ff9800;
    border-radius:12px;padding:0.9rem 1.4rem;margin:0.8rem 0;
    color:#e65100;font-weight:600;font-size:0.9rem;
}
.alert-good{
    background:linear-gradient(90deg,#e8f5e9,#f1f8e9);
    border:1px solid #a5d6a7;border-left:5px solid #43a047;
    border-radius:12px;padding:0.9rem 1.4rem;margin:0.8rem 0;
    color:#1b5e20;font-weight:600;font-size:0.9rem;
}
.forecast-card{
    background:var(--card-bg);border:1px solid var(--border);
    border-radius:16px;padding:1.3rem;text-align:center;
    backdrop-filter:blur(8px);box-shadow:0 4px 20px rgba(2,136,209,0.06);
}
.section-title{
    font-family:'Space Grotesk',sans-serif;font-size:0.72rem;
    letter-spacing:0.14em;color:var(--accent);text-transform:uppercase;
    margin-bottom:0.5rem;border-bottom:1px solid var(--border);
    padding-bottom:0.45rem;font-weight:700;
}
[data-testid="stSidebar"]{background:var(--bg2) !important;border-right:1px solid var(--border) !important;}
div[data-testid="stPlotlyChart"]{background:transparent !important;border-radius:14px !important;overflow:hidden !important;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:var(--bg3);}
::-webkit-scrollbar-thumb{background:rgba(2,136,209,0.3);border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────
AQICN_TOKEN  = os.getenv("AQICN_TOKEN", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
LAT, LON     = 33.6844, 73.0479
BG           = "rgba(255,255,255,0)"
FONT_COL     = "#1a3a4a"
MUTED        = "#5a8fa8"
GRID_COL     = "rgba(2,136,209,0.1)"

# ── Resource loaders ──────────────────────────────────────────
@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_models():
    candidates = [
        "models",
        os.path.join(os.path.dirname(__file__), "..", "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
        "/mount/src/aqi-predictor/models",
        "/app/models",
    ]
    base = None
    for path in candidates:
        full = os.path.normpath(path)
        if os.path.exists(os.path.join(full, "model_24h.pkl")):
            base = full
            break
    if base is None:
        st.error(f"❌ Models not found.")
        return None, None, None, None, None, {}
    try:
        m24  = joblib.load(os.path.join(base, "model_24h.pkl"))
        m48  = joblib.load(os.path.join(base, "model_48h.pkl"))
        m72  = joblib.load(os.path.join(base, "model_72h.pkl"))
        sc   = joblib.load(os.path.join(base, "scaler.pkl"))
        feat = joblib.load(os.path.join(base, "features.pkl"))
        with open(os.path.join(base, "metrics.json")) as f:
            met = json.load(f)
        return m24, m48, m72, sc, feat, met
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None, None, None, {}

# ── AQI helpers ───────────────────────────────────────────────
def aqi_category(aqi):
    if aqi <= 50:  return "Good",                 "#43a047", "😊"
    if aqi <= 100: return "Moderate",              "#ff9800", "😐"
    if aqi <= 150: return "Unhealthy (Sensitive)", "#fb8c00", "😷"
    if aqi <= 200: return "Unhealthy",             "#f4511e", "🤢"
    if aqi <= 300: return "Very Unhealthy",        "#c62828", "🚨"
    return               "Hazardous",              "#4e342e", "☠️"

def aqi_advice(aqi):
    if aqi <= 50:  return "Air quality is satisfactory. Enjoy outdoor activities!"
    if aqi <= 100: return "Acceptable quality. Sensitive people should limit outdoor exertion."
    if aqi <= 150: return "Sensitive groups should reduce outdoor activity."
    if aqi <= 200: return "Everyone may experience health effects. Limit outdoor activities."
    if aqi <= 300: return "Health alert! Everyone should avoid outdoor exertion."
    return "Emergency conditions. Everyone should avoid all outdoor activity."

def weather_icon(code):
    if code == 0:              return "☀️"
    if code in [1, 2, 3]:     return "⛅"
    if code in range(45, 50): return "🌫️"
    if code in range(51, 68): return "🌧️"
    if code in range(71, 78): return "❄️"
    if code in range(80, 83): return "🌦️"
    if code in range(95, 100):return "⛈️"
    return "🌤️"

# ── Data fetchers ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_live_aqi():
    try:
        r    = requests.get(f"https://api.waqi.info/feed/islamabad/?token={AQICN_TOKEN}", timeout=10)
        d    = r.json()["data"]
        iaqi = d.get("iaqi", {})
        return {
            "aqi":  float(d.get("aqi", 0) or 0),
            "pm25": float(iaqi.get("pm25", {}).get("v", 0) or 0),
            "pm10": float(iaqi.get("pm10", {}).get("v", 0) or 0),
            "no2":  float(iaqi.get("no2",  {}).get("v", 0) or 0),
            "o3":   float(iaqi.get("o3",   {}).get("v", 0) or 0),
            "co":   float(iaqi.get("co",   {}).get("v", 0) or 0),
            "so2":  float(iaqi.get("so2",  {}).get("v", 0) or 0),
        }
    except:
        return {"aqi":75,"pm25":35,"pm10":50,"no2":15,"o3":40,"co":0.5,"so2":5}

@st.cache_data(ttl=3600)
def fetch_live_weather():
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude":LAT,"longitude":LON,
            "current":["temperature_2m","relative_humidity_2m","wind_speed_10m",
                       "surface_pressure","precipitation","weather_code"],
            "hourly":["temperature_2m","relative_humidity_2m",
                      "wind_speed_10m","precipitation_probability"],
            "forecast_days":4,"timezone":"Asia/Karachi"
        }, timeout=10)
        d = r.json(); c = d["current"]
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

@st.cache_data(ttl=3600)
def load_history(days=30):
    try:
        supabase = get_supabase()
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        res = (supabase.table("aqi_features")
               .select("timestamp,aqi,pm25,pm10,temperature,humidity,windspeed")
               .gte("timestamp", since).order("timestamp").execute())
        df = pd.DataFrame(res.data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except:
        return pd.DataFrame()

# ── Prediction ────────────────────────────────────────────────
def make_prediction(aqi_data, weather, models_tup, history_df):
    m24, m48, m72, scaler, features, metrics = models_tup
    if m24 is None:
        return aqi_data["aqi"], aqi_data["aqi"], aqi_data["aqi"]
    now      = datetime.now(timezone.utc)
    aqi_vals = [] if history_df.empty else history_df["aqi"].dropna().tolist()
    def lag(n):  return float(aqi_vals[-n])            if len(aqi_vals) >= n else aqi_data["aqi"]
    def roll(n): return float(np.mean(aqi_vals[-n:]))  if len(aqi_vals) >= n else aqi_data["aqi"]
    row = {
        "aqi":aqi_data["aqi"],"pm25":aqi_data["pm25"],"pm10":aqi_data["pm10"],
        "no2":aqi_data["no2"],"o3":aqi_data["o3"],"co":aqi_data["co"]/1000,"so2":aqi_data["so2"],
        "temperature":weather["temperature"],"humidity":weather["humidity"],
        "windspeed":weather["windspeed"],"pressure":weather["pressure"],
        "precipitation":weather["precipitation"],
        "hour":now.hour,"day_of_week":now.weekday(),"month":now.month,
        "is_weekend":int(now.weekday()>=5),
        "aqi_lag_1h":lag(1),"aqi_lag_3h":lag(3),"aqi_lag_6h":lag(6),
        "aqi_lag_24h":lag(24),"aqi_lag_48h":lag(48),
        "aqi_rolling_3h":roll(3),"aqi_rolling_6h":roll(6),
        "aqi_rolling_24h":roll(24),"aqi_rolling_48h":roll(48),
        "aqi_change_rate":aqi_data["aqi"]-lag(1),
        "aqi_change_6h":aqi_data["aqi"]-lag(6),
        "aqi_change_24h":aqi_data["aqi"]-lag(24),
        "temp_humidity":weather["temperature"]*weather["humidity"]/100,
        "wind_pressure":weather["windspeed"]*weather["pressure"]/1000,
        "is_morning":int(6<=now.hour<=9),"is_evening":int(17<=now.hour<=21),
        "is_night":int(now.hour>=22 or now.hour<=5),
        "is_winter":int(now.month in [11,12,1,2]),
        "is_summer":int(now.month in [5,6,7,8]),
        "is_monsoon":int(now.month in [7,8,9]),
    }
    try:
        X  = np.array([[row.get(f, 0) for f in features]])
        Xs = scaler.transform(X)
        return (max(0, float(m24.predict(Xs)[0])),
                max(0, float(m48.predict(Xs)[0])),
                max(0, float(m72.predict(Xs)[0])))
    except:
        return aqi_data["aqi"], aqi_data["aqi"], aqi_data["aqi"]

# ════════════════════════════════════════════════════════════
#  LAYOUT
# ════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;
            padding:0.8rem 0 1rem;border-bottom:2px solid rgba(2,136,209,0.15);margin-bottom:1rem;'>
    <div>
        <h1 style='font-size:1.9rem;margin:0;color:#1a3a4a;font-family:Space Grotesk,sans-serif;font-weight:800;'>
            🌿 Islamabad <span style='color:#0288d1;'>AQI</span> Predictor
        </h1>
        <p style='color:#5a8fa8;margin:0.25rem 0 0;font-size:0.88rem;'>
            Real-time Air Quality Index · 3-Day Forecast · Powered by ML
        </p>
    </div>
    <div style='background:linear-gradient(135deg,#e3f2fd,#e0f7fa);color:#0277bd;
                font-size:0.78rem;font-weight:700;padding:0.45rem 1rem;
                border-radius:20px;border:1px solid rgba(2,136,209,0.25);'>
        📡 Live Data
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch data
with st.spinner("🔄 Fetching live data..."):
    aqi_data   = fetch_live_aqi()
    weather    = fetch_live_weather()
    history_df = load_history(30)
    models_tup = load_models()

p24, p48, p72     = make_prediction(aqi_data, weather, models_tup, history_df)
cat, color, emoji = aqi_category(aqi_data["aqi"])
now_str           = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")
aqi_val           = aqi_data["aqi"]

# Alert banner
if aqi_val > 200:
    st.markdown(f"<div class='alert-danger'>🚨 <b>HAZARD ALERT</b> — AQI is <b>{aqi_val:.0f}</b>. Everyone must avoid all outdoor activity immediately.</div>", unsafe_allow_html=True)
elif aqi_val > 150:
    st.markdown(f"<div class='alert-danger'>⚠️ <b>HEALTH WARNING</b> — AQI is <b>{aqi_val:.0f}</b> (Unhealthy). Sensitive groups must stay indoors.</div>", unsafe_allow_html=True)
elif aqi_val > 100:
    st.markdown(f"<div class='alert-warn'>⚠️ <b>CAUTION</b> — AQI is <b>{aqi_val:.0f}</b>. Sensitive individuals should limit outdoor exposure.</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='alert-good'>✅ <b>AIR QUALITY</b> — AQI is <b>{aqi_val:.0f}</b> ({cat}). {aqi_advice(aqi_val)}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Gauge + Weather ────────────────────────────────────
col_gauge, col_weather = st.columns([1, 2], gap="large")

with col_gauge:
    st.markdown("<p class='section-title'>📍 Current AQI · Islamabad</p>", unsafe_allow_html=True)
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_data["aqi"],
        delta={"reference":p24,"valueformat":".0f","prefix":"24h: ",
               "increasing":{"color":"#f4511e"},"decreasing":{"color":"#43a047"}},
        title={"text":f"{emoji} {cat}","font":{"size":17,"color":FONT_COL,"family":"Space Grotesk"}},
        gauge={
            "axis":{"range":[0,300],"tickcolor":MUTED,"tickfont":{"color":MUTED,"size":10}},
            "bar":{"color":color,"thickness":0.26},
            "bgcolor":"rgba(240,247,255,0.6)","bordercolor":"rgba(2,136,209,0.2)",
            "steps":[
                {"range":[0,50],   "color":"rgba(67,160,71,0.12)"},
                {"range":[50,100], "color":"rgba(255,152,0,0.12)"},
                {"range":[100,150],"color":"rgba(251,140,0,0.12)"},
                {"range":[150,200],"color":"rgba(244,81,30,0.12)"},
                {"range":[200,300],"color":"rgba(198,40,40,0.12)"},
            ],
            "threshold":{"line":{"color":FONT_COL,"width":3},"thickness":0.8,"value":aqi_val},
        },
        number={"font":{"size":46,"color":color,"family":"Space Grotesk"},"suffix":" AQI"},
    ))
    fig_g.update_layout(
        height=300, margin=dict(l=20,r=20,t=40,b=20),
        paper_bgcolor=BG, plot_bgcolor=BG, font_color=FONT_COL,
    )
    st.plotly_chart(fig_g, use_container_width=True)
    st.markdown(f"<div style='text-align:center;color:{MUTED};font-size:0.75rem;font-family:Space Grotesk;margin-top:-0.5rem;'>Last updated: {now_str}</div>", unsafe_allow_html=True)

with col_weather:
    st.markdown("<p class='section-title'>🌤️ Current Weather · Islamabad</p>", unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    wicon = weather_icon(weather["weather_code"])
    with w1: st.metric(f"{wicon} Temperature", f"{weather['temperature']:.1f}°C")
    with w2: st.metric("💧 Humidity",           f"{weather['humidity']:.0f}%")
    with w3: st.metric("💨 Wind Speed",         f"{weather['windspeed']:.1f} km/h")
    with w4: st.metric("🌡️ Pressure",           f"{weather['pressure']:.0f} hPa")

    st.markdown("<br><p class='section-title'>🧪 Pollutant Levels</p>", unsafe_allow_html=True)
    poll_names  = ["PM2.5","PM10","NO₂","O₃","CO×100","SO₂"]
    poll_vals   = [aqi_data["pm25"],aqi_data["pm10"],aqi_data["no2"],
                   aqi_data["o3"],aqi_data["co"]*100,aqi_data["so2"]]
    poll_colors = ["#1e88e5","#00acc1","#9c27b0","#ffa726","#43a047","#e91e63"]
    fig_p = go.Figure(go.Bar(
        x=poll_names, y=poll_vals,
        marker_color=poll_colors,
        marker_line_color="rgba(0,0,0,0)",
        marker_opacity=0.85,
        text=[f"{v:.1f}" for v in poll_vals],
        textposition="outside",
        textfont={"color":FONT_COL,"size":11},
    ))
    fig_p.update_layout(
        height=220, margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font_color=FONT_COL, showlegend=False,
        xaxis=dict(showgrid=False, color=MUTED, tickfont=dict(size=11,color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor=GRID_COL, color=MUTED,
                   title=dict(text="μg/m³", font=dict(color=MUTED))),
    )
    st.plotly_chart(fig_p, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: 3-Day Forecast ─────────────────────────────────────
st.markdown("<p class='section-title'>📅 3-Day AQI Forecast</p>", unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns(3, gap="medium")
for col, (label, pred, tag) in zip([fc1,fc2,fc3],
        [("Tomorrow",p24,"24h"),("Day After",p48,"48h"),("In 3 Days",p72,"72h")]):
    fcat, fcol, femo = aqi_category(pred)
    with col:
        st.markdown(f"""
        <div class='forecast-card' style='border-top:4px solid {fcol};'>
            <p style='color:{MUTED};font-size:0.72rem;font-family:Space Grotesk;
                      letter-spacing:0.1em;font-weight:700;margin:0;'>{tag.upper()} FORECAST</p>
            <h3 style='color:{fcol};font-size:2.8rem;margin:0.3rem 0;
                       font-family:Space Grotesk;font-weight:900;'>{pred:.0f}</h3>
            <p style='color:{MUTED};font-size:0.8rem;margin:0;'>AQI · {label}</p>
            <p style='font-size:1.6rem;margin:0.5rem 0;'>{femo}</p>
            <span style='background:{fcol}18;color:{fcol};padding:0.25rem 0.9rem;
                         border-radius:20px;font-size:0.73rem;font-weight:700;
                         font-family:Space Grotesk;border:1px solid {fcol}35;'>{fcat}</span>
            <p style='color:{MUTED};font-size:0.72rem;margin:0.6rem 0 0;
                      line-height:1.4;'>{aqi_advice(pred)[:65]}...</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 3: History + Hourly ───────────────────────────────────
col_hist, col_hourly = st.columns([3, 2], gap="large")

with col_hist:
    st.markdown("<p class='section-title'>📈 30-Day AQI History</p>", unsafe_allow_html=True)
    if not history_df.empty:
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=history_df["timestamp"], y=history_df["aqi"],
            mode="lines", name="AQI",
            line=dict(color="#0288d1", width=2),
            fill="tozeroy", fillcolor="rgba(2,136,209,0.07)",
        ))
        for level, lc, ln in [(50,"#43a047","Good"),(100,"#ff9800","Moderate"),(150,"#f4511e","Unhealthy")]:
            fig_h.add_hline(y=level, line_dash="dot", line_color=lc, opacity=0.6,
                            annotation_text=ln, annotation_font_color=lc, annotation_font_size=10)
        fig_h.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor=BG, plot_bgcolor=BG, font_color=FONT_COL,
            hovermode="x unified",
            xaxis=dict(showgrid=False, color=MUTED, tickfont=dict(color=MUTED)),
            yaxis=dict(showgrid=True, gridcolor=GRID_COL, color=MUTED,
                       title=dict(text="AQI", font=dict(color=MUTED))),
        )
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Historical data loading...")

with col_hourly:
    st.markdown("<p class='section-title'>⏱️ 24h Weather Forecast</p>", unsafe_allow_html=True)
    hourly = weather.get("hourly", {})
    if hourly:
        temps  = hourly.get("temperature_2m", [])[:24]
        rain_p = hourly.get("precipitation_probability", [])[:24]
        fig_hw = go.Figure()
        fig_hw.add_trace(go.Scatter(
            x=list(range(len(temps))), y=temps,
            name="Temp °C", line=dict(color="#f4511e", width=2.5), yaxis="y1",
        ))
        fig_hw.add_trace(go.Bar(
            x=list(range(len(rain_p))), y=rain_p,
            name="Rain %", marker_color="rgba(2,136,209,0.25)", yaxis="y2",
        ))
        fig_hw.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor=BG, plot_bgcolor=BG, font_color=FONT_COL,
            hovermode="x unified",
            xaxis=dict(showgrid=False, color=MUTED,
                       tickvals=list(range(0,24,4)),
                       ticktext=[f"{h}:00" for h in range(0,24,4)],
                       tickfont=dict(color=MUTED)),
            yaxis=dict(showgrid=True, gridcolor=GRID_COL, color="#f4511e",
                       title=dict(text="Temperature °C", font=dict(color="#f4511e"))),
            yaxis2=dict(overlaying="y", side="right", color="#0288d1",
                        title=dict(text="Rain Prob %", font=dict(color="#0288d1")),
                        range=[0,100]),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1,
                        font=dict(color=MUTED)),
        )
        st.plotly_chart(fig_hw, use_container_width=True)
    else:
        st.info("Weather forecast loading...")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 4: Heatmap + Model Metrics ───────────────────────────
col_heat, col_metrics = st.columns([3, 2], gap="large")

with col_heat:
    st.markdown("<p class='section-title'>🗓️ AQI Heatmap (Hour vs Date)</p>", unsafe_allow_html=True)
    if not history_df.empty and len(history_df) > 7:
        hdf = history_df.copy()
        hdf["date"] = hdf["timestamp"].dt.date
        hdf["hour"] = hdf["timestamp"].dt.hour
        pivot = hdf.pivot_table(values="aqi", index="hour", columns="date", aggfunc="mean")
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[f"{h:02d}:00" for h in pivot.index],
            colorscale=[[0.0,"#43a047"],[0.33,"#ff9800"],[0.66,"#f4511e"],[1.0,"#b71c1c"]],
            colorbar=dict(
                title=dict(text="AQI", font=dict(color=MUTED)),
                tickfont=dict(color=MUTED),
            ),
            hoverongaps=False,
        ))
        fig_heat.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor=BG, plot_bgcolor=BG, font_color=FONT_COL,
            xaxis=dict(showgrid=False, color=MUTED, nticks=10,
                       tickfont=dict(color=MUTED),
                       title=dict(text="Date", font=dict(color=MUTED))),
            yaxis=dict(showgrid=False, color=MUTED, autorange="reversed",
                       title=dict(text="Hour of Day", font=dict(color=MUTED))),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Need more history for heatmap...")

with col_metrics:
    st.markdown("<p class='section-title'>🤖 Model Performance</p>", unsafe_allow_html=True)
    _, _, _, _, _, met = models_tup
    if met:
        bm = met.get("best_model", "N/A")
        st.markdown(f"""
        <div style='background:rgba(2,136,209,0.07);border:1px solid rgba(2,136,209,0.25);
                    border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.8rem;'>
            <p style='color:{MUTED};font-size:0.7rem;font-family:Space Grotesk;
                      margin:0;letter-spacing:0.1em;font-weight:700;'>BEST MODEL</p>
            <p style='color:#0288d1;font-size:1.05rem;font-family:Space Grotesk;
                      margin:0.2rem 0 0;font-weight:800;'>{bm}</p>
        </div>""", unsafe_allow_html=True)
        for horizon, key in [("24h Forecast","model_24h"),
                              ("48h Forecast","model_48h"),
                              ("72h Forecast","model_72h")]:
            m = met.get(key, {})
            if m:
                r2   = m.get("r2",   0)
                rmse = m.get("rmse", 0)
                r2c  = "#43a047" if r2>0.7 else "#ff9800" if r2>0.4 else "#f4511e"
                st.markdown(f"""
                <div style='background:rgba(240,247,255,0.8);
                            border:1px solid rgba(2,136,209,0.15);
                            border-radius:10px;padding:0.7rem 1rem;margin:0.4rem 0;'>
                    <span style='color:{MUTED};font-size:0.78rem;
                                 font-family:Space Grotesk;font-weight:600;'>{horizon}</span>
                    <span style='float:right;'>
                        <span style='color:{r2c};font-family:Space Grotesk;
                                     font-size:0.82rem;font-weight:800;'>R²={r2:.3f}</span>
                        <span style='color:{MUTED};font-size:0.73rem;
                                     margin-left:0.5rem;'>RMSE={rmse:.1f}</span>
                    </span>
                </div>""", unsafe_allow_html=True)
        n_feat = met.get("n_features", 0)
        train  = met.get("train_size",  0)
        st.markdown(f"""
        <div style='margin-top:0.8rem;color:{MUTED};font-size:0.78rem;line-height:1.5;'>
            📊 Trained on <b style='color:#0288d1;'>{train:,}</b> hourly records
            using <b style='color:#0288d1;'>{n_feat}</b> features
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Model metrics not available.")

st.markdown("<br>", unsafe_allow_html=True)

# ── AQI Scale Reference ───────────────────────────────────────
st.markdown("<p class='section-title'>📋 AQI Scale Reference</p>", unsafe_allow_html=True)
scale_cols = st.columns(6)
for col, (rng,cat,clr,ico,desc) in zip(scale_cols,[
    ("0–50",   "Good",           "#43a047","😊","No health risk"),
    ("51–100", "Moderate",       "#ff9800","😐","Sensitive caution"),
    ("101–150","Unhealthy (SG)", "#fb8c00","😷","Sensitive groups"),
    ("151–200","Unhealthy",      "#f4511e","🤢","Everyone affected"),
    ("201–300","Very Unhealthy", "#c62828","🚨","Health emergency"),
    ("300+",   "Hazardous",      "#4e342e","☠️","Stay indoors"),
]):
    with col:
        st.markdown(f"""
        <div style='background:{clr}10;border:1px solid {clr}35;border-top:3px solid {clr};
                    border-radius:11px;padding:0.7rem 0.4rem;text-align:center;
                    box-shadow:0 2px 10px {clr}12;'>
            <p style='font-size:1.3rem;margin:0;'>{ico}</p>
            <p style='color:{clr};font-family:Space Grotesk;font-size:0.73rem;
                      font-weight:800;margin:0.25rem 0;'>{rng}</p>
            <p style='color:{FONT_COL};font-size:0.7rem;margin:0;font-weight:500;'>{cat}</p>
            <p style='color:{MUTED};font-size:0.63rem;margin:0.2rem 0 0;'>{desc}</p>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center;color:{MUTED};font-size:0.73rem;font-family:Space Grotesk;
            padding:1rem;border-top:1px solid rgba(2,136,209,0.15);'>
    🌿 Islamabad AQI Predictor · ML-powered · Data: AQICN + Open-Meteo · Built with Streamlit · © 2026
</div>""", unsafe_allow_html=True)