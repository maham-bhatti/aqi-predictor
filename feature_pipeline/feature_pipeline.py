import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ── Config ──────────────────────────────────────────────
AQICN_TOKEN   = os.getenv("AQICN_TOKEN")
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")
CITY          = "Islamabad"
LAT, LON      = 33.6844, 73.0479

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 1. Fetch AQI Data from AQICN ────────────────────────
def fetch_aqi():
    url = f"https://api.waqi.info/feed/islamabad/?token={AQICN_TOKEN}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]

    iaqi = data.get("iaqi", {})
    return {
        "aqi":  float(data.get("aqi", 0) or 0),
        "pm25": float(iaqi.get("pm25", {}).get("v", 0) or 0),
        "pm10": float(iaqi.get("pm10", {}).get("v", 0) or 0),
        "no2":  float(iaqi.get("no2",  {}).get("v", 0) or 0),
        "o3":   float(iaqi.get("o3",   {}).get("v", 0) or 0),
        "co":   float(iaqi.get("co",   {}).get("v", 0) or 0),
        "so2":  float(iaqi.get("so2",  {}).get("v", 0) or 0),
    }

# ── 2. Fetch Weather from Open-Meteo ────────────────────
def fetch_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  LAT,
        "longitude": LON,
        "current":   [
            "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "surface_pressure", "precipitation"
        ],
        "timezone": "Asia/Karachi"
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    c = r.json()["current"]
    return {
        "temperature": float(c.get("temperature_2m",        0) or 0),
        "humidity":    float(c.get("relative_humidity_2m",  0) or 0),
        "windspeed":   float(c.get("wind_speed_10m",        0) or 0),
        "pressure":    float(c.get("surface_pressure",      0) or 0),
        "precipitation": float(c.get("precipitation",       0) or 0),
    }

# ── 3. Fetch Recent AQI from Supabase for Lag Features ──
def fetch_recent_aqi(n=24):
    res = (supabase.table("aqi_features")
           .select("timestamp, aqi")
           .order("timestamp", desc=True)
           .limit(n)
           .execute())
    rows = res.data or []
    return sorted(rows, key=lambda x: x["timestamp"])   # oldest first

# ── 4. Compute Lag & Rolling Features ───────────────────
def compute_lag_features(current_aqi, history):
    vals = [r["aqi"] for r in history if r["aqi"] is not None]

    def lag(n):
        return float(vals[-n]) if len(vals) >= n else None

    def roll(n):
        return float(np.mean(vals[-n:])) if len(vals) >= n else None

    prev = vals[-1] if vals else current_aqi
    change_rate = float(current_aqi - prev) if prev else 0.0

    return {
        "aqi_lag_1h":       lag(1),
        "aqi_lag_3h":       lag(3),
        "aqi_lag_6h":       lag(6),
        "aqi_lag_24h":      lag(24),
        "aqi_rolling_3h":   roll(3),
        "aqi_rolling_6h":   roll(6),
        "aqi_rolling_24h":  roll(24),
        "aqi_change_rate":  change_rate,
    }

# ── 5. Compute Time Features ────────────────────────────
def compute_time_features(dt: datetime):
    return {
        "hour":        dt.hour,
        "day_of_week": dt.weekday(),   # 0=Mon … 6=Sun
        "month":       dt.month,
        "is_weekend":  dt.weekday() >= 5,
    }

# ── 6. Store in Supabase ─────────────────────────────────
def store_features(record: dict):
    # upsert so re-runs don't create duplicates
    supabase.table("aqi_features").upsert(
        record, on_conflict="timestamp"
    ).execute()
    print(f"✅  Stored record for {record['timestamp']}")

# ── Main ─────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    print(f"🚀  Running feature pipeline at {now.isoformat()}")

    aqi_data     = fetch_aqi()
    weather_data = fetch_weather()
    history      = fetch_recent_aqi(24)
    lag_features = compute_lag_features(aqi_data["aqi"], history)
    time_features = compute_time_features(now)

    record = {
        "timestamp": now.isoformat(),
        "city":      CITY,
        **aqi_data,
        **weather_data,
        **time_features,
        **lag_features,
        # targets filled later by backfill / future pipeline runs
        "aqi_next_24h": None,
        "aqi_next_48h": None,
        "aqi_next_72h": None,
    }

    store_features(record)
    print("✅  Feature pipeline completed successfully!")
    print(f"    AQI={aqi_data['aqi']}  PM2.5={aqi_data['pm25']}  "
          f"Temp={weather_data['temperature']}°C")

if __name__ == "__main__":
    main()