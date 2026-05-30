import os
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

AQICN_TOKEN  = os.getenv("AQICN_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LAT, LON     = 33.6844, 73.0479

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Fetch historical weather from Open-Meteo Archive ────────
def fetch_historical_weather(date_str):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": date_str,
        "end_date":   date_str,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "surface_pressure",
            "precipitation",
            "rain",
            "showers",
        ],
        "timezone": "Asia/Karachi"
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["hourly"]

# ── Fetch historical air quality from Open-Meteo ────────────
def fetch_historical_aqi(date_str):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": date_str,
        "end_date":   date_str,
        "hourly": [
            "pm10",
            "pm2_5",
            "nitrogen_dioxide",
            "ozone",
            "carbon_monoxide",
            "sulphur_dioxide",
            "european_aqi",
        ],
        "timezone": "Asia/Karachi"
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["hourly"]

# ── Build records for one day ────────────────────────────────
def build_day_records(date_str, weather, air):
    records = []
    hours = len(weather["time"])

    aqi_list = air.get("european_aqi", [None] * hours)

    for i in range(hours):
        ts_str = weather["time"][i]
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)

        aqi_val = float(aqi_list[i] or 0)

        # ── Precipitation: sum all rain types ───────────────
        precip = (
            float(weather.get("precipitation", [0]*hours)[i] or 0) +
            float(weather.get("rain",          [0]*hours)[i] or 0) +
            float(weather.get("showers",       [0]*hours)[i] or 0)
        )

        # ── CO: convert μg/m³ → mg/m³ ───────────────────────
        co_val = float(air.get("carbon_monoxide", [0]*hours)[i] or 0) / 1000

        # ── Lag & rolling from records built so far ──────────
        prev_vals = [r["aqi"] for r in records if r["aqi"] is not None]

        def lag(n):
            return float(prev_vals[-n]) if len(prev_vals) >= n else None

        def roll(n):
            return float(np.mean(prev_vals[-n:])) if len(prev_vals) >= n else None

        prev_aqi = prev_vals[-1] if prev_vals else aqi_val
        change_rate = float(aqi_val - prev_aqi)

        record = {
            "timestamp":       ts.isoformat(),
            "city":            "Islamabad",

            # ── AQI & Pollutants ────────────────────────────
            "aqi":   aqi_val,
            "pm25":  float(air.get("pm2_5",            [0]*hours)[i] or 0),
            "pm10":  float(air.get("pm10",             [0]*hours)[i] or 0),
            "no2":   float(air.get("nitrogen_dioxide", [0]*hours)[i] or 0),
            "o3":    float(air.get("ozone",            [0]*hours)[i] or 0),
            "co":    co_val,
            "so2":   float(air.get("sulphur_dioxide",  [0]*hours)[i] or 0),

            # ── Weather ─────────────────────────────────────
            "temperature":   float(weather.get("temperature_2m",       [0]*hours)[i] or 0),
            "humidity":      float(weather.get("relative_humidity_2m", [0]*hours)[i] or 0),
            "windspeed":     float(weather.get("wind_speed_10m",       [0]*hours)[i] or 0),
            "pressure":      float(weather.get("surface_pressure",     [0]*hours)[i] or 0),
            "precipitation": round(precip, 3),

            # ── Time features ───────────────────────────────
            "hour":        ts.hour,
            "day_of_week": ts.weekday(),
            "month":       ts.month,
            "is_weekend":  ts.weekday() >= 5,

            # ── Lag features ────────────────────────────────
            "aqi_lag_1h":  lag(1),
            "aqi_lag_3h":  lag(3),
            "aqi_lag_6h":  lag(6),
            "aqi_lag_24h": lag(24),

            # ── Rolling averages ────────────────────────────
            "aqi_rolling_3h":  roll(3),
            "aqi_rolling_6h":  roll(6),
            "aqi_rolling_24h": roll(24),

            # ── Derived ─────────────────────────────────────
            "aqi_change_rate": change_rate,

            # ── Targets (filled after all records built) ────
            "aqi_next_24h": None,
            "aqi_next_48h": None,
            "aqi_next_72h": None,
        }
        records.append(record)

    return records

# ── Update target values ─────────────────────────────────────
def update_targets(all_records):
    n = len(all_records)
    for i, rec in enumerate(all_records):
        rec["aqi_next_24h"] = all_records[i + 24]["aqi"] if i + 24 < n else None
        rec["aqi_next_48h"] = all_records[i + 48]["aqi"] if i + 48 < n else None
        rec["aqi_next_72h"] = all_records[i + 72]["aqi"] if i + 72 < n else None
    return all_records

# ── Upload to Supabase in batches ────────────────────────────
def upload_batch(records, batch_size=100):
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        supabase.table("aqi_features").upsert(
            batch, on_conflict="timestamp"
        ).execute()
        print(f"  ✅  Uploaded records {i} → {i + len(batch)}")

# ── Main backfill ─────────────────────────────────────────────
def main():
    # 1 full year back from today — covers all 4 seasons
    end_date   = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=365)

    print(f"📅  Backfilling {start_date} → {end_date} for Islamabad")
    print(f"     Expected records: ~{365 * 24:,} hourly rows\n")

    all_records = []
    current     = start_date
    skipped     = 0

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        try:
            weather    = fetch_historical_weather(date_str)
            air        = fetch_historical_aqi(date_str)
            day_recs   = build_day_records(date_str, weather, air)
            all_records.extend(day_recs)
            print(f"  ✅  {date_str} — {len(day_recs)} records")
        except Exception as e:
            print(f"  ⚠️   {date_str} — skipped: {e}")
            skipped += 1
        current += timedelta(days=1)

    print(f"\n📊  Total records built: {len(all_records):,}  (skipped {skipped} days)")

    # Fill in target columns
    all_records = update_targets(all_records)

    # Verify data quality before uploading
    non_zero_precip = sum(1 for r in all_records if r["precipitation"] > 0)
    non_zero_co     = sum(1 for r in all_records if r["co"] > 0)
    print(f"\n🔍  Data quality check:")
    print(f"     Records with precipitation > 0 : {non_zero_precip:,}")
    print(f"     Records with CO > 0            : {non_zero_co:,}")
    print(f"     Records with target 24h filled : "
          f"{sum(1 for r in all_records if r['aqi_next_24h'] is not None):,}")

    # Upload to Supabase
    print(f"\n📤  Uploading {len(all_records):,} records to Supabase...")
    upload_batch(all_records)

    print(f"\n🎉  Backfill complete!")
    print(f"     Total records: {len(all_records):,}")
    print(f"     Date range   : {start_date} → {end_date}")
    print(f"     Seasons      : All 4 seasons covered ✅")

if __name__ == "__main__":
    main()