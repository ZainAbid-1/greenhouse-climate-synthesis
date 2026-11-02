"""
Fetch historical weather data from Open-Meteo's Historical Weather API
(ERA5 reanalysis, archive-api.open-meteo.com/v1/archive) and save it as a
single CSV dataset.

Docs: https://open-meteo.com/en/docs/historical-weather-api
No API key required. Free for non-commercial use.

Usage:
    pip install requests pandas
    python fetch_open_meteo.py
"""

import os
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration - edit these to change location, dates, or variables
# ---------------------------------------------------------------------------
LATITUDE = 33.6456
LONGITUDE = 72.9903
START_DATE = "2024-06-01"
END_DATE = "2026-06-01"

# Hourly variables to pull. Add/remove as needed - full list at:
# https://open-meteo.com/en/docs/historical-weather-api
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
]

TIMEZONE = "Asia/Karachi"
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "openmeteo_islamabad_hourly.csv")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ---------------------------------------------------------------------------
# Fetch in yearly chunks (keeps each request small/robust; the API itself
# can handle multi-year ranges, but chunking makes retries cheap and avoids
# any single oversized response).
# ---------------------------------------------------------------------------
def daterange_chunks(start: str, end: str, chunk_days: int = 365):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_ts)
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt + pd.Timedelta(days=1)


def fetch_chunk(start: str, end: str, retries: int = 3) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    for attempt in range(1, retries + 1):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return pd.DataFrame(data["hourly"])
        print(f"  attempt {attempt} failed ({resp.status_code}): {resp.text[:200]}")
        time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {start} to {end} after {retries} attempts")


def main():
    frames = []
    for start, end in daterange_chunks(START_DATE, END_DATE):
        print(f"Fetching {start} -> {end} ...")
        frames.append(fetch_chunk(start, end))

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()