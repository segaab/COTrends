# hybrid_data_pipeline.py
import os
import requests
import pandas as pd
from sodapy import Socrata
from datetime import datetime, timedelta

# ==== HARDCODED CREDENTIALS ====
FRED_API_KEY = "91bb2c5920fb8f843abdbbfdfcab5345"
SODAPY_API_KEY = "1h3ijfuomvrayys00k5cvk38y3nl2wpk0whnlosfj6o7tuuu7n"
SODAPY_EMAIL = "segaab120@gmail.com"

# ==== FRED Data Fetch Function ====
def fetch_fred_series(series_id, start_date=None, end_date=None):
    """
    Fetch data from FRED API for a given series ID.
    """
    base_url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date

    print(f"Fetching FRED series {series_id}...")
    r = requests.get(base_url, params=params)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["observations"])
    if not df.empty:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
    return df

# ==== COT Data Fetch Function (Socrata API) ====
def fetch_cot_data(limit=5000):
    """
    Fetch latest COT report data from CFTC Socrata endpoint.
    """
    client = Socrata("publicreporting.cftc.gov", SODAPY_API_KEY, username=SODAPY_EMAIL)
    dataset_id = "6dca-aqww"

    print(f"Fetching latest {limit} rows from COT dataset...")
    results = client.get(dataset_id, limit=limit)
    df = pd.DataFrame.from_records(results)
    if "report_date_as_yyyy_mm_dd" in df.columns:
        df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    return df

# ==== Combined Hybrid Data Pipeline ====
def run_pipeline():
    """
    Runs the hybrid data pipeline fetching both FRED and COT data.
    """
    # Example FRED series IDs
    fred_series = {
        "FEDFUNDS": "Effective Federal Funds Rate",
        "DGS10": "10-Year Treasury Constant Maturity Rate"
    }
    start_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    fred_data = {}
    for series_id, desc in fred_series.items():
        fred_data[series_id] = fetch_fred_series(series_id, start_date, end_date)

    cot_data = fetch_cot_data(limit=2000)

    return fred_data, cot_data


if __name__ == "__main__":
    fred_data, cot_data = run_pipeline()

    print("\n=== FRED Data Sample ===")
    for series_id, df in fred_data.items():
        print(f"\n{series_id} ({len(df)} rows):")
        print(df.head())

    print("\n=== COT Data Sample ===")
    print(cot_data.head())
