# hybrid_data_pipeline.py
import pandas as pd
from sodapy import Socrata
from fredapi import Fred
from datetime import datetime, timedelta

# ========================
# HARD-CODED CONFIGURATION
# ========================
FRED_API_KEY = "91bb2c5920fb8f843abdbbfdfcab5345"
SODAPY_API_KEY = "1h3ijfuomvrayys00k5cvk38y3nl2wpk0whnlosfj6o7tuuu7n"
SECONDARY_EMAIL = "segaab120@gmail.com"  # For contact/reference only

# ========================
# INITIALIZE CLIENTS
# ========================
# FRED Client
fred = Fred(api_key=FRED_API_KEY)

# Socrata Client (public data, no auth required)
client = Socrata(
    "publicreporting.cftc.gov",
    SODAPY_API_KEY,
    timeout=60
)

# ========================
# FUNCTIONS
# ========================
def get_fred_series(series_id: str, start_date: str = None, end_date: str = None):
    """
    Fetch a time series from FRED API.
    """
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = pd.DataFrame(data, columns=["value"])
    df.index.name = "date"
    df.reset_index(inplace=True)
    return df

def get_cot_data(dataset_id="6dca-aqww", limit=1000):
    """
    Fetch COT data from Socrata Public API.
    """
    results = client.get(dataset_id, limit=limit)
    return pd.DataFrame.from_records(results)

def get_last_two_reports():
    """
    Fetch the latest and previous COT reports based on release rules.
    """
    edt_now = datetime.utcnow() - timedelta(hours=4)
    last_friday = edt_now - timedelta(days=(edt_now.weekday() - 4) % 7)
    report_time = last_friday.replace(hour=15, minute=30, second=0)

    if edt_now.weekday() == 4 and edt_now < report_time:
        last_friday -= timedelta(weeks=1)

    latest_tuesday = last_friday - timedelta(days=3)
    previous_tuesday = latest_tuesday - timedelta(weeks=1)

    latest_str = latest_tuesday.strftime('%Y-%m-%d')
    prev_str = previous_tuesday.strftime('%Y-%m-%d')

    latest_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{latest_str}'")
    prev_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{prev_str}'")

    latest_df = pd.DataFrame.from_records(latest_result) if latest_result else None
    prev_df = pd.DataFrame.from_records(prev_result)

    return latest_df, prev_df

# ========================
# MAIN EXECUTION EXAMPLE
# ========================
if __name__ == "__main__":
    # Example FRED data
    fred_df = get_fred_series("DGS10", start_date="2020-01-01", end_date="2023-12-31")
    print("FRED Data Sample:\n", fred_df.head())

    # Example COT data
    cot_df = get_cot_data(limit=5)
    print("COT Data Sample:\n", cot_df.head())

    # Latest two reports
    latest, previous = get_last_two_reports()
    print("Latest COT Report:\n", latest)
    print("Previous COT Report:\n", previous)
