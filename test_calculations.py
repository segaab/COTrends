import pandas as pd
import requests
from sodapy import Socrata
from datetime import datetime, timedelta

# ==============================
# HARDCODED API KEYS
# ==============================
FRED_API_KEY = "91bb2c5920fb8f843abdbbfdfcab5345"
SODAPY_APP_TOKEN = "1h3ijfuomvrayys00k5cvk38y3nl2wpk0whnlosfj6o7tuuu7n"

# ==============================
# FRED DATA FETCHING FUNCTION
# ==============================
def fetch_fred_data(series_id, start_date="2000-01-01", end_date=None):
    """
    Fetch time series data from the FRED API.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["observations"])
    df = df[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df

# ==============================
# SOCRATA (COT DATA) FETCHING FUNCTION
# ==============================
def get_last_two_cot_reports():
    """
    Retrieve the last two COT reports from Socrata API.
    """
    client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

    edt_now = datetime.utcnow() - timedelta(hours=4)  # ET time
    last_friday = edt_now - timedelta(days=(edt_now.weekday() - 4) % 7)

    report_time = last_friday.replace(hour=15, minute=30, second=0)
    if edt_now.weekday() == 4 and edt_now < report_time:
        last_friday -= timedelta(weeks=1)

    latest_tuesday = last_friday - timedelta(days=3)
    previous_tuesday = latest_tuesday - timedelta(weeks=1)

    latest_str = latest_tuesday.strftime("%Y-%m-%d")
    previous_str = previous_tuesday.strftime("%Y-%m-%d")

    latest_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{latest_str}'")
    previous_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{previous_str}'")

    latest_df = pd.DataFrame.from_records(latest_result) if latest_result else None
    previous_df = pd.DataFrame.from_records(previous_result)

    return latest_df, previous_df

# ==============================
# HYBRID PIPELINE EXECUTION
# ==============================
if __name__ == "__main__":
    # Example: Fetch US GDP data from FRED
    gdp_df = fetch_fred_data("GDP", start_date="2010-01-01")
    print("\n📈 GDP Data Sample:")
    print(gdp_df.head())

    # Example: Fetch last two COT reports
    latest_cot, previous_cot = get_last_two_cot_reports()
    print("\n📊 Latest COT Report:")
    print(latest_cot.head() if latest_cot is not None else "No latest report available.")

    print("\n📊 Previous COT Report:")
    print(previous_cot.head())
