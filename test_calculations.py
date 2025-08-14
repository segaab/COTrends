import pandas as pd
import os
import datetime
from datetime import timedelta
from sodapy import Socrata
from fredapi import Fred
import requests

# ===========================
# API KEYS (hardcoded or env)
# ===========================
FRED_API_KEY = "91bb2c5920fb8f843abdbbfdfcab5345"
SODAPY_APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"

# ===========================
# INITIALIZE CLIENTS
# ===========================
fred = Fred(api_key=FRED_API_KEY)
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# ===========================
# REQUIRED FIELDS FOR COT
# ===========================
required_fields = [
    "market_and_exchange_names",
    "report_date_as_yyyy_mm_dd",
    "noncomm_positions_long_all",
    "noncomm_positions_short_all"
]

# ===========================
# FUNCTION: FETCH LAST TWO COT REPORTS
# ===========================
def get_last_two_cot_reports(client):
    edt_now = datetime.datetime.utcnow() - timedelta(hours=4)
    last_friday = edt_now - timedelta(days=(edt_now.weekday() - 4) % 7)

    # Friday before 3:30 PM ET uses previous week
    report_time = last_friday.replace(hour=15, minute=30, second=0)
    if edt_now.weekday() == 4 and edt_now < report_time:
        last_friday -= timedelta(weeks=1)

    latest_tuesday = last_friday - timedelta(days=3)
    previous_tuesday = latest_tuesday - timedelta(weeks=1)

    latest_str = latest_tuesday.strftime("%Y-%m-%d")
    previous_str = previous_tuesday.strftime("%Y-%m-%d")

    latest_result = client.get(
        "6dca-aqww",
        where=f"report_date_as_yyyy_mm_dd='{latest_str}'",
        select=",".join(required_fields)
    )
    previous_result = client.get(
        "6dca-aqww",
        where=f"report_date_as_yyyy_mm_dd='{previous_str}'",
        select=",".join(required_fields)
    )

    latest_df = pd.DataFrame.from_records(latest_result) if latest_result else None
    previous_df = pd.DataFrame.from_records(previous_result) if previous_result else None

    return latest_df, previous_df

# ===========================
# FUNCTION: FETCH FRED SERIES
# ===========================
def fetch_fred_series(series_id, start_date="2010-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    df = pd.DataFrame(data, columns=["value"])
    df.index.name = "date"
    df.reset_index(inplace=True)
    return df

# ===========================
# FUNCTION: FETCH INVESTMENT BANK CSVS
# ===========================
def load_bank_csv(file_path="bank_balance_sheets.csv"):
    """
    Load bank loan portfolios and deposit data from CSV.
    """
    return pd.read_csv(file_path)

def load_interest_rate_csv(file_path="bank_interest_rates.csv"):
    """
    Load bank loan prime rate CSV.
    """
    return pd.read_csv(file_path)

# ===========================
# FUNCTION: FETCH CORPORATE BOND DATA
# ===========================
def load_corporate_bonds_csv(file_path="corporate_bonds.csv"):
    return pd.read_csv(file_path)

# ===========================
# FUNCTION: FETCH MARKET DATA (Fed Funds, SOFR, Implied Volatility)
# ===========================
def load_market_data_csv(file_path="market_data.csv"):
    return pd.read_csv(file_path)

# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    # 1. Fetch last two COT reports
    latest_cot, previous_cot = get_last_two_cot_reports(client)
    print("Latest COT Report:\n", latest_cot.head() if latest_cot is not None else "No data")
    print("\nPrevious COT Report:\n", previous_cot.head() if previous_cot is not None else "No data")

    # 2. Fetch FRED series examples
    fed_funds = fetch_fred_series("FEDFUNDS", start_date="2013-01-01")
    sofr = fetch_fred_series("SOFR", start_date="2018-01-01")
    print("\nFed Funds Sample:\n", fed_funds.head())
    print("\nSOFR Sample:\n", sofr.head())

    # 3. Load bank balance sheets and interest rates
    bank_df = load_bank_csv()
    rates_df = load_interest_rate_csv()
    print("\nBank Balance Sheets Sample:\n", bank_df.head())
    print("\nBank Interest Rates Sample:\n", rates_df.head())

    # 4. Load corporate bonds and market data
    corporate_df = load_corporate_bonds_csv()
    market_df = load_market_data_csv()
    print("\nCorporate Bonds Sample:\n", corporate_df.head())
    print("\nMarket Data Sample:\n", market_df.head())
