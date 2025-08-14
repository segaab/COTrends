import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta
from sodapy import Socrata
from fredapi import Fred
import os

# ===========================
# API KEYS
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
    "noncomm_positions_short_all",
    "comm_positions_long_all",
    "comm_positions_short_all",
    "nonrept_positions_long_all",
    "nonrept_positions_short_all"
]

# ===========================
# FUNCTIONS
# ===========================

def get_last_two_cot_reports(client):
    edt_now = datetime.datetime.utcnow() - timedelta(hours=4)
    last_friday = edt_now - timedelta(days=(edt_now.weekday() - 4) % 7)

    report_time = last_friday.replace(hour=15, minute=30, second=0)
    if edt_now.weekday() == 4 and edt_now < report_time:
        last_friday -= timedelta(weeks=1)

    latest_tuesday = last_friday - timedelta(days=3)
    previous_tuesday = latest_tuesday - timedelta(weeks=1)

    latest_str = latest_tuesday.strftime("%Y-%m-%d")
    previous_str = previous_tuesday.strftime("%Y-%m-%d")

    try:
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
    except Exception as e:
        st.error(f"Error fetching COT data: {e}")
        latest_result = []
        previous_result = []

    latest_df = pd.DataFrame.from_records(latest_result) if latest_result else pd.DataFrame(columns=required_fields)
    previous_df = pd.DataFrame.from_records(previous_result) if previous_result else pd.DataFrame(columns=required_fields)

    return latest_df, previous_df

def fetch_fred_series(series_id, start_date="2010-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    try:
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        df = pd.DataFrame(data, columns=["value"])
        df.index.name = "date"
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.warning(f"FRED series {series_id} could not be loaded: {e}")
        return pd.DataFrame(columns=["date", "value"])

def load_csv(file_path, default_columns):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.warning(f"{file_path} not found. Returning empty DataFrame.")
        return pd.DataFrame(columns=default_columns)

# ===========================
# STREAMLIT APP
# ===========================
st.title("Hybrid Market Data Dashboard")

# 1. COT Data
st.header("Commitments of Traders (COT) Reports")
latest_cot, previous_cot = get_last_two_cot_reports(client)
st.subheader("Latest COT Report")
st.dataframe(latest_cot if not latest_cot.empty else "No data available.")

st.subheader("Previous COT Report")
st.dataframe(previous_cot if not previous_cot.empty else "No data available.")

# 2. FRED Series
st.header("FRED Series Data")
fed_funds = fetch_fred_series("FEDFUNDS", start_date="2013-01-01")
sofr = fetch_fred_series("SOFR", start_date="2018-01-01")

st.subheader("Fed Funds Rate")
st.line_chart(fed_funds.set_index("date")["value"] if not fed_funds.empty else pd.DataFrame())

st.subheader("SOFR Rate")
st.line_chart(sofr.set_index("date")["value"] if not sofr.empty else pd.DataFrame())

# 3. Bank Data
st.header("Investment Bank Data")
bank_df = load_csv("bank_balance_sheets.csv", ["Loan Portfolio Value", "Deposit Base Value"])
rates_df = load_csv("bank_interest_rates.csv", ["Bank", "Date", "Loan Prime Rate"])

st.subheader("Bank Balance Sheets")
st.dataframe(bank_df)

st.subheader("Bank Interest Rates")
st.dataframe(rates_df)

# 4. Corporate Bonds & Market Data
st.header("Corporate Bonds & Market Data")
corporate_df = load_csv("corporate_bonds.csv", ["Bond", "Date", "Yield"])
market_df = load_csv("market_data.csv", ["Date", "Fed Funds Futures", "SOFR", "Implied Volatility"])

st.subheader("Corporate Bonds")
st.dataframe(corporate_df)

st.subheader("Market Data")
st.dataframe(market_df), market_df.head())
