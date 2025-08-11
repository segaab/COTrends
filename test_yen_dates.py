import streamlit as st
import pandas as pd
import numpy as np
import logging
import datetime
from sodapy import Socrata
from yahooquery import Ticker
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Hardcoded Socrata API key (your token)
APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"

# Initialize Socrata client
client = Socrata("publicreporting.cftc.gov", APP_TOKEN, timeout=60)

# Assets to analyze (example: Gold and Silver plus others)
assets_cot_names = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    # Add your other assets here...
}

# Lookback window in days used in calculations (adjust as needed)
LOOKBACK_DAYS = 30

def fetch_cot_data(market_name, start_date, end_date):
    """
    Fetch COT data for given market between start_date and end_date (inclusive).
    start_date/end_date: datetime.date objects.
    Return DataFrame with report_date_as_yyyy_mm_dd as datetime.date.
    """
    logger.info(f"Fetching COT data for {market_name} from {start_date} to {end_date}")
    try:
        where_clause = (
            f"market_and_exchange_names = '{market_name}' AND "
            f"report_date_as_yyyy_mm_dd >= '{start_date.isoformat()}' AND "
            f"report_date_as_yyyy_mm_dd <= '{end_date.isoformat()}'"
        )
        results = client.get(
            "6dca-aqww",
            where=where_clause,
            order="report_date_as_yyyy_mm_dd ASC",
            limit=1000
        )
        if not results:
            logger.warning(f"No COT data found for {market_name} in given date range.")
            return pd.DataFrame()
        df = pd.DataFrame.from_records(results)
        # Convert report date to datetime.date
        df['report_date'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd']).dt.date
        # Convert open_interest_all to numeric (some fields might be strings)
        df['open_interest_all'] = pd.to_numeric(df['open_interest_all'], errors='coerce')
        return df
    except Exception as e:
        logger.error(f"Error fetching COT data for {market_name}: {e}")
        return pd.DataFrame()

def fetch_price_data(ticker_symbol, start_date, end_date):
    """
    Fetch daily price data from yahooquery between start_date and end_date.
    Returns DataFrame with date as datetime.date.
    """
    logger.info(f"Fetching Yahoo price data for {ticker_symbol} from {start_date} to {end_date}")
    try:
        ticker = Ticker(ticker_symbol)
        hist = ticker.history(start=start_date.isoformat(), end=end_date.isoformat(), interval='1d')
        if hist.empty:
            logger.warning(f"No price data for {ticker_symbol} in given date range.")
            return pd.DataFrame()
        df = hist.reset_index()
        # Normalize date to datetime.date
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    except Exception as e:
        logger.error(f"Error fetching Yahoo data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def merge_cot_price(cot_df, price_df):
    """
    Merge weekly COT open interest to daily price data by forward filling open_interest_all
    for each day until next COT report.
    """
    if cot_df.empty or price_df.empty:
        logger.warning("Empty DataFrame passed to merge_cot_price; returning price_df as is.")
        return price_df

    cot_df_sorted = cot_df.sort_values('report_date')[['report_date', 'open_interest_all']].drop_duplicates()

    # Create a DataFrame with all dates from min to max of price_df
    full_dates = pd.DataFrame({'date': pd.date_range(price_df['date'].min(), price_df['date'].max())})
    full_dates['date'] = full_dates['date'].dt.date

    # Merge COT data onto full dates, then forward fill open interest
    merged = pd.merge_asof(
        full_dates.sort_values('date'),
        cot_df_sorted.rename(columns={'report_date': 'date'}).sort_values('date'),
        on='date',
        direction='backward'  # forward-fill means take last COT report before or on date
    )

    # Merge open_interest_all into price_df by date
    price_df = price_df.merge(merged, on='date', how='left')

    # In case there are missing open interest values at the start, fill them forward
    price_df['open_interest_all'] = price_df['open_interest_all'].fillna(method='ffill')

    return price_df

def calculate_health_gauge(price_df):
    """
    Placeholder for your health gauge calculations using price, volume, rvol and open interest.
    This is where you implement your logic combining the metrics.
    """
    # For example, add a combined score column
    price_df['health_score'] = (
        (price_df['close'] / price_df['close'].rolling(LOOKBACK_DAYS).mean()) +
        (price_df['open_interest_all'] / price_df['open_interest_all'].rolling(LOOKBACK_DAYS).mean())
    ) / 2
    return price_df

def main():
    st.title("Gold & Silver Health Gauge with Extended Assets")

    # Define analysis date range
    start_date = st.date_input("Start Date", datetime.date(2024, 8, 9))
    end_date = st.date_input("End Date", datetime.date(2025, 8, 9))

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # To cover lookback windows, extend cot start date backward by LOOKBACK_DAYS
    cot_start_date = start_date - datetime.timedelta(days=LOOKBACK_DAYS)

    # Example only: analyzing Gold and Silver here; extend to your full list as needed
    for cot_name, yahoo_ticker in assets_cot_names.items():
        st.header(f"{cot_name} ({yahoo_ticker})")

        cot_df = fetch_cot_data(cot_name, cot_start_date, end_date)
        price_df = fetch_price_data(yahoo_ticker, start_date, end_date)

        if price_df.empty:
            st.warning(f"No price data available for {yahoo_ticker}. Skipping.")
            continue

        # Merge COT open interest onto daily price data
        merged_df = merge_cot_price(cot_df, price_df)

        # Calculate health gauge or other analytics here
        analyzed_df = calculate_health_gauge(merged_df)

        # Display sample data and charts
        st.dataframe(analyzed_df.tail(10))

        st.line_chart(analyzed_df.set_index('date')[['close', 'open_interest_all', 'health_score']])

        time.sleep(1)  # Be polite with API calls

if __name__ == "__main__":
    main()
