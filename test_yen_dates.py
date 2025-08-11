import streamlit as st
import pandas as pd
import numpy as np
import datetime
import logging
import time
from sodapy import Socrata
from yahooquery import Ticker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded API Key
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"

# Initialize Socrata client once (no auth needed for public data)
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# Asset definitions (COT market names mapped to CME futures tickers)
assets = {
    # Metals and Commodities (COMEX/NYMEX)
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "PL=F",
    "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": "PA=F",
    "WTI CRUDE OIL FINANCIAL - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "COPPER - COMMODITY EXCHANGE INC.": "HG=F",

    # Currency Futures (CME)
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6A=F",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "6B=F",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6C=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J=F",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S=F",

    # Crypto Futures (CME)
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC=F",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE": "MBT=F",
    "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE": "MET=F",

    # Equity Index Futures
    "E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "DJR"
}

def fetch_cot_data(market_name, max_attempts=3):
    logger.info(f"Fetching COT data for {market_name}")
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get("6dca-aqww", where=where_clause,
                                 order="report_date_as_yyyy_mm_dd DESC", limit=1000)
            if results:
                df = pd.DataFrame.from_records(results)
                df['report_date'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
                df['open_interest_all'] = pd.to_numeric(df['open_interest_all'], errors='coerce')
                logger.info(f"Fetched {len(df)} rows for {market_name}")
                return df
            else:
                logger.warning(f"No data returned for {market_name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching COT data for {market_name}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed to fetch COT data for {market_name} after {max_attempts} attempts.")
    return pd.DataFrame()

def fetch_yahooquery_data(ticker, start_date, end_date, max_attempts=3):
    logger.info(f"Fetching Yahoo data for {ticker} from {start_date} to {end_date}")
    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval='1d')
            if hist.empty:
                logger.warning(f"No data returned for {ticker} from yahooquery.")
                return pd.DataFrame()
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.loc[ticker]
            hist = hist.reset_index()
            hist.rename(columns={"date": "date"}, inplace=True)
            logger.info(f"Fetched {len(hist)} rows for {ticker}")
            return hist
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed to fetch Yahoo data for {ticker} after {max_attempts} attempts.")
    return pd.DataFrame()

def merge_cot_price(cot_df, price_df):
    if cot_df.empty or price_df.empty:
        logger.warning("One or both DataFrames are empty; skipping merge.")
        return price_df
    cot_merge = cot_df[['report_date', 'open_interest_all']].drop_duplicates().copy()
    cot_merge.rename(columns={'report_date': 'date'}, inplace=True)
    cot_merge['date'] = pd.to_datetime(cot_merge['date'])
    price_df['date'] = pd.to_datetime(price_df['date'])
    full_dates = pd.DataFrame({'date': pd.date_range(price_df['date'].min(), price_df['date'].max())})
    merged_dates = pd.merge_asof(full_dates, cot_merge, on='date', direction='backward')
    merged_df = pd.merge(price_df, merged_dates[['date', 'open_interest_all']], on='date', how='left')
    merged_df['open_interest_all'] = merged_df['open_interest_all'].fillna(method='ffill')
    return merged_df

def calculate_rvol(df, window=20):
    if 'volume' not in df.columns and 'Volume' not in df.columns:
        logger.warning("Volume data not present; cannot compute RVol.")
        df['rvol'] = np.nan
        return df
    vol_col = 'volume' if 'volume' in df.columns else 'Volume'
    df['rvol'] = df[vol_col] / df[vol_col].rolling(window).mean()
    return df

def calculate_health_gauge(df):
    if df.empty:
        return np.nan
    oi_norm = (df['open_interest_all'] - df['open_interest_all'].min()) / (df['open_interest_all'].max() - df['open_interest_all'].min() + 1e-9)
    price_change = df['close'].pct_change().fillna(0) if 'close' in df.columns else df['Close'].pct_change().fillna(0)
    price_norm = (price_change - price_change.min()) / (price_change.max() - price_change.min() + 1e-9)
    rvol_norm = (df['rvol'] - df['rvol'].min()) / (df['rvol'].max() - df['rvol'].min() + 1e-9)
    health_score = 0.4 * oi_norm + 0.4 * price_norm + 0.2 * rvol_norm
    return health_score * 100

def main():
    st.title("COT Futures Health Gauge")
    start_date = st.date_input("Start Date", datetime.date(2024, 8, 9))
    end_date = st.date_input("End Date", datetime.date(2025, 8, 9))
    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
        return
    for cot_name, ticker in assets.items():
        st.subheader(f"{cot_name} ({ticker})")
        cot_df = fetch_cot_data(cot_name)
        if cot_df.empty:
            st.warning(f"No COT data for {cot_name}")
            continue
        cot_start = cot_df['report_date'].min().date()
        cot_end = cot_df['report_date'].max().date()
        adj_start = max(start_date, cot_start)
        adj_end = min(end_date, cot_end + datetime.timedelta(days=7))
        price_df = fetch_yahooquery_data(ticker, adj_start.isoformat(), adj_end.isoformat())
        if price_df.empty:
            st.warning(f"No price data for {ticker}")
            continue
        price_df = calculate_rvol(price_df)
        merged_df = merge_cot_price(cot_df, price_df)
        merged_df['health_score'] = calculate_health_gauge(merged_df)
        st.write(merged_df.tail(5))
        latest_health = merged_df['health_score'].iloc[-1]
        st.metric("Latest Health Gauge Score", f"{latest_health:.2f}")
        st.markdown("---")

if __name__ == "__main__":
    main()
