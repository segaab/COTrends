import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from yahooquery import Ticker
from sodapy import Socrata
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Gold & Silver Health Gauge", layout="wide")

# Constants
COT_MARKETS = [
    "GOLD - COMMODITY EXCHANGE INC.",
    "SILVER - COMMODITY EXCHANGE INC."
]

YAHOO_SYMBOLS = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F"
}

WEIGHT_PV_RVOL = 0.40
WEIGHT_COT = 0.35
WEIGHT_OI = 0.25

COT_SHORT_TERM_WT = 0.40
COT_LONG_TERM_WT = 0.60

COT_LONG_TERM_WEEKS = 12
LOOKBACK_DAYS_SPECTRUM = 30  # ~6 weeks approx

# --- Fetch Yahoo Data ---
@st.cache_data(ttl=3600)
def fetch_yahoo_data(symbol, start, end):
    t = Ticker(symbol)
    df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)
        else:
            df = df.reset_index()
            if "date" not in df.columns:
                raise ValueError("No date column found in Yahoo data")
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "close":
            rename_map[col] = "Close"
        elif col_lower == "volume":
            rename_map[col] = "Volume"
        elif col_lower == "openinterest":
            rename_map[col] = "OpenInterest"
    df = df.rename(columns=rename_map)
    if "OpenInterest" not in df.columns:
        df["OpenInterest"] = 0
    if "Volume" not in df.columns:
        df["Volume"] = 0
    for col in ["date", "Close", "Volume", "OpenInterest"]:
        if col not in df.columns:
            if col == "date":
                raise ValueError("Date column missing in Yahoo data")
            df[col] = 0
    df = df[["date", "Close", "Volume", "OpenInterest"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# --- Fetch COT Data ---
@st.cache_data(ttl=3600)
def fetch_cot_data(market_name):
    app_token = "WSCaavlIcDgtLVZbJA1FKkq40"
    if not app_token:
        st.warning("Missing CFTC Socrata API token in environment variables. COT data will not load.")
        return pd.DataFrame()
    client = Socrata("publicreporting.cftc.gov", app_token, timeout=30)
    today = datetime.utcnow()
    one_year_ago = today - timedelta(days=365)
    where_clause = f"market_and_exchange_names = '{market_name}' AND report_date_as_yyyy_mm_dd >= '{one_year_ago.strftime('%Y-%m-%d')}'"
    try:
        results = client.get("6dca-aqww", where=where_clause, limit=5000, order="report_date_as_yyyy_mm_dd ASC")
        df = pd.DataFrame.from_records(results)
        if df.empty:
            return df
        df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        cols_to_int = [
            "noncomm_positions_long_all",
            "noncomm_positions_short_all",
            "noncomm_postions_spread_all",
            "comm_positions_long_all",
            "comm_positions_short_all",
            "tot_rept_positions_long_all",
            "tot_rept_positions_short",
            "nonrept_positions_long_all",
            "nonrept_positions_short_all"
        ]
        for col in cols_to_int:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            else:
                df[col] = 0
        return df.sort_values("report_date_as_yyyy_mm_dd").reset_index(drop=True)
    except Exception as e:
        st.warning(f"Error fetching COT data for {market_name}: {e}")
        return pd.DataFrame()

# --- Process COT Net Positions ---
def compute_cot_net_positions(df):
    df = df.copy()
    df["noncommercial_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    df["commercial_net"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
    return df

# --- Short Term COT Score ---
def cot_short_term_score(df):
    if df.empty or len(df) < 2:
        return 3
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    nc_change = latest["noncommercial_net"] - prev["noncommercial_net"]  # bullish if positive
    c_change = latest["commercial_net"] - prev["commercial_net"]        # bearish if positive

    def scale_net_change(val, bullish=True):
        if bullish:
            if val > 2000:
                return 5
            elif val > 500:
                return 4
            elif val > -500:
                return 3
            elif val > -2000:
                return 2
            else:
                return 1
        else:
            if val > 2000:
                return 0
            elif val > 500:
                return 1
            elif val > -500:
                return 3
            elif val > -2000:
                return 4
            else:
                return 5
    nc_score = scale_net_change(nc_change, bullish=True)
    c_score = scale_net_change(c_change, bullish=False)
    return (nc_score + c_score) / 2

# --- Long Term COT Score ---
def cot_long_term_score(df):
    if df.empty or len(df) < COT_LONG_TERM_WEEKS:
        return 3
    df = df.copy()
    df["noncommercial_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]
    df["commercial_net"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
    df["nc_rolling_avg"] = df["noncommercial_net"].rolling(window=COT_LONG_TERM_WEEKS).mean()
    df["c_rolling_avg"] = df["commercial_net"].rolling(window=COT_LONG_TERM_WEEKS).mean()
    latest = df.iloc[-1]
    past = df.iloc[-COT_LONG_TERM_WEEKS]
    def score_long_term_change(current, past, bullish=True):
        diff = current - past
        if bullish:
            if diff > 1000:
                return 5
            elif diff > 250:
                return 4
            elif diff > -250:
                return 3
            elif diff > -1000:
                return 2
            else:
                return 1
        else:
            if diff > 1000:
                return 0
            elif diff > 250:
                return 1
            elif diff > -250:
                return 3
            elif diff > -1000:
                return 4
            else:
                return 5
    nc_score = score_long_term_change(latest["nc_rolling_avg"], past["nc_rolling_avg"], bullish=True)
    c_score = score_long_term_change(latest["c_rolling_avg"], past["c_rolling_avg"], bullish=False)
    return (nc_score + c_score) / 2

# --- Open Interest Score ---
def open_interest_score(df):
    if df.empty or len(df) < 7:
        return 3
    df = df.sort_values("date")
    latest_oi = df.iloc[-1]["OpenInterest"]
    prev_oi = df.iloc[-7]["OpenInterest"]
    if prev_oi == 0:
        return 3
    pct_change = (latest_oi - prev_oi) / prev_oi
    if pct_change > 0.1:
        return 5
    elif pct_change > 0.05:
        return 4
    elif pct_change > -0.05:
        return 3
    elif pct_change > -0.1:
        return 2
    else:
        return 1

# --- Calculate Relative Volume (RVOL) ---
def calculate_rvol(df):
    df = df.copy()
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    df["RVOL"] = df["Volume"] / df["Volume_MA20"]
    df = df.dropna(subset=["RVOL"])
    return df

# --- Spectrum Score Calculation with scikit-learn trend detection ---
def calculate_spectrum_score(df):
    df = df.sort_values("date").copy()
    # Use last 30 trading days approx
    df = df.tail(LOOKBACK_DAYS_SPECTRUM)
    if len(df) < LOOKBACK_DAYS_SPECTRUM:
        return 3  # neutral if insufficient data

    df = calculate_rvol(df)
    if df.empty:
        return 3

    # Calculate 78th percentile RVOL threshold
    rvol_78th = df["RVOL"].quantile(0.78)

    total_weight = 0
    count = 0

    # Accumulation and Distribution return scenarios weights
    def accumulation_weight(ret_pct):
        if 0 <= ret_pct < 1:
            return 1
        elif 1 <= ret_pct < 2:
            return 2
        elif ret_pct >= 2:
            return 3
        return 0

    def distribution_weight(ret_pct):
        if -1 <= ret_pct < 0:
            return -1
        elif -2 <= ret_pct < -1:
            return -2
        elif ret_pct < -2:
            return -3
        return 0

    # Calculate trending score via linear regression on price
    # Use scikit-learn LinearRegression to detect trend on last 6 weeks price
    if len(df) < 10:
        trend_score = 3
    else:
        prices = df["Close"].values.reshape(-1, 1)
        days = np.arange(len(prices)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(days, prices)
        slope = model.coef_[0][0]

        # Map slope to score (example thresholds, tweak as needed)
        if slope > 0.1:
            trend_score = 5  # Strong uptrend
        elif 0.05 < slope <= 0.1:
            trend_score = 4  # Moderate uptrend
        elif -0.05 <= slope <= 0.05:
            trend_score = 3  # Neutral
        elif -0.1 <= slope < -0.05:
            trend_score = 2  # Moderate downtrend
        else:
            trend_score = 1  # Strong downtrend

    # For each day with RVOL >= 78th percentile, apply accumulation/distribution weights
    for idx in range(1, len(df)):
        row = df.iloc[idx]
        if row["RVOL"] >= rvol_78th:
            prev_close = df.iloc[idx - 1]["Close"]
            if prev_close == 0:
                continue
            ret_pct = ((row["Close"] - prev_close) / prev_close) * 100
            if ret_pct >= 0:
                weight = accumulation_weight(ret_pct)
            else:
                weight = distribution_weight(ret_pct)

            total_weight += weight
            count += 1

    if count == 0:
        # No significant RVOL spikes found, treat as neutral (trend only)
        base_score = 3
    else:
        avg_weight = total_weight / count
        # Normalize from [-3,3] to [0,5]
        base_score = (avg_weight + 3) / 6 * 5
        base_score = max(0, min(5, base_score))

    # Combine base score and trend score (weight 70%-30%)
    combined_score = 0.7 * base_score + 0.3 * trend_score

    return combined_score

# --- Compute Overall Health ---
def compute_asset_health(yahoo_df, cot_df, asset_name):
    pv_score = calculate_spectrum_score(yahoo_df)
    cot_df = compute_cot_net_positions(cot_df)
    cot_st = cot_short_term_score(cot_df)
    cot_lt = cot_long_term_score(cot_df)
    cot_score = cot_st * COT_SHORT_TERM_WT + cot_lt * COT_LONG_TERM_WT
    oi_score = open_interest_score(yahoo_df)
    health_raw = pv_score * WEIGHT_PV_RVOL + cot_score * WEIGHT_COT + oi_score * WEIGHT_OI
    health = int(round(health_raw))
    return health, pv_score, cot_score, oi_score

def health_color(score):
    colors = {
        0: "🔴 Red",
        1: "🟠 Orange-Red",
        2: "🟠 Orange",
        3: "🟡 Yellow",
        4: "🟢 Light Green",
        5: "🟢 Green"
    }
    return colors.get(score, "⚪️ Unknown")

# --- UI ---

st.title("Gold & Silver Health Gauge")

max_end = datetime.utcnow().date()
start_date = st.sidebar.date_input("Start Date", max_end - timedelta(days=365), max_value=max_end)
end_date = st.sidebar.date_input("End Date", max_end, max_value=max_end)

if start_date >= end_date:
    st.error("Start Date must be before End Date.")
    st.stop()

if st.button("Reload Data"):
    st.cache_data.clear()

cols = st.columns(len(COT_MARKETS))
for i, market in enumerate(COT_MARKETS):
    with cols[i]:
        st.header(market)
        yahoo_symbol = YAHOO_SYMBOLS.get(market)
        if not yahoo_symbol:
            st.error(f"Yahoo symbol for {market} not found.")
            continue

        yahoo_df = fetch_yahoo_data(yahoo_symbol, start_date, end_date)
        cot_df = fetch_cot_data(market)

        if yahoo_df.empty:
            st.warning("No Yahoo data available.")
            continue
        if cot_df.empty:
            st.warning("No COT data available.")
            continue

        health, pv_score, cot_score, oi_score = compute_asset_health(yahoo_df, cot_df, market)

        st.markdown(f"**Health Gauge Score:** {health} / 5  {health_color(health)}")
        st.write(f"- Price + Volume Spectrum Score: {pv_score:.2f}")
        st.write(f"- COT Score (Short+Long Term): {cot_score:.2f}")
        st.write(f"- Open Interest Score: {oi_score:.2f}")
