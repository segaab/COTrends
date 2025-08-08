import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from yahooquery import Ticker
from sodapy import Socrata
import os

st.set_page_config(page_title="Gold & Silver Health Gauge", layout="wide")

# Constants
COT_MARKETS = [
    "GOLD - COMMODITY EXCHANGE INC.",
    "SILVER - COMMODITY EXCHANGE INC."
]

WEIGHT_PV_RVOL = 0.40
WEIGHT_COT = 0.35
WEIGHT_OI = 0.25

COT_SHORT_TERM_WT = 0.40
COT_LONG_TERM_WT = 0.60

ARV_LOOKBACK_DAYS = 20
COT_LONG_TERM_WEEKS = 12

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
        results = client.get("6dca-aqww", where=where_clause, limit=5000, order="report_date_as_yyyy_mm_dd DESC")
        df = pd.DataFrame.from_records(results)
        if df.empty:
            return df
        df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        for col in ["noncommercial_positions_long", "noncommercial_positions_short", "commercial_positions_long", "commercial_positions_short"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df.sort_values("report_date_as_yyyy_mm_dd")
    except Exception as e:
        st.warning(f"Error fetching COT data for {market_name}: {e}")
        return pd.DataFrame()

# --- Process COT Net Positions ---
def compute_cot_net_positions(df):
    df = df.copy()
    df["noncommercial_net"] = df["noncommercial_positions_long"] - df["noncommercial_positions_short"]
    df["commercial_net"] = df["commercial_positions_long"] - df["commercial_positions_short"]
    return df

# --- Short Term COT Score ---
def cot_short_term_score(df):
    if df.empty or len(df) < 2:
        return 3  # Neutral fallback
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
    combined = (nc_score + c_score) / 2
    return combined

# --- Long Term COT Score ---
def cot_long_term_score(df):
    if df.empty or len(df) < COT_LONG_TERM_WEEKS:
        return 3  # Neutral fallback
    df = df.copy()
    df["noncommercial_net"] = df["noncommercial_positions_long"] - df["noncommercial_positions_short"]
    df["commercial_net"] = df["commercial_positions_long"] - df["commercial_positions_short"]
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
    combined = (nc_score + c_score) / 2
    return combined

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

# --- Price, Volume and Relative Volume Score ---
def price_volume_rvol_score(df):
    if df.empty or len(df) < ARV_LOOKBACK_DAYS + 3:
        return 3  # Neutral fallback
    df = df.sort_values("date").copy()
    df["PctChange"] = df["Close"].pct_change() * 100
    df["AvgVol20"] = df["Volume"].rolling(ARV_LOOKBACK_DAYS).mean()
    df = df.dropna(subset=["AvgVol20"])
    df["RVOL"] = df["Volume"] / df["AvgVol20"]
    recent_rvols = df["RVOL"].tail(ARV_LOOKBACK_DAYS)
    p30 = np.percentile(recent_rvols, 30)
    p70 = np.percentile(recent_rvols, 70)
    p90 = np.percentile(recent_rvols, 90)
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    two_days_ago = df.iloc[-3]
    price_chg_today = today["PctChange"]
    rvol_today = today["RVOL"]
    def is_trending_up():
        return (today["Close"] > yesterday["Close"] > two_days_ago["Close"]) and \
               (today["Volume"] > df["AvgVol20"].iloc[-1]) and \
               (yesterday["Volume"] > df["AvgVol20"].iloc[-1]) and \
               (two_days_ago["Volume"] > df["AvgVol20"].iloc[-1])
    def is_trending_down():
        return (today["Close"] < yesterday["Close"] < two_days_ago["Close"]) and \
               (today["Volume"] > df["AvgVol20"].iloc[-1]) and \
               (yesterday["Volume"] > df["AvgVol20"].iloc[-1]) and \
               (two_days_ago["Volume"] > df["AvgVol20"].iloc[-1])
    if price_chg_today >= 2 and rvol_today >= p90:
        return 5
    if 1 <= price_chg_today < 2 and p70 <= rvol_today < p90:
        return 4
    if price_chg_today <= -2 and rvol_today >= p90:
        return 0
    if -2 < price_chg_today <= -1 and p70 <= rvol_today < p90:
        return 1
    if is_trending_up():
        return 4
    if is_trending_down():
        return 1
    if abs(price_chg_today) < 1 and p30 <= rvol_today <= p70:
        return 3
    return 3

# --- Compute Overall Health ---
def compute_asset_health(yahoo_df, cot_df, asset_name):
    pv_score = price_volume_rvol_score(yahoo_df)
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

# --- MAIN ---

st.title("Gold & Silver Health Gauge")

max_end = datetime.utcnow().date()
start_date = st.sidebar.date_input("Start Date", max_end - timedelta(days=365), max_value=max_end)
end_date = st.sidebar.date_input("End Date", max_end, max_value=max_end)

if start_date >= end_date:
    st.error("Start Date must be before End Date.")
    st.stop()

if "rerun_counter" not in st.session_state:
    st.session_state.rerun_counter = 0

if st.button("Refresh All Data"):
    st.session_state.rerun_counter += 1
    st.experimental_rerun()

gold_yahoo = fetch_yahoo_data("GC=F", start_date, end_date)
silver_yahoo = fetch_yahoo_data("SI=F", start_date, end_date)

gold_cot_raw = fetch_cot_data(COT_MARKETS[0])
silver_cot_raw = fetch_cot_data(COT_MARKETS[1])

gold_cot = compute_cot_net_positions(gold_cot_raw)
silver_cot = compute_cot_net_positions(silver_cot_raw)

gold_health, gold_pv, gold_cot_score, gold_oi = compute_asset_health(gold_yahoo, gold_cot, "Gold")
silver_health, silver_pv, silver_cot_score, silver_oi = compute_asset_health(silver_yahoo, silver_cot, "Silver")

st.header("Health Gauge Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gold")
    st.metric("Overall Health Score", gold_health, delta=None)
    st.markdown(f"**Color:** {health_color(gold_health)}")
    st.write(f"Price/Volume/RVol Score: {gold_pv:.2f}")
    st.write(f"COT Score: {gold_cot_score:.2f}")
    st.write(f"Open Interest Score: {gold_oi:.2f}")

with col2:
    st.subheader("Silver")
    st.metric("Overall Health Score", silver_health, delta=None)
    st.markdown(f"**Color:** {health_color(silver_health)}")
    st.write(f"Price/Volume/RVol Score: {silver_pv:.2f}")
    st.write(f"COT Score: {silver_cot_score:.2f}")
    st.write(f"Open Interest Score: {silver_oi:.2f}")

st.markdown("---")

st.info(
    """
    **Legend:**  
    0: 🔴 Red (Strong Bearish)  
    1: 🟠 Orange-Red (Moderate Bearish)  
    2: 🟠 Orange (Slight Bearish)  
    3: 🟡 Yellow (Neutral)  
    4: 🟢 Light Green (Moderate Bullish)  
    5: 🟢 Green (Strong Bullish)  
    """
)

# End of script
