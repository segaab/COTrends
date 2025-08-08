# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from yahooquery import Ticker
from sodapy import Socrata
import os

st.set_page_config(page_title="Gold & Silver Health Gauge", layout="wide")

# --- PARAMETERS & CONSTANTS ---
COT_MARKETS = [
    "GOLD - COMMODITY EXCHANGE INC.",
    "SILVER - COMMODITY EXCHANGE INC."
]

# Weights for final health score
WEIGHT_PV_RVOL = 0.40
WEIGHT_COT = 0.35
WEIGHT_OI = 0.25

# Within COT weight, short/long term weights
COT_SHORT_TERM_WT = 0.40
COT_LONG_TERM_WT = 0.60

# Lookback windows
ARV_LOOKBACK_DAYS = 20
COT_LONG_TERM_WEEKS = 12

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_yahoo_data(symbol, start, end):
    t = Ticker(symbol)
    df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    if df.empty:
        return pd.DataFrame()
    
    # Flatten MultiIndex if present
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Ensure 'date' column exists or rename if possible
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)
        else:
            df = df.reset_index()
            if "date" not in df.columns:
                raise ValueError("No date column found in Yahoo data")
    
    # Normalize column names (case insensitive)
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

    # Fill missing columns with zeros if absent
    if "OpenInterest" not in df.columns:
        df["OpenInterest"] = 0
    if "Volume" not in df.columns:
        df["Volume"] = 0

    # Confirm required columns exist
    for col in ["date", "Close", "Volume", "OpenInterest"]:
        if col not in df.columns:
            if col == "date":
                raise ValueError("Date column missing in Yahoo data")
            df[col] = 0
    
    df = df[["date", "Close", "Volume", "OpenInterest"]]
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data(ttl=3600)
def fetch_cot_data(market_name):
    # Load CFTC credentials from env vars
    app_token = "WSCaavlIcDgtLVZbJA1FKkq40"
    username = os.getenv("CFTC_USERNAME")
    password = os.getenv("CFTC_PASSWORD")

    if not (app_token and username and password):
        st.warning("Missing CFTC Socrata credentials in environment variables. COT data will not load.")
        return pd.DataFrame()

    client = Socrata(
        "publicreporting.cftc.gov",
        app_token,
        username=username,
        password=password,
        timeout=30
    )

    # Query data filtered by market name, last 1 year approx
    today = datetime.utcnow()
    one_year_ago = today - timedelta(days=365)

    where_clause = f"market_and_exchange_names = '{market_name}' AND report_date_as_yyyy_mm_dd >= '{one_year_ago.strftime('%Y-%m-%d')}'"
    try:
        results = client.get("6dca-aqww", where=where_clause, limit=5000, order="report_date_as_yyyy_mm_dd DESC")
        df = pd.DataFrame.from_records(results)
        if df.empty:
            return df
        # Convert datatypes
        df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        # Convert relevant numeric columns to int
        for col in ["noncommercial_positions_long", "noncommercial_positions_short", "commercial_positions_long", "commercial_positions_short"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df.sort_values("report_date_as_yyyy_mm_dd")
    except Exception as e:
        st.warning(f"Error fetching COT data for {market_name}: {e}")
        return pd.DataFrame()

def compute_cot_net_positions(df):
    # Net positions: longs - shorts for commercials and non-commercials
    df = df.copy()
    df["noncommercial_net"] = df["noncommercial_positions_long"] - df["noncommercial_positions_short"]
    df["commercial_net"] = df["commercial_positions_long"] - df["commercial_positions_short"]
    return df

def cot_short_term_score(df):
    # Use latest two reports to find 1-week net position change
    if df.empty or len(df) < 2:
        return 3  # Neutral score fallback

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Net position changes
    nc_change = latest["noncommercial_net"] - prev["noncommercial_net"]  # bullish if positive
    c_change = latest["commercial_net"] - prev["commercial_net"]  # bearish if positive

    # Score logic: scale changes to score 0-5
    # We'll define thresholds for "large increase", "minimal", "large decrease"
    # Thresholds are arbitrary and can be tuned

    def scale_net_change(val, bullish=True):
        # val positive bullish means more bullish for NC, more bearish for C
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
            # For commercial (bearish), positive change means more bearish
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

    # Non-commercial sets bullish weight, Commercial sets bearish weight, so combine carefully
    # Let's average their scores (weighted 50/50 internally) then normalize 0-5

    combined = (nc_score + c_score) / 2
    return combined

def cot_long_term_score(df):
    if df.empty or len(df) < COT_LONG_TERM_WEEKS:
        return 3  # Neutral fallback

    # Calculate rolling 12-week average net positions
    df = df.copy()
    df["noncommercial_net"] = df["noncommercial_positions_long"] - df["noncommercial_positions_short"]
    df["commercial_net"] = df["commercial_positions_long"] - df["commercial_positions_short"]

    df["nc_rolling_avg"] = df["noncommercial_net"].rolling(window= COT_LONG_TERM_WEEKS).mean()
    df["c_rolling_avg"] = df["commercial_net"].rolling(window= COT_LONG_TERM_WEEKS).mean()

    latest = df.iloc[-1]

    # To score long-term trend, compare latest rolling avg to rolling avg 12 weeks ago
    past = df.iloc[-COT_LONG_TERM_WEEKS]

    def score_long_term_change(current, past, bullish=True):
        diff = current - past
        # Using similar thresholds as short term but adjusted for longer period
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

def open_interest_score(df):
    # Calculate 1-week % change in open interest from last available day and 7 days ago
    if df.empty or len(df) < 7:
        return 3
    df = df.sort_values("date")
    latest_oi = df.iloc[-1]["OpenInterest"]
    prev_oi = df.iloc[-7]["OpenInterest"]
    if prev_oi == 0:
        return 3
    pct_change = (latest_oi - prev_oi) / prev_oi

    # Score scale: 
    # Large increase in OI = bullish (score 5)
    # Large decrease in OI = bearish (score 0)
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

def price_volume_rvol_score(df):
    # Use last 20 days volume for ARV, calculate percentiles for volume
    if df.empty or len(df) < ARV_LOOKBACK_DAYS + 3:
        return 3  # Neutral fallback

    df = df.sort_values("date").copy()
    df["PctChange"] = df["Close"].pct_change() * 100  # daily % change

    # Calculate 20-day average volume
    df["AvgVol20"] = df["Volume"].rolling(ARV_LOOKBACK_DAYS).mean()
    df = df.dropna(subset=["AvgVol20"])

    # Calculate Relative Volume (RVOL)
    df["RVOL"] = df["Volume"] / df["AvgVol20"]

    # Calculate percentiles of RVOL over trailing 20 days
    recent_rvols = df["RVOL"].tail(ARV_LOOKBACK_DAYS)
    p30 = np.percentile(recent_rvols, 30)
    p70 = np.percentile(recent_rvols, 70)
    p90 = np.percentile(recent_rvols, 90)

    # Analyze last day for phase
    today = df.iloc[-1]
    yesterday = df.iloc[-2]
    two_days_ago = df.iloc[-3]
    three_days_ago = df.iloc[-4]

    price_chg_today = today["PctChange"]
    vol_today = today["Volume"]
    rvol_today = today["RVOL"]

    # Helper to check trending for last 3 days
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

    # Apply your spectrum rules:
    # Strong Accumulation (5)
    if price_chg_today >= 2 and rvol_today >= p90:
        return 5

    # Moderate Accumulation (4)
    if 1 <= price_chg_today < 2 and p70 <= rvol_today < p90:
        return 4

    # Strong Distribution (0)
    if price_chg_today <= -2 and rvol_today >= p90:
        return 0

    # Moderate Distribution (1)
    if -2 < price_chg_today <= -1 and p70 <= rvol_today < p90:
        return 1

    # Trending Upwards (4)
    if is_trending_up():
        return 4

    # Trending Downwards (1)
    if is_trending_down():
        return 1

    # Neutral/Consolidation (3)
    if abs(price_chg_today) < 1 and p30 <= rvol_today <= p70:
        return 3

    # Fallback neutral
    return 3

# --- Main UI ---

st.title("Gold & Silver Health Gauge")

# Date selection for price data (Yahooquery only)
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

# Fetch data for Gold and Silver

gold_yahoo = fetch_yahoo_data("GC=F", start_date, end_date)
silver_yahoo = fetch_yahoo_data("SI=F", start_date, end_date)

gold_cot_raw = fetch_cot_data(COT_MARKETS[0])
silver_cot_raw = fetch_cot_data(COT_MARKETS[1])

gold_cot = compute_cot_net_positions(gold_cot_raw)
silver_cot = compute_cot_net_positions(silver_cot_raw)

# Compute scores

def compute_asset_health(yahoo_df, cot_df, asset_name):
    pv_score = price_volume_rvol_score(yahoo_df)
    cot_st = cot_short_term_score(cot_df)
    cot_lt = cot_long_term_score(cot_df)
    cot_score = cot_st * COT_SHORT_TERM_WT + cot_lt * COT_LONG_TERM_WT
    oi_score = open_interest_score(yahoo_df)

    # Weighted health score
    health = pv_score * WEIGHT_PV_RVOL + cot_score * WEIGHT_COT + oi_score * WEIGHT_OI
    health_rounded = round(health, 2)

    return {
        "PriceVolRVOL": pv_score,
        "COT Short Term": round(cot_st, 2),
        "COT Long Term": round(cot_lt, 2),
        "COT Combined": round(cot_score, 2),
        "Open Interest": oi_score,
        "Health Score": health_rounded
    }

gold_scores = compute_asset_health(gold_yahoo, gold_cot, "Gold")
silver_scores = compute_asset_health(silver_yahoo, silver_cot, "Silver")

def color_code(score):
    # 0: Red, 1: Orange-Red, 2: Orange, 3: Yellow, 4: Light Green, 5: Green
    if score <= 0.5:
        return "🔴"
    elif score <= 1.5:
        return "🟠"
    elif score <= 2.5:
        return "🟠"
    elif score <= 3.5:
        return "🟡"
    elif score <= 4.5:
        return "🟢"
    else:
        return "🟢"

st.markdown("### Health Gauge Scores & Breakdown")
cols = st.columns(2)

with cols[0]:
    st.subheader("Gold")
    for k, v in gold_scores.items():
        if k == "Health Score":
            st.markdown(f"**{k}: {v} {color_code(v)}**")
        else:
            st.write(f"{k}: {v}")

with cols[1]:
    st.subheader("Silver")
    for k, v in silver_scores.items():
        if k == "Health Score":
            st.markdown(f"**{k}: {v} {color_code(v)}**")
        else:
            st.write(f"{k}: {v}")

st.markdown("---")
st.markdown("### Latest Price & Volume Data (last 5 days)")

st.write("Gold (GC=F)")
st.dataframe(gold_yahoo.tail(5))

st.write("Silver (SI=F)")
st.dataframe(silver_yahoo.tail(5))

st.markdown("---")
st.markdown("### Latest COT Data Snapshot")

def latest_cot_table(cot_df, asset):
    if cot_df.empty:
        return pd.DataFrame()
    latest = cot_df.iloc[-5:][
        [
            "report_date_as_yyyy_mm_dd",
            "noncommercial_positions_long",
            "noncommercial_positions_short",
            "commercial_positions_long",
            "commercial_positions_short",
        ]
    ].copy()
    latest.rename(columns={"report_date_as_yyyy_mm_dd": "Date"}, inplace=True)
    latest["Date"] = latest["Date"].dt.strftime("%Y-%m-%d")
    return latest

st.write("Gold COT")
st.dataframe(latest_cot_table(gold_cot, "Gold"))

st.write("Silver COT")
st.dataframe(latest_cot_table(silver_cot, "Silver"))

# Combine data for export
combined_data = pd.concat([
    gold_yahoo.assign(Asset="Gold"),
    silver_yahoo.assign(Asset="Silver")
])
csv_bytes = combined_data.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Price & Volume Data CSV",
    csv_bytes,
    "gold_silver_data.csv",
    "text/csv",
)

st.markdown(
    """
    ---
    Data Sources:
    - Price, Volume, Open Interest: Yahoo Finance (via yahooquery)
    - COT Data: CFTC Socrata API
    """
)
