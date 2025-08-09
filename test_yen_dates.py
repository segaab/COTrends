import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sodapy import Socrata
from yahooquery import Ticker
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import requests

# ==============================
# CONFIGURATION & CONSTANTS
# ==============================
TIMEOUT = 60  # seconds
UPTREND_THRESHOLD = 0.04  # +4%
DOWNTREND_THRESHOLD = -0.04  # -4%
LOOKBACK_DAYS_SPECTRUM = 30  # ~6 weeks trading days

COT_MARKETS = [
    "GOLD - COMMODITY EXCHANGE INC.",
    "SILVER - COMMODITY EXCHANGE INC.",
]

YAHOO_SYMBOLS = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
}

# Weights for health calculation
WEIGHT_PV_RVOL = 0.40
WEIGHT_COT = 0.35
WEIGHT_OI = 0.25

# ==============================
# SOCrata Client Setup
# ==============================
APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"  # Replace with your app token or set env variable

@st.cache_resource(show_spinner=False)
def get_socrata_client():
    try:
        client = Socrata("publicreporting.cftc.gov", APP_TOKEN, timeout=TIMEOUT)
        st.info("Socrata client initialized.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Socrata client: {e}")
        return None

# ==============================
# DATA FETCHING
# ==============================
@st.cache_data(ttl=60*60*6, show_spinner=True)
def fetch_cot_data(market_name):
    client = get_socrata_client()
    if client is None:
        st.error("Cannot fetch COT data because Socrata client is unavailable.")
        return pd.DataFrame()

    where_clause = f"market_and_exchange_name='{market_name}'"
    try:
        results = client.get("6dca-aqww", where=where_clause, order="report_date DESC", limit=1000)
        if not results:
            st.warning(f"No COT data returned for {market_name}.")
            return pd.DataFrame()
        df = pd.DataFrame.from_records(results)
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        if df["report_date"].isnull().all():
            st.warning(f"All report dates are invalid in COT data for {market_name}.")
        numeric_cols = [
            "noncomm_positions_long_all", "noncomm_positions_short_all", "noncomm_positions_spread_all",
            "comm_positions_long_all", "comm_positions_short_all",
            "tot_rept_positions_long_all", "tot_rept_positions_short",
            "nonrept_positions_long_all", "nonrept_positions_short_all",
            "open_interest_all"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        st.success(f"COT data fetched for {market_name}: {len(df)} rows.")
        return df.sort_values("report_date")
    except requests.exceptions.Timeout:
        st.error(f"Timeout while fetching COT data for {market_name}.")
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching COT data for {market_name}: {e}")
    except Exception as e:
        st.error(f"Unexpected error fetching COT data for {market_name}: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=60*60*2, show_spinner=True)
def fetch_yahoo_data(symbol, start_date, end_date):
    st.info(f"Fetching Yahoo data for {symbol} from {start_date} to {end_date}...")
    try:
        ticker = Ticker(symbol, asynchronous=False, timeout=TIMEOUT)
        data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No Yahoo data found for {symbol}.")
            return pd.DataFrame()
        data = data.reset_index()
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
        data = data.dropna(subset=["date", "close", "volume"])
        data.columns = [c.lower() for c in data.columns]
        st.success(f"Yahoo data fetched for {symbol}: {len(data)} rows.")
        return data
    except requests.exceptions.Timeout:
        st.error(f"Timeout while fetching Yahoo data for {symbol}.")
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching Yahoo data for {symbol}: {e}")
    except Exception as e:
        st.error(f"Unexpected error fetching Yahoo data for {symbol}: {e}")
    return pd.DataFrame()

# ==============================
# DATA PROCESSING
# ==============================
def calculate_rvol(df):
    df = df.sort_values("date")
    df["avgvol20"] = df["volume"].rolling(window=20).mean()
    df["rvol"] = df["volume"] / df["avgvol20"]
    df.dropna(subset=["rvol"], inplace=True)
    return df

def compute_cot_net_positions(cot_df):
    cot_df = cot_df.sort_values("report_date")
    cot_df["net_comm"] = cot_df["comm_positions_long_all"] - cot_df["comm_positions_short_all"]
    cot_df["net_noncomm"] = cot_df["noncomm_positions_long_all"] - cot_df["noncomm_positions_short_all"]
    return cot_df

def cot_short_term_score(cot_df):
    if cot_df.shape[0] < 2:
        return 3
    latest = cot_df.iloc[-1]
    prev = cot_df.iloc[-2]
    delta_comm = latest["net_comm"] - prev["net_comm"]
    delta_noncomm = latest["net_noncomm"] - prev["net_noncomm"]
    comm_score = 5 if delta_comm > 0 else 0 if delta_comm < 0 else 3
    noncomm_score = 5 if delta_noncomm > 0 else 0 if delta_noncomm < 0 else 3
    return (comm_score + noncomm_score) / 2

def cot_long_term_score(cot_df):
    if cot_df.shape[0] < 12:
        return 3
    df_12 = cot_df.tail(12)
    net_comm = df_12["net_comm"].values
    net_noncomm = df_12["net_noncomm"].values
    comm_trend = np.polyfit(range(12), net_comm, 1)[0]
    noncomm_trend = np.polyfit(range(12), net_noncomm, 1)[0]
    comm_score = 5 if comm_trend < 0 else 0 if comm_trend > 0 else 3
    noncomm_score = 5 if noncomm_trend > 0 else 0 if noncomm_trend < 0 else 3
    return (comm_score + noncomm_score) / 2

def open_interest_score(yahoo_df):
    if yahoo_df.empty or "openinterest" not in yahoo_df.columns or yahoo_df["openinterest"].isna().all():
        return 3
    oi = yahoo_df["openinterest"].dropna()
    if len(oi) < 20:
        return 3
    slope = np.polyfit(range(len(oi)), oi, 1)[0]
    return 5 if slope > 0 else 0 if slope < 0 else 3

# ==============================
# SCENARIO & SPECTRUM SCORING
# ==============================
def classify_scenario(ret):
    if ret > UPTREND_THRESHOLD:
        return "Uptrend"
    elif ret < DOWNTREND_THRESHOLD:
        return "Downtrend"
    else:
        return "Accumulation/Distribution"

def calculate_spectrum_score(df):
    df = df.sort_values("date").tail(LOOKBACK_DAYS_SPECTRUM).copy()
    df = calculate_rvol(df)
    if df.empty:
        return 3, pd.DataFrame()
    rvol_78th = df["rvol"].quantile(0.78)

    total_weight = 0
    count = 0
    points = []

    def acc_weight(r):
        if 0 <= r < 0.01:
            return 1
        elif 0.01 <= r < 0.02:
            return 2
        elif r >= 0.02:
            return 3
        return 0

    def dist_weight(r):
        if -0.01 <= r < 0:
            return -1
        elif -0.02 <= r < -0.01:
            return -2
        elif r < -0.02:
            return -3
        return 0

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        if row["rvol"] >= rvol_78th:
            prev_close = df.iloc[i - 1]["close"]
            if prev_close == 0:
                continue
            ret_pct = (row["close"] - prev_close) / prev_close
            if ret_pct >= 0:
                weight = acc_weight(ret_pct)
                scenario = "Accumulation"
            else:
                weight = dist_weight(ret_pct)
                scenario = "Distribution"
            total_weight += weight
            count += 1

            next_close = df.iloc[i + 1]["close"]
            next_ret = (next_close - row["close"]) / row["close"]
            points.append({
                "Date": row["date"],
                "Scenario": scenario,
                "Return": ret_pct,
                "Weight": weight,
                "Next Day Return": next_ret
            })

    avg_weight = total_weight / count if count else 0
    base_score = max(0, min(5, (avg_weight + 3) / 6 * 5))

    prices = df["close"].values.reshape(-1, 1)
    days = np.arange(len(prices)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(days, prices)
    slope = model.coef_[0][0]
    trend_score = 5 if slope > 0.1 else 4 if slope > 0.05 else 3 if abs(slope) <= 0.05 else 2 if slope > -0.1 else 1

    combined_score = 0.7 * base_score + 0.3 * trend_score
    points_df = pd.DataFrame(points)
    return combined_score, points_df

# ==============================
# HEALTH CALCULATION
# ==============================
def compute_health(yahoo_df, cot_df):
    pv_score, points_df = calculate_spectrum_score(yahoo_df)
    cot_df = compute_cot_net_positions(cot_df)
    cot_st = cot_short_term_score(cot_df)
    cot_lt = cot_long_term_score(cot_df)
    cot_score = cot_st * 0.4 + cot_lt * 0.6
    oi_score = open_interest_score(yahoo_df)
    health_raw = pv_score * WEIGHT_PV_RVOL + cot_score * WEIGHT_COT + oi_score * WEIGHT_OI
    health = int(round(health_raw))
    return health, pv_score, cot_score, oi_score, points_df

def health_color(score):
    colors = {
        0: "🔴 Red",
        1: "🟠 Orange-Red",
        2: "🟠 Orange",
        3: "🟡 Yellow",
        4: "🟢 Light Green",
        5: "🟢 Green",
    }
    return colors.get(score, "⚪ Neutral")

# ==============================
# PLOTTING
# ==============================
def plot_points_with_next_day(yahoo_df, points_df, selected_index):
    if points_df.empty:
        st.info("No significant points detected.")
        return
    point = points_df.iloc[selected_index]
    date = point["Date"]
    dates = pd.to_datetime(yahoo_df["date"])
    idx = dates[dates.dt.date == pd.to_datetime(date).date()].index
    if len(idx) == 0:
        st.warning("Selected date not in price data.")
        return
    idx = idx[0]
    start_idx = max(0, idx - 1)
    end_idx = min(len(yahoo_df) - 1, idx + 1)
    plot_df = yahoo_df.iloc[start_idx:end_idx + 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["date"], plot_df["close"], marker="o", linestyle="-", color="blue", label="Close Price")
    ax.set_title(f"{point['Scenario']} on {date} and Next Day Price Action")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Gold & Silver Health Gauge", layout="wide")
st.title("Gold & Silver Health Gauge")
st.markdown("""
This dashboard evaluates the health of Gold and Silver markets using:
- COT data (Commitments of Traders)
- Price, Volume & Relative Volume spectrum analysis
- Open Interest trends
""")

today = datetime.today().date()
default_start = today - timedelta(days=365)

start_date = st.date_input("Start Date", value=default_start)
end_date = st.date_input("End Date", value=today)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if st.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

cols = st.columns(len(COT_MARKETS))

for i, market in enumerate(COT_MARKETS):
    with cols[i]:
        st.header(market)
        symbol = YAHOO_SYMBOLS.get(market)
        if not symbol:
            st.error(f"No Yahoo symbol mapping for market: {market}")
            continue

        cot_df = fetch_cot_data(market)
        yahoo_df = fetch_yahoo_data(symbol, start_date.isoformat(), end_date.isoformat())

        if cot_df.empty:
            st.warning(f"COT data unavailable for {market}")
        if yahoo_df.empty:
            st.warning(f"Yahoo data unavailable for {symbol}")

        if cot_df.empty or yahoo_df.empty:
            st.info("Skipping analysis due to missing data.")
            continue

        health, pv_score, cot_score, oi_score, points_df = compute_health(yahoo_df, cot_df)
        st.metric("Market Health Score", f"{health}/5", delta=None)
        st.write(f"Price & RVOL Spectrum Score: {pv_score:.2f}")
        st.write(f"COT Score: {cot_score:.2f}")
        st.write(f"Open Interest Score: {oi_score:.2f}")
        st.write(f"Health Color: {health_color(health)}")

        if not points_df.empty:
            scenario_options = points_df.index.to_list()
            selected_idx = st.selectbox("Select spectrum scenario point by index", scenario_options)
            plot_points_with_next_day(yahoo_df, points_df, selected_idx)
        else:
            st.info("No scenario points to display.")
