import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
from sodapy import Socrata
from yahooquery import Ticker
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# --- Constants ---
COT_MARKETS = [
    "GOLD - COMMODITY EXCHANGE INC.",
    "SILVER - COMMODITY EXCHANGE INC.",
]

YAHOO_SYMBOLS = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
}

LOOKBACK_DAYS_SPECTRUM = 30
WEIGHT_PV_RVOL = 0.40
WEIGHT_COT = 0.35
WEIGHT_OI = 0.25
COT_SHORT_TERM_WT = 0.40
COT_LONG_TERM_WT = 0.60
TIMEOUT = 60  # seconds

# Replace with your actual Socrata App Token or environment variable
APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"


# --- Helper Functions ---

@st.cache_resource(show_spinner=False)
def get_socrata_client():
    try:
        client = Socrata("publicreporting.cftc.gov", APP_TOKEN, timeout=TIMEOUT)
        st.info("Socrata client initialized.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Socrata client: {e}")
        return None


@st.cache_data(ttl=60*60*6, show_spinner=True)
def fetch_cot_data(market_name):
    client = get_socrata_client()
    if client is None:
        st.error("Cannot fetch COT data because Socrata client is unavailable.")
        return pd.DataFrame()

    where_clause = f"market_and_exchange_names='{market_name}'"  # Correct column name
    try:
        results = client.get("6dca-aqww", where=where_clause, order="report_date DESC", limit=1000)
        if not results:
            st.warning(f"No COT data returned for {market_name}.")
            return pd.DataFrame()

        df = pd.DataFrame.from_records(results)
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

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

        # Ensure date column is timezone naive before .dt.date
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        if data["date"].dt.tz is not None:
            data["date"] = data["date"].dt.tz_localize(None)
        data["date"] = data["date"].dt.date

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
    if slope > 0:
        return 5
    elif slope < 0:
        return 0
    else:
        return 3


def calculate_spectrum_score(df):
    df = df.sort_values("date").copy()
    df = df.tail(LOOKBACK_DAYS_SPECTRUM)
    if len(df) < LOOKBACK_DAYS_SPECTRUM:
        return 3, pd.DataFrame()

    df = calculate_rvol(df)
    if df.empty:
        return 3, pd.DataFrame()

    rvol_78th = df["rvol"].quantile(0.78)

    total_weight = 0
    count = 0

    detected_points = []

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

    if len(df) < 10:
        trend_score = 3
    else:
        prices = df["close"].values.reshape(-1, 1)
        days = np.arange(len(prices)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(days, prices)
        slope = model.coef_[0][0]
        if slope > 0.1:
            trend_score = 5
        elif 0.05 < slope <= 0.1:
            trend_score = 4
        elif -0.05 <= slope <= 0.05:
            trend_score = 3
        elif -0.1 <= slope < -0.05:
            trend_score = 2
        else:
            trend_score = 1

    for idx in range(1, len(df) - 1):
        row = df.iloc[idx]
        if row["rvol"] >= rvol_78th:
            prev_close = df.iloc[idx - 1]["close"]
            if prev_close == 0:
                continue
            ret_pct = ((row["close"] - prev_close) / prev_close) * 100
            if ret_pct >= 0:
                weight = accumulation_weight(ret_pct)
                point_type = "Accumulation"
            else:
                weight = distribution_weight(ret_pct)
                point_type = "Distribution"

            total_weight += weight
            count += 1

            next_close = df.iloc[idx + 1]["close"]
            next_ret = ((next_close - row["close"]) / row["close"]) * 100

            detected_points.append({
                "Date": row["date"],
                "Type": point_type,
                "Return %": round(ret_pct, 2),
                "Next Day Return %": round(next_ret, 2),
                "Weight": weight,
            })

    if count == 0:
        base_score = 3
    else:
        avg_weight = total_weight / count
        base_score = (avg_weight + 3) / 6 * 5
        base_score = max(0, min(5, base_score))

    combined_score = 0.7 * base_score + 0.3 * trend_score
    return combined_score, pd.DataFrame(detected_points)


def compute_asset_health(yahoo_df, cot_df):
    pv_score, points_df = calculate_spectrum_score(yahoo_df)
    cot_df = compute_cot_net_positions(cot_df)
    cot_st = cot_short_term_score(cot_df)
    cot_lt = cot_long_term_score(cot_df)
    cot_score = cot_st * COT_SHORT_TERM_WT + cot_lt * COT_LONG_TERM_WT
    oi_score = open_interest_score(yahoo_df)
    health_raw = pv_score * WEIGHT_PV_RVOL + cot_score * WEIGHT_COT + oi_score * WEIGHT_OI
    health = int(round(health_raw))
    return health, pv_score, cot_score, oi_score, points_df


def plot_price_action_with_points(yahoo_df, points_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yahoo_df["date"], y=yahoo_df["close"], mode='lines+markers', name="Close Price"))
    if not points_df.empty:
        accum_points = points_df[points_df["Type"] == "Accumulation"]
        distrib_points = points_df[points_df["Type"] == "Distribution"]
        fig.add_trace(go.Scatter(x=accum_points["Date"], y=[yahoo_df[yahoo_df["date"] == d]["close"].values[0] for d in accum_points["Date"]],
                                 mode='markers', name="Accumulation", marker=dict(color='green', size=10, symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=distrib_points["Date"], y=[yahoo_df[yahoo_df["date"] == d]["close"].values[0] for d in distrib_points["Date"]],
                                 mode='markers', name="Distribution", marker=dict(color='red', size=10, symbol='triangle-down')))
    fig.update_layout(title="Price Action with Accumulation/Distribution Points", xaxis_title="Date", yaxis_title="Price")
    return fig


# --- Streamlit UI ---

st.title("Gold & Silver Health Gauge")

start_date = st.date_input("Start Date", datetime.date(2024, 8, 9))
end_date = st.date_input("End Date", datetime.date(2025, 8, 9))

market = st.selectbox("Select Market", COT_MARKETS)
yahoo_symbol = YAHOO_SYMBOLS[market]

if start_date >= end_date:
    st.error("Start Date must be before End Date.")
else:
    cot_df = fetch_cot_data(market)
    yahoo_df = fetch_yahoo_data(yahoo_symbol, start_date.isoformat(), end_date.isoformat())

    if cot_df.empty or yahoo_df.empty:
        st.warning("Data is incomplete, cannot compute health gauge.")
    else:
        health, pv_score, cot_score, oi_score, points_df = compute_asset_health(yahoo_df, cot_df)
        st.metric(label=f"{market} Health Score (0-5)", value=health)
        st.write("Price/Volume RVOL Score:", round(pv_score, 2))
        st.write("COT Score:", round(cot_score, 2))
        st.write("Open Interest Score:", round(oi_score, 2))

        st.subheader("Accumulation/Distribution Points Detected")
        if points_df.empty:
            st.write("No significant points detected in the selected period.")
        else:
            st.dataframe(points_df)

        st.subheader("Price Action Chart")
        fig = plot_price_action_with_points(yahoo_df, points_df)
        st.plotly_chart(fig, use_container_width=True)
