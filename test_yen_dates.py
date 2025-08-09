import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sodapy import Socrata
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Constants
COT_MARKETS = [
    "GOLD - COMMODITY EXCHANGE INC.",
    "SILVER - COMMODITY EXCHANGE INC.",
]

YAHOO_SYMBOLS = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
}

LOOKBACK_DAYS_SPECTRUM = 30  # Approx 6 weeks of trading days
WEIGHT_PV_RVOL = 0.40
WEIGHT_COT = 0.35
WEIGHT_OI = 0.25
COT_SHORT_TERM_WT = 0.40
COT_LONG_TERM_WT = 0.60

APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"  # Replace with your Socrata App Token

# --- Caching Socrata client ---
@st.cache_resource(show_spinner=False)
def get_socrata_client():
    return Socrata("publicreporting.cftc.gov", APP_TOKEN)

# --- Fetch COT Data ---
@st.cache_data(ttl=60*60*6, show_spinner=True)
def fetch_cot_data(market_name):
    client = get_socrata_client()
    where_clause = f"market_and_exchange_name='{market_name}'"
    results = client.get("6dca-aqww", where=where_clause, order="report_date DESC", limit=1000)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(results)
    # Convert dates
    df["report_date"] = pd.to_datetime(df["report_date"])
    # Convert numeric columns to float
    for col in [
        "noncomm_positions_long_all", "noncomm_positions_short_all", "noncomm_positions_spread_all",
        "comm_positions_long_all", "comm_positions_short_all",
        "tot_rept_positions_long_all", "tot_rept_positions_short",
        "nonrept_positions_long_all", "nonrept_positions_short_all",
        "open_interest_all"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("report_date")

# --- Fetch Yahoo Data ---
@st.cache_data(ttl=60*60*2, show_spinner=True)
def fetch_yahoo_data(symbol, start_date, end_date):
    import yahooquery as yq
    try:
        data = yq.Ticker(symbol).history(start=start_date, end=end_date)
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    # Reset index in case date is index or multiindex
    data = data.reset_index()
    # Make sure 'date' column exists
    if "date" not in data.columns:
        possible_date_cols = [col for col in data.columns if "date" in col.lower()]
        if possible_date_cols:
            data.rename(columns={possible_date_cols[0]: "date"}, inplace=True)
        else:
            return pd.DataFrame()
    # Convert date column to datetime
    try:
        data["date"] = pd.to_datetime(data["date"], errors='coerce')
    except Exception:
        return pd.DataFrame()
    data = data.dropna(subset=["date"])
    data["date"] = data["date"].dt.date

    # Ensure required columns exist
    cols_needed = ["date", "close", "volume", "openInterest"]
    for c in cols_needed:
        if c not in data.columns:
            data[c] = np.nan

    data = data[cols_needed]
    data.rename(columns={"close": "Close", "volume": "Volume", "openInterest": "OpenInterest"}, inplace=True)
    data = data.dropna(subset=["Close"])
    return data

# --- Calculate Relative Volume (RVOL) ---
def calculate_rvol(df):
    df = df.sort_values("date")
    df["AvgVol20"] = df["Volume"].rolling(window=20).mean()
    df["RVOL"] = df["Volume"] / df["AvgVol20"]
    df.dropna(subset=["RVOL"], inplace=True)
    return df

# --- Calculate net positions for COT ---
def compute_cot_net_positions(cot_df):
    cot_df = cot_df.sort_values("report_date")
    cot_df["net_comm"] = cot_df["comm_positions_long_all"] - cot_df["comm_positions_short_all"]
    cot_df["net_noncomm"] = cot_df["noncomm_positions_long_all"] - cot_df["noncomm_positions_short_all"]
    return cot_df

# --- Calculate Short-term COT Score ---
def cot_short_term_score(cot_df):
    if cot_df.shape[0] < 2:
        return 3  # Neutral
    latest = cot_df.iloc[-1]
    prev = cot_df.iloc[-2]
    delta_comm = latest["net_comm"] - prev["net_comm"]
    delta_noncomm = latest["net_noncomm"] - prev["net_noncomm"]
    comm_score = 5 if delta_comm > 0 else 0 if delta_comm < 0 else 3
    noncomm_score = 5 if delta_noncomm > 0 else 0 if delta_noncomm < 0 else 3
    return (comm_score + noncomm_score) / 2

# --- Calculate Long-term COT Score ---
def cot_long_term_score(cot_df):
    if cot_df.shape[0] < 12:
        return 3  # Neutral
    df_12 = cot_df.tail(12)
    net_comm = df_12["net_comm"].values
    net_noncomm = df_12["net_noncomm"].values
    comm_trend = np.polyfit(range(12), net_comm, 1)[0]
    noncomm_trend = np.polyfit(range(12), net_noncomm, 1)[0]
    comm_score = 5 if comm_trend < 0 else 0 if comm_trend > 0 else 3
    noncomm_score = 5 if noncomm_trend > 0 else 0 if noncomm_trend < 0 else 3
    return (comm_score + noncomm_score) / 2

# --- Open Interest Score ---
def open_interest_score(yahoo_df):
    if yahoo_df.empty or "OpenInterest" not in yahoo_df.columns or yahoo_df["OpenInterest"].isna().all():
        return 3  # Neutral if no data
    oi = yahoo_df["OpenInterest"].dropna()
    if len(oi) < 20:
        return 3
    slope = np.polyfit(range(len(oi)), oi, 1)[0]
    if slope > 0:
        return 5
    elif slope < 0:
        return 0
    else:
        return 3

# --- Spectrum Score and Event Detection ---
def calculate_spectrum_score(df):
    df = df.sort_values("date").copy()
    df = df.tail(LOOKBACK_DAYS_SPECTRUM)
    if len(df) < LOOKBACK_DAYS_SPECTRUM:
        return 3, pd.DataFrame()  # neutral if insufficient data, no points

    df = calculate_rvol(df)
    if df.empty:
        return 3, pd.DataFrame()

    rvol_78th = df["RVOL"].quantile(0.78)

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

    # Trend detection via linear regression slope of price
    if len(df) < 10:
        trend_score = 3
    else:
        prices = df["Close"].values.reshape(-1, 1)
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

    for idx in range(1, len(df) - 1):  # leave room for next day
        row = df.iloc[idx]
        if row["RVOL"] >= rvol_78th:
            prev_close = df.iloc[idx - 1]["Close"]
            if prev_close == 0:
                continue
            ret_pct = ((row["Close"] - prev_close) / prev_close) * 100
            if ret_pct >= 0:
                weight = accumulation_weight(ret_pct)
                point_type = "Accumulation"
            else:
                weight = distribution_weight(ret_pct)
                point_type = "Distribution"

            total_weight += weight
            count += 1

            next_close = df.iloc[idx + 1]["Close"]
            next_ret = ((next_close - row["Close"]) / row["Close"]) * 100

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

# --- Compute overall asset health ---
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

# --- Health color ---
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

# --- Plot accumulation/distribution point and next day ---
def plot_point_and_next_day(yahoo_df, point_date):
    df = yahoo_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    if point_date not in df["date"].values:
        st.warning(f"Date {point_date} not in price data.")
        return

    idx = df.index[df["date"] == point_date][0]

    start_idx = max(0, idx - 1)
    end_idx = min(len(df) - 1, idx + 1)

    plot_df = df.loc[start_idx:end_idx].reset_index(drop=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df["date"],
        y=plot_df["Close"],
        mode="lines+markers",
        line=dict(color="blue"),
        name="Close Price",
    ))

    # Highlight the event day
    fig.add_vrect(
        x0=plot_df["date"].iloc[1],
        x1=plot_df["date"].iloc[1],
        fillcolor="LightGreen" if plot_df["date"].iloc[1] == point_date else "LightCoral",
        opacity=0.3,
        layer="below",
        line_width=0,
    )
    # Highlight the next day
    if end_idx > idx:
        fig.add_vrect(
            x0=plot_df["date"].iloc[2],
            x1=plot_df["date"].iloc[2],
            fillcolor="LightBlue",
            opacity=0.3,
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title=f"Price around {point_date} (Event day & next day)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Main Streamlit App ---
st.set_page_config(page_title="Gold & Silver Health Gauge", layout="wide")

st.title("Gold & Silver Health Gauge")

st.markdown(
    """
    This dashboard evaluates the health of Gold and Silver markets using:
    - COT data (Commitments of Traders)
    - Price, Volume & Relative Volume spectrum analysis
    - Open Interest trends
    """
)

today = datetime.date.today()
default_start = today - datetime.timedelta(days=365)

start_date = st.date_input("Start Date", value=default_start)
end_date = st.date_input("End Date", value=today)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if st.button("Refresh Data"):
    st.cache_data.clear()

cols = st.columns(len(COT_MARKETS))

for i, market in enumerate(COT_MARKETS):
    with cols[i]:
        st.header(market)
        yahoo_symbol = YAHOO_SYMBOLS.get(market)
        if not yahoo_symbol:
            st.error(f"Yahoo symbol for {market} not found.")
            continue

        with st.spinner(f"Fetching Yahoo data for {market}..."):
            yahoo_df = fetch_yahoo_data(yahoo_symbol, start_date.isoformat(), end_date.isoformat())
        with st.spinner(f"Fetching COT data for {market}..."):
            cot_df = fetch_cot_data(market)

        if yahoo_df.empty:
            st.warning("No Yahoo data available.")
            continue
        if cot_df.empty:
            st.warning("No COT data available.")
            continue

        health, pv_score, cot_score, oi_score, points_df = compute_asset_health(yahoo_df, cot_df)

        st.markdown(f"### Health Gauge Score: {health} / 5 {health_color(health)}")
        st.write(f"- Price + Volume Spectrum Score: {pv_score:.2f}")
        st.write(f"- COT Score (Short+Long Term): {cot_score:.2f}")
        st.write(f"- Open Interest Score: {oi_score:.2f}")

        if not points_df.empty:
            st.subheader("Accumulation/Distribution Points & Next Day Returns")
            point_idx = st.select_slider(
                f"Select event date for {market}:",
                options=points_df.index.tolist(),
                format_func=lambda x: f"{points_df.loc[x,'Date']} ({points_df.loc[x,'Type']})"
            )
            selected_date = points_df.loc[point_idx, "Date"]
            plot_point_and_next_day(yahoo_df, selected_date)
        else:
            st.info("No significant accumulation/distribution points detected.")
