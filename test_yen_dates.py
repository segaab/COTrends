import streamlit as st
import pandas as pd
import numpy as np
import datetime
import logging
from sodapy import Socrata
from sklearn.linear_model import LinearRegression
import yahooquery as yq

# ----------------------------
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Constants & Configurations

APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"

WEIGHT_PV_RVOL = 0.40
WEIGHT_COT = 0.35
WEIGHT_OI = 0.25
COT_SHORT_TERM_WT = 0.40
COT_LONG_TERM_WT = 0.60

LOOKBACK_DAYS_SPECTRUM = 30  # approx 6 weeks trading days

# ----------------------------
# Assets with common display names (COT key -> display name)

ASSETS = {
    "GOLD - COMMODITY EXCHANGE INC.": "Gold",
    "SILVER - COMMODITY EXCHANGE INC.": "Silver",

    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "AUDUSD",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "GBPUSD",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "USDCAD",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "EURUSD",
    "EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE EXCHANGE": "EURGBP",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "JPY",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "CHF",

    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC-USD",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC-USD",
    "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE": "ETH-USD",

    "E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "DJR",

    "WTI CRUDE OIL FINANCIAL - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "PL=F",
    "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": "PA=F",
    "COPPER - COMMODITY EXCHANGE INC.": "HG=F"
}

# Yahoo ticker symbols mapped from display names
YAHOO_SYMBOLS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "AUDUSD": "AUDUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDCAD": "USDCAD=X",
    "EURUSD": "EURUSD=X",
    "EURGBP": "EURGBP=X",
    "JPY": "JPY=X",
    "CHF": "CHF=X",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "ES=F": "ES=F",
    "DJR": "DJR",
    "CL=F": "CL=F",
    "PL=F": "PL=F",
    "PA=F": "PA=F",
    "HG=F": "HG=F"
}

# ----------------------------
# Socrata client cached with timeout

@st.cache_resource(show_spinner=False)
def get_socrata_client():
    logger.info("Initializing Socrata client")
    return Socrata("publicreporting.cftc.gov", APP_TOKEN, timeout=60)

# ----------------------------
# Fetch COT data function

@st.cache_data(ttl=60*60*6, show_spinner=True)
def fetch_cot_data(market_name):
    client = get_socrata_client()
    where_clause = f"market_and_exchange_names='{market_name}'"
    try:
        results = client.get("6dca-aqww", where=where_clause, order="report_date DESC", limit=1000)
        if not results:
            logger.warning(f"No COT data returned for {market_name}")
            return pd.DataFrame()
        df = pd.DataFrame.from_records(results)
        df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors='coerce')
        cols = [
            "noncomm_positions_long_all", "noncomm_positions_short_all",
            "comm_positions_long_all", "comm_positions_short_all",
            "open_interest_all"
        ]
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("report_date")
    except Exception as e:
        logger.error(f"Error fetching COT data for {market_name}: {e}")
        return pd.DataFrame()

# ----------------------------
# Fetch Yahoo Finance data function

@st.cache_data(ttl=60*60*2, show_spinner=True)
def fetch_yahoo_data(symbol, start_date, end_date):
    try:
        ticker = yq.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        if data.empty:
            logger.warning(f"No Yahoo data for symbol {symbol}")
            return pd.DataFrame()
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()
        else:
            data = data.reset_index()
        data.rename(columns={"date": "date", "close": "Close", "volume": "Volume", "openInterest": "OpenInterest"}, inplace=True)
        data["date"] = pd.to_datetime(data["date"], errors='coerce')
        return data[["date", "Close", "Volume", "OpenInterest"]].dropna(subset=["date"])
    except Exception as e:
        logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
        return pd.DataFrame()

# ----------------------------
# Calculate relative volume

def calculate_rvol(df):
    df = df.sort_values("date").copy()
    df["AvgVol20"] = df["Volume"].rolling(window=20).mean()
    df["RVOL"] = df["Volume"] / df["AvgVol20"]
    df.dropna(subset=["RVOL"], inplace=True)
    return df

# ----------------------------
# Compute COT net positions

def compute_cot_net_positions(cot_df):
    cot_df = cot_df.sort_values("report_date").copy()
    cot_df["net_comm"] = cot_df["comm_positions_long_all"] - cot_df["comm_positions_short_all"]
    cot_df["net_noncomm"] = cot_df["noncomm_positions_long_all"] - cot_df["noncomm_positions_short_all"]
    return cot_df

# ----------------------------
# COT short term score

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

# ----------------------------
# COT long term score

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

# ----------------------------
# Open interest score

def open_interest_score(yahoo_df):
    if yahoo_df.empty or "OpenInterest" not in yahoo_df.columns or yahoo_df["OpenInterest"].isna().all():
        return 3
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

# ----------------------------
# Spectrum score and detection points

def calculate_spectrum_score(df):
    df = df.sort_values("date").copy()
    df = df.tail(LOOKBACK_DAYS_SPECTRUM)
    if len(df) < LOOKBACK_DAYS_SPECTRUM:
        return 3, pd.DataFrame()

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

    for idx in range(1, len(df) - 1):
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
                "Date": row["date"].date(),
                "Type": point_type,
                "Return %": round(ret_pct, 2),
                "Next Day Return %": round(next_ret, 2),
                "Weight": weight,
            })

    avg_weight = total_weight / count if count > 0 else 0
    spectrum_score = trend_score + avg_weight
    spectrum_score = max(1, min(5, spectrum_score))  # Clamp between 1 and 5

    points_df = pd.DataFrame(detected_points)

    return spectrum_score, points_df

# ----------------------------
# Main Streamlit app function

def main():
    st.title("Gold, Silver & Other Markets Health Gauge")

    st.markdown("""
    This dashboard evaluates the health of multiple markets using:
    - COT data (Commitments of Traders)
    - Price, Volume & Relative Volume spectrum analysis
    - Open Interest trends
    """)

    start_date = st.date_input("Start Date", datetime.date(2024, 8, 9))
    end_date = st.date_input("End Date", datetime.date(2025, 8, 9))

    if start_date > end_date:
        st.error("Start Date must be before End Date.")
        return

    for cot_name, display_name in ASSETS.items():
        with st.spinner(f"Fetching data for {display_name}..."):
            cot_df = fetch_cot_data(cot_name)
            yahoo_symbol = YAHOO_SYMBOLS.get(display_name)
            yahoo_df = pd.DataFrame()
            if yahoo_symbol:
                yahoo_df = fetch_yahoo_data(yahoo_symbol, start_date.isoformat(), end_date.isoformat())

        st.header(f"{display_name} Market Health")

        if cot_df.empty:
            st.warning(f"No COT data available for {display_name}. Skipping analysis.")
            continue

        if yahoo_df.empty:
            st.warning(f"No price/volume data available for {display_name}. Skipping analysis.")
            continue

        # Compute scores
        cot_df = compute_cot_net_positions(cot_df)
        short_score = cot_short_term_score(cot_df)
        long_score = cot_long_term_score(cot_df)
        cot_score = COT_SHORT_TERM_WT * short_score + COT_LONG_TERM_WT * long_score
        oi_score = open_interest_score(yahoo_df)
        spectrum_score, spectrum_points = calculate_spectrum_score(yahoo_df)

        overall_score = (
            WEIGHT_PV_RVOL * spectrum_score +
            WEIGHT_COT * cot_score +
            WEIGHT_OI * oi_score
        )

        st.write(f"**COT Short-term Score:** {short_score:.2f}")
        st.write(f"**COT Long-term Score:** {long_score:.2f}")
        st.write(f"**Combined COT Score:** {cot_score:.2f}")
        st.write(f"**Open Interest Score:** {oi_score:.2f}")
        st.write(f"**Spectrum Score:** {spectrum_score:.2f}")
        st.write(f"**Overall Health Score:** {overall_score:.2f} (1=Weakest, 5=Strongest)")

        # Display spectrum points if any
        if not spectrum_points.empty:
            st.subheader("Spectrum Detected Points")
            st.dataframe(spectrum_points)

        # Price chart
        st.line_chart(yahoo_df.set_index("date")["Close"])

if __name__ == "__main__":
    main()
