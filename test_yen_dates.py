# gold_silver_health_gauge.py
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from yahooquery import Ticker
from sodapy import Socrata
import os

st.set_page_config(page_title="Gold & Silver Health Gauge", layout="wide")

# --- Sidebar Parameters ---
st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.today())

gold_weight = st.sidebar.slider("Gold Weight", 0.0, 1.0, 0.5, 0.05)
silver_weight = st.sidebar.slider("Silver Weight", 0.0, 1.0, 0.5, 0.05)

# --- Fetch Gold & Silver Price Data ---
@st.cache_data
def get_price_data(symbol, start, end):
    t = Ticker(symbol)
    hist = t.history(start=start, end=end)
    if isinstance(hist, pd.DataFrame):
        hist = hist.reset_index()
    return hist

st.subheader("Fetching Market Data...")
gold_df = get_price_data("GC=F", start_date, end_date)
silver_df = get_price_data("SI=F", start_date, end_date)

# --- Compute Simple Momentum/Trend Scores ---
def compute_trend_score(df):
    if df is None or df.empty:
        return None
    df["MA50"] = df["close"].rolling(50).mean()
    df["MA200"] = df["close"].rolling(200).mean()
    latest = df.iloc[-1]
    score = 0
    if latest["close"] > latest["MA50"]:
        score += 0.5
    if latest["close"] > latest["MA200"]:
        score += 0.5
    return round(score, 2)

gold_score = compute_trend_score(gold_df)
silver_score = compute_trend_score(silver_df)

# --- Fetch COT Data ---
@st.cache_data
def get_cot_data(market):
    try:
        client = Socrata(
            "publicreporting.cftc.gov",
            os.getenv("MY_APP_TOKEN"),
            username=os.getenv("CFTC_USERNAME"),
            password=os.getenv("CFTC_PASSWORD"),
        )
        result = client.get("6dca-aqww", where=f"market_and_exchange_names = '{market}'", limit=5000)
        df = pd.DataFrame.from_records(result)
        return df
    except Exception as e:
        st.warning(f"COT data unavailable for {market}: {e}")
        return pd.DataFrame()

gold_cot = get_cot_data("GOLD - COMMODITY EXCHANGE INC.")
silver_cot = get_cot_data("SILVER - COMMODITY EXCHANGE INC.")

# --- Combine into Overall Health Score ---
def compute_health(g_score, s_score, gw, sw):
    if g_score is None or s_score is None:
        return None
    return round(g_score * gw + s_score * sw, 2)

overall_health = compute_health(gold_score, silver_score, gold_weight, silver_weight)

# --- Display ---
col1, col2, col3 = st.columns(3)
col1.metric("Gold Score", gold_score)
col2.metric("Silver Score", silver_score)
col3.metric("Overall Health", overall_health)

st.subheader("Gold Price Data")
st.dataframe(gold_df.tail())

st.subheader("Silver Price Data")
st.dataframe(silver_df.tail())

# --- Export CSV ---
combined_df = pd.concat([
    gold_df.assign(asset="Gold"),
    silver_df.assign(asset="Silver")
])
csv = combined_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Data as CSV", csv, "gold_silver_data.csv", "text/csv")

st.markdown("Data source: Yahoo Finance (prices) & CFTC Socrata (COT).")
