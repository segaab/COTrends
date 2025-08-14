import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import os
import logging
import threading
from sodapy import Socrata
from yahooquery import Ticker
from huggingface_hub import InferenceClient
from sklearn.linear_model import LinearRegression

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Setup ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# Initialize HF client once, expects HF_TOKEN in environment
hf_client = InferenceClient(provider="cerebras", api_key=os.getenv("HF_TOKEN"))

# --- Asset mapping (COT market names -> Yahoo futures tickers) ---
assets = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "PL=F",
    "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": "PA=F",
    "WTI CRUDE OIL FINANCIAL - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "COPPER - COMMODITY EXCHANGE INC.": "HG=F",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6A=F",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "6B=F",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6C=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J=F",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S=F",
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC=F",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE": "MBT=F",
    "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE": "ETH=F",
    "E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "DJR",
}# --- Fetch COT Data ---
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching COT data for {market_name}")
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get(
                "6dca-aqww",
                where=where_clause,
                order="report_date_as_yyyy_mm_dd DESC",
                limit=1500,
            )
            if results:
                df = pd.DataFrame.from_records(results)
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
                df["open_interest_all"] = pd.to_numeric(df["open_interest_all"], errors="coerce")
                try:
                    df["commercial_net"] = df["commercial_long_all"].astype(float) - df["commercial_short_all"].astype(float)
                    df["non_commercial_net"] = df["non_commercial_long_all"].astype(float) - df["non_commercial_short_all"].astype(float)
                except KeyError:
                    df["commercial_net"] = 0.0
                    df["non_commercial_net"] = 0.0
                return df.sort_values("report_date")
            else:
                logger.warning(f"No COT data for {market_name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching COT data for {market_name}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching COT data for {market_name} after {max_attempts} attempts.")
    return pd.DataFrame()

# --- Fetch Price Data (yahooquery) ---
def fetch_yahooquery_data(ticker: str, start_date: str, end_date: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching Yahoo data for {ticker} from {start_date} to {end_date}")
    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.loc[ticker]
            hist = hist.reset_index()
            hist["date"] = pd.to_datetime(hist["date"])
            return hist.sort_values("date")
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching Yahoo data for {ticker} after {max_attempts} attempts.")
    return pd.DataFrame()

# --- Calculate Relative Volume (RVOL) ---
def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    vol_col = "volume" if "volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
    if vol_col is None:
        df["rvol"] = np.nan
        return df
    rolling_avg = df[vol_col].rolling(window).mean()
    df["rvol"] = df[vol_col] / rolling_avg
    return df

# --- Merge COT and Price Data ---
def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df.empty or price_df.empty:
        return pd.DataFrame()
    cot_df_small = cot_df[["report_date", "open_interest_all", "commercial_net", "non_commercial_net"]].copy()
    cot_df_small.rename(columns={"report_date": "date"}, inplace=True)
    cot_df_small["date"] = pd.to_datetime(cot_df_small["date"])
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values("date")
    cot_df_small = cot_df_small.sort_values("date")
    full_dates = pd.DataFrame({"date": pd.date_range(price_df["date"].min(), price_df["date"].max())})
    cot_df_filled = pd.merge_asof(full_dates, cot_df_small, on="date", direction="backward")
    merged = pd.merge(price_df, cot_df_filled[["date", "open_interest_all", "commercial_net", "non_commercial_net"]], on="date", how="left")
    merged["open_interest_all"] = merged["open_interest_all"].ffill()
    merged["commercial_net"] = merged["commercial_net"].ffill()
    merged["non_commercial_net"] = merged["non_commercial_net"].ffill()
    return merged

# --- Health Gauge Calculation ---
def calculate_health_gauge(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> float:
    if cot_df.empty or price_df.empty:
        return np.nan
    last_date = pd.to_datetime(price_df["date"]).max()
    one_year_ago = last_date - pd.Timedelta(days=365)
    three_months_ago = last_date - pd.Timedelta(days=90)

    oi = price_df["open_interest_all"].dropna()
    oi_score = float((oi - oi.min()) / (oi.max() - oi.min() + 1e-9).iloc[-1]) if not oi.empty else 0.0

    commercial = cot_df[["report_date", "commercial_net"]].dropna().copy()
    commercial["report_date"] = pd.to_datetime(commercial["report_date"])
    st_score = 0.0
    short_term = commercial[commercial["report_date"] >= three_months_ago]
    if not short_term.empty:
        st_score = float((short_term["commercial_net"].iloc[-1] - short_term["commercial_net"].min()) / 
                         (short_term["commercial_net"].max() - short_term["commercial_net"].min() + 1e-9))

    noncomm = cot_df[["report_date", "non_commercial_net"]].dropna().copy()
    noncomm["report_date"] = pd.to_datetime(noncomm["report_date"])
    lt_score = 0.0
    long_term = noncomm[noncomm["report_date"] >= one_year_ago]
    if not long_term.empty:
        lt_score = float((long_term["non_commercial_net"].iloc[-1] - long_term["non_commercial_net"].min()) /
                         (long_term["non_commercial_net"].max() - long_term["non_commercial_net"].min() + 1e-9))

    cot_analytics = 0.4 * st_score + 0.6 * lt_score

    recent = price_df[pd.to_datetime(price_df["date"]) >= three_months_ago].copy()
    pv_score = 0.0
    if not recent.empty:
        close_col = "close" if "close" in recent.columns else ("Close" if "Close" in recent.columns else None)
        vol_col = "volume" if "volume" in recent.columns else ("Volume" if "Volume" in recent.columns else None)
        if close_col and vol_col and "rvol" in recent.columns:
            recent["return"] = recent[close_col].pct_change().fillna(0.0)
            rvol_75 = recent["rvol"].quantile(0.75)
            recent["vol_avg20"] = recent[vol_col].rolling(20).mean()
            recent["vol_spike"] = recent[vol_col] > recent["vol_avg20"]
            filt = recent[(recent["rvol"] >= rvol_75) & (recent["vol_spike"])]
            if not filt.empty:
                last_ret = float(filt["return"].iloc[-1])
                bucket = 5 if last_ret >= 0.02 else 4 if last_ret >= 0.01 else 3 if last_ret >= -0.01 else 2 if last_ret >= -0.02 else 1
                pv_score = (bucket - 1) / 4.0

    return float((0.25 * oi_score + 0.35 * cot_analytics + 0.40 * pv_score) * 10.0)

# --- Threaded batch fetching (unchanged) ---
# ... (reuse fetch_batch and fetch_all_data from previous chunk)

# --- Forecasting Module ---
def generate_forecasts(price_results: dict, forecast_days: int = 7) -> dict:
    forecasts = {}
    for asset, df in price_results.items():
        if df.empty:
            forecasts[asset] = []
            continue
        close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
        if close_col is None or df.shape[0] < 10:
            forecasts[asset] = []
            continue
        recent = df.tail(30).copy()
        recent["day_index"] = np.arange(len(recent))
        X = recent[["day_index"]].values
        y = recent[close_col].values
        model = LinearRegression()
        model.fit(X, y)
        future_index = np.arange(len(recent), len(recent) + forecast_days).reshape(-1, 1)
        pred = model.predict(future_index)
        forecasts[asset] = pred.tolist()
    return forecasts

# --- LLM Prompt Builder ---
def build_llm_prompt(price_results: dict, cot_results: dict, forecasts: dict) -> str:
    prompt = (
        "You are an expert market analyst.\n\n"
        "For each asset, generate a separate titled paragraph using <div> formatting for the title. "
        "Format your response in ChatGPT-style paragraphs. Include next 7-day forecasts if provided.\n\n"
    )

    for cot_name, merged_df in price_results.items():
        if merged_df.empty or "health_score" not in merged_df.columns:
            continue
        health_score = float(merged_df["health_score"].iloc[-1])
        recent = merged_df.tail(90).copy()
        close_col = "close" if "close" in recent.columns else ("Close" if "Close" in recent.columns else None)
        if close_col is None:
            continue
        current_price = float(recent[close_col].iloc[-1])
        high_7d, low_7d = float(recent[close_col].tail(7).max()), float(recent[close_col].tail(7).min())
        high_30d, low_30d = float(recent[close_col].tail(30).max()), float(recent[close_col].tail(30).min())
        forecasted = forecasts.get(cot_name, [])
        forecast_text = ""
        if forecasted:
            forecast_text = "Next 7-day forecasted prices: " + ", ".join([f"{p:.2f}" for p in forecasted]) + ". "

        prompt += (
            f"<div style='font-size:18px;font-weight:bold;'>{cot_name}</div>\n"
            f"Health Gauge Score: {health_score:.2f}. "
            f"Current price: {current_price:.2f}. "
            f"Last 7 sessions: {low_7d:.2f}-{high_7d:.2f}, "
            f"last 30 sessions: {low_30d:.2f}-{high_30d:.2f}. "
            f"{forecast_text}\n"
        )

        if health_score < 3:
            prompt += (
                "This low reading suggests a sell regime may be developing or already active. "
                "Give more weight to sessions where price declines coincide with relative volume spikes. "
            )
        elif health_score > 5:
            prompt += (
                "This elevated reading indicates a buy regime may be developing or already in place. "
                "Prioritize sessions where price advances occur alongside relative volume spikes. "
            )
        else:
            prompt += (
                "The reading is moderate, pointing to possible consolidation or indecision. "
                "Watch for expansion days where relative volume spikes in the direction of the break. "
            )

        prompt += f"Key levels to monitor: 7-day high {high_7d:.2f}, 7-day low {low_7d:.2f}, 30-day high {high_30d:.2f}, 30-day low {low_30d:.2f}.\n\n"

    return prompt

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="COT Futures Health Gauge & Newsletter", layout="wide")
    st.title("COT Futures Health Gauge Dashboard")

    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "cot_results" not in st.session_state:
        st.session_state.cot_results = {}
    if "price_results" not in st.session_state:
        st.session_state.price_results = {}
    if "forecasts" not in st.session_state:
        st.session_state.forecasts = {}
    if "newsletter_text" not in st.session_state:
        st.session_state.newsletter_text = ""

    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365)
    start_date = st.date_input("Select Start Date", default_start)
    end_date = st.date_input("Select End Date", today)
    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
        return

    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            cot_res, price_res = fetch_all_data(assets, start_date, end_date, batch_size=5)
            st.session_state.cot_results = cot_res
            st.session_state.price_results = price_res
            st.session_state.forecasts = generate_forecasts(price_res)
            st.session_state.data_loaded = True
        st.success("Data loaded successfully. You may now generate the newsletter.")

    if st.button("Generate Newsletter"):
        if not st.session_state.data_loaded or not st.session_state.price_results:
            st.warning("Data not ready yet.")
        else:
            with st.spinner("Generating newsletter..."):
                prompt = build_llm_prompt(
                    st.session_state.price_results,
                    st.session_state.cot_results,
                    st.session_state.forecasts,
                )
                try:
                    completion = hf_client.chat.completions.create(
                        model="meta-llama/Llama-3.1-8B-Instruct",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    msg = completion.choices[0].message
                    content = msg.content if hasattr(msg, "content") else msg["content"]
                    st.session_state.newsletter_text = content
                except Exception as e:
                    st.error(f"Failed to generate newsletter: {e}")

    if st.session_state.newsletter_text:
        st.markdown("### Market Newsletter")
        st.write(st.session_state.newsletter_text)

if __name__ == "__main__":
    main()
