"""
Streamlit dashboard: Fed Funds futures (ZQ=F) + SOFR combined-rate (35th percentile filter)
- Uses yahooquery for market prices and fredapi for SOFR/FEDFUNDS
- Combined Rate = (SOFR_rolling_avg + ZQ_implied_rate) / 2
- Entry rule: combined_rate < HISTORICAL_35TH_PERCENTILE (calculated on full combined series)
- Fixed data window: yesterday 23:00 ET minus 365 days -> yesterday 23:00 ET
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from fredapi import Fred
from yahooquery import Ticker

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Combined Rate (35th pct) Strategy Dashboard", layout="wide")
FRED_API_KEY = "91bb2c5920fb8f843abdbbfdfcab5345"
CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------
# Date range: yesterday 23:00 ET back 365 days
# ----------------------------
now_utc = dt.datetime.utcnow()
et_offset = dt.timedelta(hours=4)  # approximate ET offset (adjust if you need exact ET handling)
now_et = now_utc - et_offset
yesterday_et = dt.datetime(year=now_et.year, month=now_et.month, day=now_et.day) - dt.timedelta(days=1)
yesterday_23 = yesterday_et.replace(hour=23, minute=0, second=0, microsecond=0)
START_DATE = yesterday_23 - dt.timedelta(days=365)
END_DATE = yesterday_23

# ----------------------------
# Helper: fetch price history via yahooquery (robust datetime handling)
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_yahooquery_data(tickers, start, end):
    """Return dict: ticker -> DataFrame (indexed by datetime) for the given date window (inclusive)."""
    ticker_obj = Ticker(tickers)
    # yahooquery.history end param is exclusive; include +1 day to ensure last-day data
    df = ticker_obj.history(start=start.strftime("%Y-%m-%d"),
                            end=(end + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
                            interval="1d")
    if df is None or df.empty:
        return {t: pd.DataFrame() for t in tickers}

    dfs = {}
    if isinstance(df.index, pd.MultiIndex):
        for t in tickers:
            try:
                df_t = df.loc[t].copy()
            except Exception:
                dfs[t] = pd.DataFrame()
                continue

            # Convert index robustly, remove tz, drop invalid rows
            try:
                if not pd.api.types.is_datetime64_any_dtype(df_t.index):
                    dt_index = pd.to_datetime(df_t.index, errors="coerce")
                else:
                    dt_index = df_t.index

                # remove timezone if present
                if getattr(dt_index, "tz", None) is not None:
                    dt_index = dt_index.tz_convert(None) if hasattr(dt_index, "tz_convert") else dt_index.tz_localize(None)

                df_t = df_t.assign(_dt_index=dt_index)
                df_t = df_t.dropna(subset=["_dt_index"])
                df_t.index = df_t["_dt_index"]
                df_t = df_t.drop(columns=["_dt_index"])
            except Exception as e:
                st.warning(f"Date parsing issue for {t}: {e}")
                dfs[t] = pd.DataFrame()
                continue

            # filter to requested window
            df_t = df_t.loc[(df_t.index >= start) & (df_t.index <= end)]
            dfs[t] = df_t
    else:
        # single-ticker response
        try:
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                dt_index = pd.to_datetime(df.index, errors="coerce")
            else:
                dt_index = df.index
            if getattr(dt_index, "tz", None) is not None:
                dt_index = dt_index.tz_convert(None) if hasattr(dt_index, "tz_convert") else dt_index.tz_localize(None)
            df = df.assign(_dt_index=dt_index)
            df = df.dropna(subset=["_dt_index"])
            df.index = df["_dt_index"]
            df = df.drop(columns=["_dt_index"])
            df = df.loc[(df.index >= start) & (df.index <= end)]
        except Exception as e:
            st.warning(f"Date parsing issue for single ticker response: {e}")
            df = pd.DataFrame()
        dfs[tickers[0]] = df

    return dfs

# ----------------------------
# Helper: fetch FRED series (cached)
# ----------------------------
def fetch_fred_series(series_id, start=None, end=None, fred_client=None):
    fred_client = fred_client or Fred(api_key=FRED_API_KEY)
    cache_path = os.path.join(CACHE_DIR, f"{series_id}.csv")
    if os.path.exists(cache_path):
        try:
            s = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")["value"]
            if start:
                s = s[s.index >= pd.to_datetime(start)]
            if end:
                s = s[s.index <= pd.to_datetime(end)]
            return s
        except Exception:
            pass
    # fallback to live fetch
    s = fred_client.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s = s.rename(series_id)
    s.to_csv(cache_path, header=["value"])
    if start:
        s = s[s.index >= pd.to_datetime(start)]
    if end:
        s = s[s.index <= pd.to_datetime(end)]
    return s

# ----------------------------
# Compute combined rate series
# ----------------------------
def compute_combined_rate_series(zq_df, sofr_series, sofr_window_days=30):
    """Return DataFrame with columns: sofr, sofr_avg, implied_rate, combined_rate (daily)."""
    # require 'close' column on zq_df
    if zq_df is None or zq_df.empty or "close" not in zq_df.columns:
        return pd.DataFrame()

    # ZQ implied rate = 100 - close price (daily)
    zq_daily = zq_df["close"].resample("D").last().dropna()
    implied_rate = 100.0 - zq_daily
    # SOFR: fill to daily and compute rolling avg
    sofr = sofr_series.copy()
    sofr = sofr.resample("D").ffill()
    df = pd.DataFrame({"implied_rate": implied_rate}).join(sofr.rename("sofr"), how="inner")
    df["sofr_avg"] = df["sofr"].rolling(window=sofr_window_days, min_periods=1).mean()
    df["combined_rate"] = (df["implied_rate"] + df["sofr_avg"]) / 2.0
    return df.dropna()

# ----------------------------
# Entries using 35th percentile (historical)
# ----------------------------
def find_entries_by_percentile(combined_df, percentile=35.0):
    """Return percentile_value and DataFrame of entry dates where combined_rate < percentile_value."""
    if combined_df is None or combined_df.empty:
        return None, pd.DataFrame()
    pct_val = np.percentile(combined_df["combined_rate"].values, percentile)
    entries = combined_df[combined_df["combined_rate"] < pct_val].copy()
    return float(pct_val), entries

# ----------------------------
# Compute returns for entries (gold & silver)
# ----------------------------
def compute_entry_returns(entries_df, gold_df, silver_df, holding_days=14):
    results = []
    for entry_date in entries_df.index:
        buy = entry_date
        sell = buy + pd.Timedelta(days=holding_days)
        g_slice = gold_df.loc[buy:sell]
        s_slice = silver_df.loc[buy:sell]
        if g_slice.empty or s_slice.empty:
            continue
        try:
            g_entry = g_slice["close"].iloc[0]; g_exit = g_slice["close"].iloc[-1]
            s_entry = s_slice["close"].iloc[0]; s_exit = s_slice["close"].iloc[-1]
        except Exception:
            continue
        results.append({
            "entry_date": buy,
            "sell_date": sell,
            "ret_gold": (g_exit / g_entry) - 1.0,
            "ret_silver": (s_exit / s_entry) - 1.0,
            "entry_gold": g_entry, "exit_gold": g_exit,
            "entry_silver": s_entry, "exit_silver": s_exit
        })
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).set_index("entry_date")
    df["avg_ret"] = df[["ret_gold", "ret_silver"]].mean(axis=1)
    return df

# ----------------------------
# Streamlit UI: sidebar controls
# ----------------------------
st.sidebar.header("Settings")
fred_key_input = st.sidebar.text_input("FRED API Key (or set FRED_API_KEY env var)", value=FRED_API_KEY)
if fred_key_input:
    FRED_API_KEY = fred_key_input.strip()

sofr_window = st.sidebar.number_input("SOFR rolling window (days)", value=30, min_value=1, max_value=180)
holding_days = st.sidebar.number_input("Holding days after entry", value=14, min_value=1, max_value=90)
refresh = st.sidebar.button("Refresh Data (clear cache)")

st.markdown(f"**Data Range (fixed):** {START_DATE.date()} → {END_DATE.date()}")
st.info("This dashboard computes the 35th percentile on the full historical combined-rate series and uses it as the entry filter.")

# require FRED key
if not FRED_API_KEY or len(FRED_API_KEY) != 32:
    st.error("Please provide a valid 32-character FRED API key (sidebar or env var).")
    st.stop()

# ----------------------------
# Load data (with caching & optional refresh)
# ----------------------------
if refresh:
    # clear cached data by bumping cache keys: easiest is to remove cache files
    for f in os.listdir(CACHE_DIR):
        try:
            os.remove(os.path.join(CACHE_DIR, f))
        except Exception:
            pass
    st.experimental_rerun()

# fetch prices
tickers = ["ZQ=F", "GC=F", "SI=F"]
dfs = fetch_yahooquery_data(tickers, START_DATE, END_DATE + dt.timedelta(days=30))  # gold/silver extra days for forward returns
zq_df = dfs.get("ZQ=F", pd.DataFrame())
gold_df = dfs.get("GC=F", pd.DataFrame())
silver_df = dfs.get("SI=F", pd.DataFrame())

# debug columns
st.write("ZQ=F columns:", list(zq_df.columns) if not zq_df.empty else "empty")
st.write("GC=F columns:", list(gold_df.columns) if not gold_df.empty else "empty")
st.write("SI=F columns:", list(silver_df.columns) if not silver_df.empty else "empty")

# fetch FRED series
fred_client = Fred(api_key=FRED_API_KEY)
try:
    sofr_series = fetch_fred_series("SOFR", start=START_DATE, end=END_DATE, fred_client=fred_client)
except Exception as e:
    st.error(f"Error fetching SOFR from FRED: {e}")
    sofr_series = pd.Series(dtype=float)

# ----------------------------
# Compute combined series & percentile filter
# ----------------------------
combined_df = compute_combined_rate_series(zq_df, sofr_series, sofr_window_days=sofr_window)
if combined_df.empty:
    st.error("Combined rate series is empty — check ZQ=F and SOFR data.")
    st.stop()

pct35_value, entries_df = find_entries_by_percentile(combined_df, percentile=35.0)
st.write(f"Historic 35th percentile of combined_rate = {pct35_value:.6f}")

# ----------------------------
# Compute returns for entries
# ----------------------------
entries_returns = compute_entry_returns(entries_df, gold_df, silver_df, holding_days=holding_days)
if not entries_returns.empty:
    avg_entry_return = entries_returns["avg_ret"].mean()
    compounded = (1 + entries_returns["avg_ret"]).prod() - 1
else:
    avg_entry_return = np.nan
    compounded = np.nan

# ----------------------------
# Dashboard panels
# ----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Combined Rate (implied ZQ vs SOFR avg)")
    st.line_chart(combined_df[["implied_rate", "sofr_avg", "combined_rate"]])

    st.markdown(f"**Historical 35th percentile threshold:** {pct35_value:.6f}")
    st.write("Entries are days where combined_rate < 35th percentile (filtering logic).")
    st.dataframe(entries_df[["combined_rate"]].assign(threshold=pct35_value).head(10))

with col2:
    st.subheader("Entry Statistics")
    st.metric("Number of entries", len(entries_df))
    st.metric("Avg entry return (gold+silver avg)", f"{avg_entry_return:.4%}" if not np.isnan(avg_entry_return) else "N/A")
    st.metric("Compounded return across entries", f"{compounded:.4%}" if not np.isnan(compounded) else "N/A")

# Entry navigator and charting
if entries_df.empty:
    st.info("No entries found using the 35th-percentile filter in the selected historical window.")
else:
    if "entry_idx" not in st.session_state:
        st.session_state["entry_idx"] = 0

    entry_dates = list(entries_df.index)
    coln1, coln2, coln3 = st.columns([1, 1, 2])
    with coln1:
        if st.button("Prev"):
            st.session_state["entry_idx"] = max(0, st.session_state["entry_idx"] - 1)
    with coln2:
        if st.button("Next"):
            st.session_state["entry_idx"] = min(len(entry_dates) - 1, st.session_state["entry_idx"] + 1)
    with coln3:
        st.number_input("Go to entry index", min_value=0, max_value=len(entry_dates) - 1, key="entry_idx_input",
                        value=st.session_state["entry_idx"], on_change=lambda: st.session_state.update({"entry_idx": st.session_state["entry_idx_input"]}))

    idx = st.session_state["entry_idx"]
    sel_date = entry_dates[idx]
    st.markdown(f"### Entry {idx+1}/{len(entry_dates)} — {sel_date.date()} (combined_rate={entries_df.loc[sel_date,'combined_rate']:.6f})")

    buy = sel_date
    sell = buy + pd.Timedelta(days=holding_days)
    g_slice = gold_df.loc[buy:sell]
    s_slice = silver_df.loc[buy:sell]

    st.subheader(f"Gold & Silver from {buy.date()} to {sell.date()}")
    if g_slice.empty or s_slice.empty:
        st.warning("No gold/silver price data available for this entry's forward window.")
    else:
        st.line_chart(g_slice["close"], use_container_width=True)
        st.line_chart(s_slice["close"], use_container_width=True)

# CSV downloads
st.markdown("---")
st.download_button("Download combined_df CSV", data=combined_df.to_csv().encode("utf-8"), file_name="combined_rates.csv", mime="text/csv")
if not entries_returns.empty:
    st.download_button("Download entries returns CSV", data=entries_returns.to_csv().encode("utf-8"), file_name="entries_returns.csv", mime="text/csv")
else:
    st.info("No entries/returns CSV to download (no entries found).")
