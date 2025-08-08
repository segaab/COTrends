import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from fredapi import Fred
from yahooquery import Ticker
import os

# --- Config & Cache directory ---
FRED_API_KEY="91bb2c5920fb8f843abdbbfdfcab5345"
CACHE_DIR = "./data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- Date Range Setup ---
now_utc = dt.datetime.utcnow()
et_offset = dt.timedelta(hours=4)  # approximate ET offset from UTC
now_et = now_utc - et_offset

yesterday_et = dt.datetime(year=now_et.year, month=now_et.month, day=now_et.day) - dt.timedelta(days=1)
yesterday_23 = yesterday_et.replace(hour=23, minute=0, second=0, microsecond=0)

start_date = yesterday_23 - dt.timedelta(days=365)
end_date = yesterday_23

# --- Fetch yahooquery data ---
@st.cache_data(ttl=3600)
def fetch_yahooquery_data(tickers, start, end):
    ticker_obj = Ticker(tickers)
    df = ticker_obj.history(
        start=start.strftime('%Y-%m-%d'),
        end=(end + dt.timedelta(days=1)).strftime('%Y-%m-%d'),
        interval='1d'
    )

    if df.empty:
        return {t: pd.DataFrame() for t in tickers}

    dfs = {}
    if isinstance(df.index, pd.MultiIndex):
        for t in tickers:
            try:
                df_t = df.loc[t].copy()
            except KeyError:
                dfs[t] = pd.DataFrame()
                continue

            try:
                if not pd.api.types.is_datetime64_any_dtype(df_t.index):
                    dt_index = pd.to_datetime(df_t.index, errors='coerce')
                else:
                    dt_index = df_t.index

                if dt_index.tz is not None:
                    dt_index = dt_index.tz_localize(None)

                df_t = df_t.assign(_dt_index=dt_index)
                df_t = df_t.dropna(subset=['_dt_index'])
                df_t.index = df_t['_dt_index']
                df_t = df_t.drop(columns=['_dt_index'])
            except Exception as e:
                st.warning(f"Date parsing issue for ticker {t}: {e}")
                dfs[t] = pd.DataFrame()
                continue

            df_t = df_t.loc[(df_t.index >= start) & (df_t.index <= end)]
            dfs[t] = df_t
    else:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                dt_index = pd.to_datetime(df.index, errors='coerce')
            else:
                dt_index = df.index

            if dt_index.tz is not None:
                dt_index = dt_index.tz_localize(None)

            df = df.assign(_dt_index=dt_index)
            df = df.dropna(subset=['_dt_index'])
            df.index = df['_dt_index']
            df = df.drop(columns=['_dt_index'])

            df = df.loc[(df.index >= start) & (df.index <= end)]
        except Exception as e:
            st.warning(f"Date parsing issue for single ticker: {e}")
            df = pd.DataFrame()
        dfs[tickers[0]] = df

    return dfs

# --- Fetch FRED series with caching ---
def fetch_fred_series(series_id, start=None, end=None, fred_client=None):
    fred_client = fred_client or Fred(api_key=FRED_API_KEY)
    cache_path = f"{CACHE_DIR}/{series_id}.csv"
    if os.path.exists(cache_path):
        try:
            s = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')['value']
            if start:
                s = s[s.index >= pd.to_datetime(start)]
            if end:
                s = s[s.index <= pd.to_datetime(end)]
            return s
        except Exception:
            pass

    s = fred_client.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s = s.rename(series_id)
    s.to_csv(cache_path, header=['value'])
    if start:
        s = s[s.index >= pd.to_datetime(start)]
    if end:
        s = s[s.index <= pd.to_datetime(end)]
    return s

# --- Combine Fed Funds futures implied rate and SOFR rolling average ---
def compute_combined_rates(zq_df, sofr_series, window_days=30):
    if 'close' not in zq_df.columns:
        st.error("ZQ=F data missing 'close' column.")
        return pd.DataFrame()
    zq_daily = zq_df['close'].resample('D').last().dropna()
    implied_rate = 100.0 - zq_daily

    sofr = sofr_series.copy()
    sofr = sofr.resample('D').ffill()

    df = pd.DataFrame({'implied_rate': implied_rate}).join(sofr.rename('sofr'), how='inner')
    df['sofr_avg'] = df['sofr'].rolling(window=window_days, min_periods=1).mean()
    df['combined_rate'] = (df['implied_rate'] + df['sofr_avg']) / 2.0
    return df.dropna()

# --- Compute entry points and returns ---
def compute_entries_and_returns(combined_df, gold_df, silver_df, holding_days=14):
    if combined_df.empty:
        return pd.DataFrame(), {}
    q25 = combined_df['combined_rate'].quantile(0.25)
    entries = combined_df[combined_df['combined_rate'] < q25].copy()
    entries['entry_date'] = entries.index

    results = []
    for dt_entry in entries['entry_date']:
        buy_date = dt_entry
        sell_date = buy_date + pd.Timedelta(days=holding_days)

        slice_gold = gold_df.loc[buy_date:sell_date]
        slice_silver = silver_df.loc[buy_date:sell_date]

        if slice_gold.empty or slice_silver.empty:
            continue
        try:
            entry_gold = slice_gold['close'].iloc[0]
            exit_gold = slice_gold['close'].iloc[-1]
            entry_silver = slice_silver['close'].iloc[0]
            exit_silver = slice_silver['close'].iloc[-1]
        except Exception:
            continue
        ret_gold = (exit_gold / entry_gold) - 1.0
        ret_silver = (exit_silver / entry_silver) - 1.0
        results.append({
            'entry_date': buy_date,
            'sell_date': sell_date,
            'entry_gold': entry_gold,
            'exit_gold': exit_gold,
            'ret_gold': ret_gold,
            'entry_silver': entry_silver,
            'exit_silver': exit_silver,
            'ret_silver': ret_silver
        })
    results_df = pd.DataFrame(results).set_index('entry_date') if results else pd.DataFrame()

    if not results_df.empty:
        results_df['avg_ret'] = results_df[['ret_gold', 'ret_silver']].mean(axis=1)
        avg_return = results_df['avg_ret'].mean()
        compounded = (1 + results_df['avg_ret']).prod() - 1
        summary = {
            'n_entries': len(results_df),
            'average_return_per_entry': avg_return,
            'compounded_return_all_entries': compounded
        }
    else:
        summary = {'n_entries': 0, 'average_return_per_entry': np.nan, 'compounded_return_all_entries': np.nan}

    return results_df, summary

# --- Streamlit UI ---
st.set_page_config(page_title="Fed Funds ZQ + Metals Dashboard", layout='wide')
st.title("Fed Funds Futures (ZQ=F) + SOFR + Gold & Silver Dashboard")

with st.sidebar:
    st.header('Data & Parameters')
    fred_key_input = st.text_input('FRED API Key (or set FRED_API_KEY env var)', value=FRED_API_KEY)
    if fred_key_input:
        FRED_API_KEY = fred_key_input.strip()
    sofr_window = st.number_input('SOFR rolling window (days)', min_value=1, max_value=90, value=30)
    holding_days = st.number_input('Holding days after entry', min_value=1, max_value=60, value=14)
    refresh = st.button('Refresh Data')

st.markdown(f"### Data Range: {start_date.date()} to {end_date.date()}")
st.markdown('**Fetching data — this may take a few seconds (uses yahooquery & FRED)**')

if not FRED_API_KEY or len(FRED_API_KEY) != 32:
    st.error("Please provide a valid 32-character FRED API Key!")
    st.stop()

try:
    fred_client = Fred(api_key=FRED_API_KEY)
except Exception as e:
    st.error(f'FRED client init error: {e}')
    st.stop()

zq_cache_path = f"{CACHE_DIR}/zq_yq_history_{start_date.date()}_{end_date.date()}.csv"
gold_cache_path = f"{CACHE_DIR}/gold_yq_history_{start_date.date()}_{end_date.date()}.csv"
silver_cache_path = f"{CACHE_DIR}/silver_yq_history_{start_date.date()}_{end_date.date()}.csv"

use_cache = (
    os.path.exists(zq_cache_path) and
    os.path.exists(gold_cache_path) and
    os.path.exists(silver_cache_path) and
    not refresh
)

if use_cache:
    zq_df = pd.read_csv(zq_cache_path, index_col=0, parse_dates=True)
    gold_df = pd.read_csv(gold_cache_path, index_col=0, parse_dates=True)
    silver_df = pd.read_csv(silver_cache_path, index_col=0, parse_dates=True)
else:
    dfs = fetch_yahooquery_data(['ZQ=F', 'GC=F', 'SI=F'], start_date, end_date + dt.timedelta(days=30))
    zq_df = dfs.get('ZQ=F', pd.DataFrame())
    gold_df = dfs.get('GC=F', pd.DataFrame())
    silver_df = dfs.get('SI=F', pd.DataFrame())

    zq_df.to_csv(zq_cache_path)
    gold_df.to_csv(gold_cache_path)
    silver_df.to_csv(silver_cache_path)

# Debug columns
st.write("ZQ=F columns:", list(zq_df.columns))
st.write("Gold (GC=F) columns:", list(gold_df.columns))
st.write("Silver (SI=F) columns:", list(silver_df.columns))

try:
    sofr_series = fetch_fred_series('SOFR', start=start_date, end=end_date, fred_client=fred_client)
except Exception as e:
    st.error(f"Error fetching SOFR from FRED: {e}")
    sofr_series = pd.Series(dtype=float)

combined_df = compute_combined_rates(zq_df, sofr_series, window_days=sofr_window)

if combined_df.empty:
    st.error("Combined rate data is empty. Check your data sources.")
    st.stop()

st.subheader('Combined Rate Overview')
st.write('Combined Rate is average of ZQ implied rate and SOFR rolling average')
st.line_chart(combined_df[['implied_rate', 'sofr_avg', 'combined_rate']])

entries_df, summary = compute_entries_and_returns(combined_df, gold_df, silver_df, holding_days=holding_days)

st.subheader('Entry Points (Combined Rate < 25th percentile)')
st.write(f"Number of entries: {summary['n_entries']}")
if not np.isnan(summary['average_return_per_entry']):
    st.write(f"Average return per entry (gold+silver avg): {summary['average_return_per_entry']:.4f}")
if not np.isnan(summary['compounded_return_all_entries']):
    st.write(f"Compounded return across entries: {summary['compounded_return_all_entries']:.4f}")

if entries_df.empty:
    st.info('No entries found in the selected date range.')
else:
    entry_dates = list(entries_df.index)
    if 'entry_idx' not in st.session_state:
        st.session_state['entry_idx'] = 0

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button('Prev'):
            st.session_state['entry_idx'] = max(0, st.session_state['entry_idx'] - 1)
    with col2:
        if st.button('Next'):
            st.session_state['entry_idx'] = min(len(entry_dates) - 1, st.session_state['entry_idx'] + 1)
    with col3:
        st.session_state['entry_idx'] = st.number_input('Go to entry index', min_value=0, max_value=len(entry_dates) - 1,
                                                      value=st.session_state['entry_idx'])

    idx = st.session_state['entry_idx']
    sel_date = entry_dates[idx]
    st.markdown(f"### Entry {idx + 1}/{len(entry_dates)} — {sel_date.date()}")

    st.write(combined_df.loc[sel_date])

    buy = sel_date
    sell = buy + pd.Timedelta(days=holding_days)

    slice_gold = gold_df.loc[buy:sell]
    slice_silver = silver_df.loc[buy:sell]

    st.subheader(f"Gold and Silver Prices from {buy.date()} to {sell.date()}")

    if slice_gold.empty or slice_silver.empty:
        st.warning("No price data available for gold or silver in this date range.")
    else:
        st.line_chart(slice_gold['close'], use_container_width=True)
        st.line_chart(slice_silver['close'], use_container_width=True)

    # Show CSV download for entries summary
    csv = entries_df.to_csv().encode('utf-8')
    st.download_button('Download Entries CSV', data=csv, file_name='entries.csv', mime='text/csv')
