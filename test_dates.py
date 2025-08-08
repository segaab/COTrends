"""
Streamlit dashboard + scraping script for Fed Funds futures (ZQ=F), SOFR, and Gold/Silver prices.

Features:
- Fixed date range: yesterday 23:00 ET - 365 days back
- Fetch historical Fed Funds futures (ZQ=F) via Yahoo Finance unofficial endpoint
- Fetch SOFR and FEDFUNDS from FRED via fredapi
- Fetch Gold/Silver (GC=F, SI=F) via yfinance
- Compute averaged SOFR (30-day rolling) and Combined Rate = average(FedFundsFutures_implied, SOFR_avg)
- Find 25th percentile of Combined Rate and mark entry dates where Combined Rate < 25th percentile
- Display 2-week forward charts for Gold & Silver for each entry point; interactive "Next" button cycles through entries
- Calculate overall returns (per-entry returns, average return, compounded return) for taking entries at 25th percentile

Notes:
- Uses Yahoo's undocumented API for ZQ=F (grey area). Use responsibly.
- Install dependencies: pip install streamlit yfinance fredapi pandas requests
- Run: streamlit run fedfunds_zq_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
from fredapi import Fred
import yfinance as yf
import os

# ----------------------------
# Config / API Keys
# ----------------------------
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')  # set this in your env
CACHE_DIR = "./data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# ----------------------------
# Fixed date range: yesterday 23:00 ET minus 365 days
# ----------------------------
now_utc = dt.datetime.utcnow()
# Approximate ET = UTC-4 for daylight savings approx (better if you want exact ET conversion)
et_offset = dt.timedelta(hours=4)
now_et = now_utc - et_offset

yesterday_et = dt.datetime(year=now_et.year, month=now_et.month, day=now_et.day) - dt.timedelta(days=1)
yesterday_23 = yesterday_et.replace(hour=23, minute=0, second=0, microsecond=0)

start_date = yesterday_23 - dt.timedelta(days=365)
end_date = yesterday_23

# ----------------------------
# Helper: Fetch Yahoo ZQ=F
# ----------------------------
def fetch_yahoo_zq_history(range_='1y', interval='1d'):
    """
    Fetch historical ZQ=F data from Yahoo (undocumented endpoint).
    Returns a DataFrame with datetime index and OHLCV and implied_rate column.
    """
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/ZQ=F'
    params = {
        'range': range_,  # '1y' or '365d' is accepted, but '1y' works fine
        'interval': interval
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    result = data.get('chart', {}).get('result')
    if not result:
        raise RuntimeError('Yahoo response missing chart/result')
    res = result[0]
    timestamps = res.get('timestamp', [])
    quote = res.get('indicators', {}).get('quote', [{}])[0]
    if not timestamps:
        return pd.DataFrame()
    df = pd.DataFrame({
        'datetime': [dt.datetime.fromtimestamp(int(ts)) for ts in timestamps],
        'open': quote.get('open'),
        'high': quote.get('high'),
        'low': quote.get('low'),
        'close': quote.get('close'),
        'volume': quote.get('volume')
    })
    df = df.set_index('datetime')
    # Convert price to implied rate: implied_rate = 100 - price
    df['implied_rate'] = 100.0 - df['close']
    return df.loc[(df.index >= start_date) & (df.index <= end_date)]

# ----------------------------
# Helper: Fetch FRED series
# ----------------------------
def fetch_fred_series(series_id, start=None, end=None, fred_client=None):
    """
    Fetch series from FRED and return a pandas Series.
    Cache to CSV to avoid repeated hits.
    """
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

# ----------------------------
# Helper: Fetch Gold & Silver via yfinance
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_metals(start, end):
    # Gold futures GC=F and Silver futures SI=F; if you prefer ETFs use GLD/SLV
    tickers = ["GC=F", "SI=F"]
    data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker')
    gold = data['GC=F']['Adj Close'].rename('gold') if ('GC=F' in data and 'Adj Close' in data['GC=F']) else pd.Series(dtype=float)
    silver = data['SI=F']['Adj Close'].rename('silver') if ('SI=F' in data and 'Adj Close' in data['SI=F']) else pd.Series(dtype=float)
    df = pd.concat([gold, silver], axis=1)
    return df

# ----------------------------
# Compute combined rate and entry signals
# ----------------------------
def compute_combined_rates(zq_df, sofr_series, window_days=30):
    """
    Align ZQ daily implied rates with SOFR series and compute averaged SOFR (rolling window)
    and combined rate = mean([zq_implied, sofr_avg]).
    Returns DataFrame with columns: implied_rate, sofr, sofr_avg, combined_rate
    """
    zq_daily = zq_df['implied_rate'].resample('D').last().dropna()
    sofr = sofr_series.copy()
    sofr = sofr.resample('D').ffill()
    df = pd.DataFrame({'implied_rate': zq_daily}).join(sofr.rename('sofr'), how='inner')
    df['sofr_avg'] = df['sofr'].rolling(window=window_days, min_periods=1).mean()
    df['combined_rate'] = (df['implied_rate'] + df['sofr_avg']) / 2.0
    return df.dropna()

# ----------------------------
# Entry points and returns
# ----------------------------
def compute_entries_and_returns(combined_df, metals_df, holding_days=14):
    """
    Find entry dates where combined_rate < 25th percentile.
    For each entry, calculate forward returns for gold and silver over holding_days.
    Returns a DataFrame of entries and a summary of returns.
    """
    q25 = combined_df['combined_rate'].quantile(0.25)
    entries = combined_df[combined_df['combined_rate'] < q25].copy()
    entries['entry_date'] = entries.index

    results = []
    for dt_entry in entries['entry_date']:
        buy_date = dt_entry
        sell_date = buy_date + pd.Timedelta(days=holding_days)
        slice_metals = metals_df.loc[buy_date:sell_date]
        if slice_metals.empty:
            continue
        try:
            entry_gold = slice_metals['gold'].iloc[0]
            entry_silver = slice_metals['silver'].iloc[0]
            exit_gold = slice_metals['gold'].iloc[-1]
            exit_silver = slice_metals['silver'].iloc[-1]
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

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fed Funds ZQ Dashboard", layout='wide')
st.title("Fed Funds Futures (ZQ) + SOFR + Metals Dashboard")

with st.sidebar:
    st.header('Data & Parameters')
    fred_key_input = st.text_input('FRED API Key (or set FRED_API_KEY env var)', value=FRED_API_KEY)
    if fred_key_input:
        FRED_API_KEY = fred_key_input
    sofr_window = st.number_input('SOFR rolling window (days)', min_value=1, max_value=90, value=30)
    holding_days = st.number_input('Holding days after entry', min_value=1, max_value=60, value=14)
    refresh = st.button('Refresh Data')

st.markdown(f"### Data Range: {start_date.date()} to {end_date.date()}")
st.markdown('**Fetching data — this may take a few seconds (uses Yahoo & FRED)**')

try:
    fred_client = Fred(api_key=FRED_API_KEY)
except Exception as e:
    st.error(f'FRED client init error: {e}')
    fred_client = None

zq_cache_path = f"{CACHE_DIR}/zq_history_{start_date.date()}_{end_date.date()}.csv"
use_cache = os.path.exists(zq_cache_path) and not refresh
if use_cache:
    zq_df = pd.read_csv(zq_cache_path, parse_dates=['datetime'], index_col='datetime')
else:
    zq_df = fetch_yahoo_zq_history(range_='1y', interval='1d')
    zq_df = zq_df.loc[start_date:end_date]
    zq_df.to_csv(zq_cache_path)

sofr_series = fetch_fred_series('SOFR', start=start_date, end=end_date, fred_client=fred_client)
fedfunds_series = fetch_fred_series('FEDFUNDS', start=start_date, end=end_date, fred_client=fred_client)

metals_df = fetch_metals(start=start_date, end=end_date + dt.timedelta(days=30))  # extra for forward returns

combined_df = compute_combined_rates(zq_df, sofr_series, window_days=sofr_window)

st.subheader('Combined Rate Overview')
st.write('Combined Rate is (ZQ implied rate + SOFR rolling average) / 2')
st.line_chart(combined_df[['implied_rate', 'sofr_avg', 'combined_rate']])

entries_df, summary = compute_entries_and_returns(combined_df, metals_df, holding_days=holding_days)

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
    slice_metals = metals_df.loc[buy:sell]

    st.write(f"Showing gold & silver from {buy.date()} to {sell.date()}")
    st.line_chart(slice_metals)

    row = entries_df.loc[sel_date]
    st.write('Return summary for this entry:')
    st.write(pd.DataFrame({
        'metric': ['entry_gold', 'exit_gold', 'ret_gold', 'entry_silver', 'exit_silver', 'ret_silver'],
        'value': [row['entry_gold'], row['exit_gold'], row['ret_gold'], row['entry_silver'], row['exit_silver'], row['ret_silver']]
    }))

st.markdown('---')
st.caption('This dashboard uses Yahoo Finance unofficial endpoints for ZQ=F and FRED for SOFR/FEDFUNDS. Use for prototyping; for production get licensed real-time feeds.')

if not combined_df.empty:
    if st.button('Download combined rates CSV'):
        csv = combined_df.to_csv().encode('utf-8')
        st.download_button('Download Combined CSV', data=csv, file_name='combined_rates.csv', mime='text/csv')

if not entries_df.empty:
    if st.button('Download entries CSV'):
        csv = entries_df.to_csv().encode('utf-8')
        st.download_button('Download Entries CSV', data=csv, file_name='entries.csv', mime='text/csv')
