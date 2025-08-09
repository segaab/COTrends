# app.py
import os
import random
import math
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from statsmodels.tsa.statespace.sarimax import SARIMAX
from supabase import create_client
import altair as alt
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Configuration / Secrets
# -------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # service role if you need writes
SOFR_SERIES = "SOFR"
DGS10_SERIES = "DGS10"
FED_FUNDS_FUTURES_TICKER = "ZQ=F"   # Yahoo ticker for Fed funds futures (daily)
TEN_YR_TICKER = "ZN=F"              # 10y futures ticker (used if you want futures prices); we'll use DGS10 yields for current yield
ROLLING_VOL_WINDOW = 21             # days for realized vol proxy
FORECAST_DAYS = 7                   # one-week ahead

# Supabase client (optional)
supa = None
if SUPABASE_URL and SUPABASE_KEY:
    supa = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# Utilities / Data fetch
# -------------------------
def latest_friday_end():
    """Return latest Friday (date) up to today (UTC)."""
    today = datetime.utcnow().date()
    # Friday is weekday=4
    days_since_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_since_friday)
    # interpret as end of that day
    return datetime.combine(last_friday, datetime.max.time())

def fetch_sofr(start_date, end_date):
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is required to fetch SOFR.")
    fred = Fred(api_key=FRED_API_KEY)
    s = fred.get_series(SOFR_SERIES, observation_start=start_date.strftime("%Y-%m-%d"), observation_end=end_date.strftime("%Y-%m-%d"))
    df = s.rename("sofr").to_frame().reset_index().rename(columns={"index": "date"})
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df

def fetch_fedfunds_futures(start_date, end_date):
    # Using yfinance daily Close on ZQ=F
    df = yf.download(FED_FUNDS_FUTURES_TICKER, start=start_date.strftime("%Y-%m-%d"), end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        raise RuntimeError("No Fed Funds futures data returned.")
    df = df[['Close']].rename(columns={"Close": "fedfund_future"})
    df = df.reset_index().rename(columns={"Date": "date"})
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df

def fetch_dgs10_latest_and_history(start_date, end_date):
    # DGS10 daily yields from FRED
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is required to fetch DGS10.")
    fred = Fred(api_key=FRED_API_KEY)
    s = fred.get_series(DGS10_SERIES, observation_start=start_date.strftime("%Y-%m-%d"), observation_end=end_date.strftime("%Y-%m-%d"))
    df = s.rename("dgs10").to_frame().reset_index().rename(columns={"index": "date"})
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    return df

# -------------------------
# Processing & vol construction
# -------------------------
def combined_rate_daily(sofr_df, fed_df):
    # inner join on date
    df = pd.merge(sofr_df, fed_df, on='date', how='inner')
    df['combined_rate'] = df[['sofr','fedfund_future']].mean(axis=1)
    return df[['date','sofr','fedfund_future','combined_rate']].sort_values('date')

def realized_vol_proxy(ten_yield_df, window_days=ROLLING_VOL_WINDOW):
    # compute pct change of daily yields, rolling std dev, annualize
    df = ten_yield_df.copy().sort_values('date')
    df['ret'] = df['dgs10'].pct_change()
    df['realized_vol'] = df['ret'].rolling(window=window_days).std() * np.sqrt(252)  # annualized daily vol
    return df[['date','dgs10','realized_vol']]

def initial_vol_guess_from_yield(current_yield):
    # Determine a range by adding/subtracting 5% (i.e., current_yield * (1 +/- 0.05))
    lower = current_yield * 0.95
    upper = current_yield * 1.05
    # choose random number in that range
    return random.uniform(lower, upper)

def scale_vol_series_to_initial(vol_series, initial_guess):
    """
    vol_series: pd.Series (realized vol values)
    scale entire series so last value == initial_guess (proportional scaling)
    If last value is zero, just fill with initial_guess.
    """
    last = vol_series.iloc[-1]
    if pd.isna(last) or last == 0:
        scaled = pd.Series([initial_guess]*len(vol_series), index=vol_series.index)
    else:
        scale_factor = initial_guess / float(last)
        scaled = vol_series * scale_factor
    return scaled

# -------------------------
# Modeling & Forecast
# -------------------------
def train_sarimax(endog_series, exog_series, order=(1,1,1), steps=FORECAST_DAYS):
    # endog_series/index: pd.Series with DateTimeIndex daily
    # exog_series: pd.Series or DataFrame aligned with endog
    endog = endog_series.copy().astype(float).dropna()
    exog = exog_series.copy().astype(float).reindex(endog.index).fillna(method='ffill').fillna(0)
    model = SARIMAX(endog, exog=exog, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    # Build future exog as last observed repeated for `steps` days
    last_exog = exog.iloc[-1].values.reshape(1, -1)
    future_exog = np.repeat(last_exog, steps, axis=0)
    pred = res.get_forecast(steps=steps, exog=future_exog)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)
    return res, mean, ci

# -------------------------
# Supabase push
# -------------------------
def push_to_supabase(supabase_client, forecast_generated_at_iso, input_rows, forecast_rows):
    if supabase_client is None:
        st.warning("Supabase not configured; skipping push.")
        return
    # insert input rows (is_forecast False)
    inputs = []
    for r in input_rows:
        inputs.append({
            "forecast_generated_at": forecast_generated_at_iso,
            "forecast_time": r['date'].isoformat(),
            "sofr": float(r['sofr']),
            "fedfund_future": float(r['fedfund_future']),
            "ten_close": float(r.get('ten_close', None)) if r.get('ten_close') is not None else None,
            "combined_rate": float(r['combined_rate']),
            "treas_implied_vol": float(r.get('treas_implied_vol', None)) if r.get('treas_implied_vol') is not None else None,
            "is_forecast": False,
            "forecast_lower_95": None,
            "forecast_upper_95": None
        })
    # insert forecast rows (is_forecast True)
    forecasts = []
    for r in forecast_rows:
        forecasts.append({
            "forecast_generated_at": forecast_generated_at_iso,
            "forecast_time": r['forecast_time'].isoformat() if isinstance(r['forecast_time'], (datetime, pd.Timestamp)) else r['forecast_time'],
            "sofr": None,
            "fedfund_future": None,
            "ten_close": None,
            "combined_rate": float(r['combined_rate']),
            "treas_implied_vol": None,
            "is_forecast": True,
            "forecast_lower_95": float(r.get('lower_95')) if r.get('lower_95') is not None else None,
            "forecast_upper_95": float(r.get('upper_95')) if r.get('upper_95') is not None else None
        })
    # push batches (no upsert to keep history)
    try:
        supabase_client.table("hourly_forecasts").insert(inputs).execute()
        supabase_client.table("hourly_forecasts").insert(forecasts).execute()
        st.success("Pushed inputs + forecasts to Supabase.")
    except Exception as e:
        st.error(f"Failed to push to Supabase: {e}")

# -------------------------
# Streamlit UI
# -------------------------
st.title("ARIMA (with exog) interest rate forecasting")

st.markdown("""
This app fetches daily SOFR and Fed Funds futures, constructs a combined rate,
creates a treasury implied-vol proxy, trains SARIMAX with exogenous vol, forecasts one week ahead,
and optionally pushes results to Supabase.
""")

col1, col2 = st.columns(2)
with col1:
    st.write("### Data window")
    latest_friday = latest_friday_end()
    st.write("Latest Friday (UTC):", latest_friday.date())
    one_year_ago = (latest_friday - timedelta(days=365)).date()
    st.write("One year back:", one_year_ago)
    st.info("Data will be fetched for that one-year window up to last Friday.")

with col2:
    st.write("### Model / Run options")
    p = st.number_input("AR order p", min_value=0, max_value=5, value=1)
    d = st.number_input("I order d", min_value=0, max_value=2, value=1)
    q = st.number_input("MA order q", min_value=0, max_value=5, value=1)
    run_and_push = st.checkbox("Push results to Supabase (table: hourly_forecasts)", value=False)

if st.button("Run pipeline and forecast now"):
    try:
        end_dt = latest_friday
        start_dt = datetime.combine(one_year_ago, datetime.min.time())

        st.info("Fetching SOFR from FRED...")
        sofr = fetch_sofr(start_dt, end_dt)  # date, sofr

        st.info("Fetching Fed Funds futures from Yahoo...")
        fed = fetch_fedfunds_futures(start_dt, end_dt)  # date, fedfund_future

        st.info("Fetching DGS10 history from FRED...")
        dgs10 = fetch_dgs10_latest_and_history(start_dt, end_dt)  # date, dgs10

        st.success("Data fetched. Building combined series...")
        combined = combined_rate_daily(sofr, fed)  # date, sofr, fedfund_future, combined_rate

        # attach ten_close (dgs10) if available
        combined = combined.merge(dgs10.rename(columns={'dgs10':'ten_close'}), on='date', how='left')

        # compute realized vol proxy
        vol_df = realized_vol_proxy(dgs10, window_days=ROLLING_VOL_WINDOW)  # date, dgs10, realized_vol
        # align vol to combined (on date)
        vol_df = vol_df[['date','realized_vol']].rename(columns={'realized_vol':'realized_vol'})
        combined = combined.merge(vol_df, on='date', how='left')
        combined['realized_vol'] = combined['realized_vol'].fillna(method='ffill').fillna(0.0)

        # initial volatility guess per instructions:
        current_yield = float(dgs10['dgs10'].dropna().iloc[-1])
        initial_vol = initial_vol_guess_from_yield(current_yield)
        st.write("Current 10Y yield (DGS10):", current_yield)
        st.write("Initial volatility guess (random within ±5% of yield):", initial_vol)

        # scale realized vol series to initial estimate (so the last value matches initial guess)
        combined['treas_implied_vol'] = scale_vol_series_to_initial(combined['realized_vol'], initial_vol)

        # train SARIMAX on combined_rate with exog treas_implied_vol
        st.info("Training SARIMAX model...")
        order = (int(p), int(d), int(q))
        res, mean_pred, ci = train_sarimax(combined.set_index('date')['combined_rate'], combined.set_index('date')[['treas_implied_vol']], order=order, steps=FORECAST_DAYS)
        st.success("Model trained.")

        # assemble forecast dataframe
        last_date = combined['date'].max()
        fc_index = pd.date_range(start=last_date + timedelta(days=1), periods=FORECAST_DAYS, freq='D')
        forecast_df = pd.DataFrame({'forecast_time': fc_index, 'combined_rate': mean_pred.values})
        forecast_df['lower_95'] = ci.iloc[:,0].values
        forecast_df['upper_95'] = ci.iloc[:,1].values

        # show charts
        st.subheader("Combined rate history + forecast")
        hist = combined[['date','combined_rate']].rename(columns={'date':'date', 'combined_rate':'combined_rate'})
        hist['is_forecast'] = False
        fc_plot = forecast_df.copy()
        fc_plot['is_forecast'] = True
        fc_plot = fc_plot.rename(columns={'forecast_time':'date'})
        plot_df = pd.concat([hist, fc_plot], ignore_index=True)
        plot_df['date'] = pd.to_datetime(plot_df['date'])

        base = alt.Chart(plot_df).encode(x='date:T')
        line_hist = base.mark_line(color='blue').encode(y='combined_rate:Q').transform_filter(alt.datum.is_forecast == False)
        line_fc = base.mark_line(color='orange', strokeDash=[5,5]).encode(y='combined_rate:Q').transform_filter(alt.datum.is_forecast == True)
        band = alt.Chart(fc_plot).mark_area(color='orange', opacity=0.2).encode(x='date:T', y='lower_95:Q', y2='upper_95:Q')
        st.altair_chart(line_hist + line_fc + band, use_container_width=True)

        st.subheader("Forecast table (7 days)")
        st.dataframe(forecast_df.set_index('forecast_time'))

        # if requested, push to Supabase
        if run_and_push and supa is not None:
            st.info("Pushing results to Supabase...")
            # prepare input rows (list of dict)
            input_rows = combined[['date','sofr','fedfund_future','ten_close','combined_rate','treas_implied_vol']].to_dict(orient='records')
            # prepare forecast rows
            frows = []
            for _, r in forecast_df.iterrows():
                frows.append({
                    'forecast_time': r['forecast_time'],
                    'combined_rate': float(r['combined_rate']),
                    'lower_95': float(r['lower_95']),
                    'upper_95': float(r['upper_95'])
                })
            push_to_supabase(supa, datetime.utcnow().isoformat(), input_rows, frows)
        elif run_and_push and supa is None:
            st.warning("Supabase not configured; set SUPABASE_URL and SUPABASE_KEY to push results.")

    except Exception as e:
        st.error("Pipeline failed: " + str(e))
        st.exception(e)

st.markdown("---")
st.caption("Prototype: ARIMA with exogenous (treasury implied vol proxy). Adjust orders and push to Supabase if desired.")
