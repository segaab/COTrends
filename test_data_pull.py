import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from yahooquery import Ticker
import statsmodels.api as sm
from fredapi import Fred

# Constants
FED_FUNDS_SYMBOL = "ZQ=F"
TREASURY_SYMBOL = "^TNX"
FORECAST_DAYS = 7

# Hardcoded FRED API Key (use with caution)
FRED_API_KEY = "91bb2c5920fb8f843abdbbfdfcab5345"
fred = Fred(api_key=FRED_API_KEY)

@st.cache_data(show_spinner=False)
def fetch_sofr_from_fred(start_date, end_date):
    data = fred.get_series('SOFR', observation_start=start_date, observation_end=end_date)
    df = data.reset_index()
    df.columns = ['Date', 'Close']
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

@st.cache_data(show_spinner=False)
def fetch_yahoo_data(symbol, start_date, end_date):
    t = Ticker(symbol)
    df = t.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if df.empty:
        st.error(f"No data for {symbol}")
        return None
    df = df.reset_index()
    if 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    if 'close' in df.columns:
        df.rename(columns={'close': 'Close'}, inplace=True)
    df = df[['Date', 'Close']].dropna()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def calc_implied_volatility(series):
    return series.rolling(window=20).std().fillna(method='bfill')

def combined_interest_rate(sofr_df, fedfund_df):
    fedfund_df = fedfund_df.copy()
    fedfund_df['InterestRate'] = 100 - fedfund_df['Close']  # futures price to rate %
    merged = pd.merge(sofr_df, fedfund_df[['Date', 'InterestRate']], on='Date')
    merged['Combined'] = (merged['Close'] + merged['InterestRate']) / 2
    return merged[['Date', 'Combined']]

def train_arima(data_df, exog_df):
    data_df = data_df.set_index('Date')
    exog_df = exog_df.set_index('Date')
    combined = pd.concat([data_df, exog_df], axis=1).dropna()
    model = sm.tsa.ARIMA(combined['Combined'], exog=combined[exog_df.columns], order=(1,1,1))
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, exog_forecast, steps=FORECAST_DAYS):
    forecast_res = model_fit.get_forecast(steps=steps, exog=exog_forecast)
    forecast_df = forecast_res.summary_frame()
    forecast_df['Date'] = pd.date_range(start=exog_forecast.index[0], periods=steps)
    forecast_df.set_index('Date', inplace=True)
    return forecast_df

def get_last_friday():
    today = datetime.utcnow().date()
    offset = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=offset)
    return last_friday

def main():
    st.title("Interest Rate Forecasting Dashboard with FRED SOFR")

    last_friday = get_last_friday()
    start_date = last_friday - timedelta(days=365)

    st.write(f"Fetching data from **{start_date}** to **{last_friday}**")

    if st.button("Fetch Data"):
        with st.spinner("Fetching SOFR (FRED), Fed Funds, Treasury data..."):
            sofr = fetch_sofr_from_fred(start_date, last_friday)
            fedfund = fetch_yahoo_data(FED_FUNDS_SYMBOL, start_date, last_friday)
            treasury = fetch_yahoo_data(TREASURY_SYMBOL, start_date, last_friday)
            if sofr is None or fedfund is None or treasury is None:
                st.stop()
            st.session_state['sofr'] = sofr
            st.session_state['fedfund'] = fedfund
            st.session_state['treasury'] = treasury

    if 'sofr' in st.session_state and 'fedfund' in st.session_state and 'treasury' in st.session_state:
        sofr = st.session_state['sofr']
        fedfund = st.session_state['fedfund']
        treasury = st.session_state['treasury']

        combined_df = combined_interest_rate(sofr, fedfund)
        st.subheader("Combined Interest Rate (SOFR & Fed Funds Futures)")
        st.line_chart(combined_df.set_index('Date')['Combined'])

        # Debug prints before resetting index
        st.write("=== Treasury DataFrame BEFORE reset_index ===")
        st.write(f"Columns: {treasury.columns.tolist()}")
        st.write(f"Index name: {treasury.index.name}")
        st.write(treasury.head())

        # Reset index if index is 'Date' or 'date'
        if treasury.index.name in ['Date', 'date']:
            treasury = treasury.reset_index()

        # Debug prints after resetting index
        st.write("=== Treasury DataFrame AFTER reset_index ===")
        st.write(f"Columns: {treasury.columns.tolist()}")
        st.write(f"Index name: {treasury.index.name}")
        st.write(treasury.head())

        treasury['ImpliedVol'] = calc_implied_volatility(treasury['Close'])

        st.subheader("Treasury Yield & Implied Volatility")
        st.line_chart(treasury.set_index('Date')[['Close', 'ImpliedVol']])

        st.session_state['combined_df'] = combined_df
        st.session_state['treasury'] = treasury

    if st.session_state.get('combined_df') is not None and st.session_state.get('treasury') is not None:
        if st.button("Train ARIMA(1,1,1) and Forecast"):
            combined_df = st.session_state['combined_df']
            treasury = st.session_state['treasury']

            # Debug prints before preparing exog_df
            st.write("=== Treasury DataFrame BEFORE preparing exog_df ===")
            st.write(f"Columns: {treasury.columns.tolist()}")
            st.write(f"Index name: {treasury.index.name}")
            st.write(treasury.head())

            # Reset index if needed
            if treasury.index.name in ['Date', 'date']:
                treasury = treasury.reset_index()

            # Check if 'Date' column exists
            if 'Date' not in treasury.columns:
                st.error("Error: 'Date' column missing from treasury DataFrame!")
                st.stop()

            exog_df = treasury[['Date', 'ImpliedVol']].copy()
            exog_df.set_index('Date', inplace=True)

            with st.spinner("Training ARIMA model..."):
                model_fit = train_arima(combined_df, exog_df)

            last_date = exog_df.index.max()
            forecast_start = last_date + timedelta(days=1)
            last_vol = exog_df.iloc[-1]['ImpliedVol']
            exog_forecast = pd.DataFrame({'ImpliedVol': [last_vol]*FORECAST_DAYS},
                                        index=pd.date_range(start=forecast_start, periods=FORECAST_DAYS))

            forecast_df = forecast_arima(model_fit, exog_forecast, steps=FORECAST_DAYS)

            st.subheader(f"{FORECAST_DAYS}-Day Ahead Forecast")
            st.line_chart(forecast_df['mean'])

            st.dataframe(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']])

if __name__ == "__main__":
    main()
