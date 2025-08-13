# requirements:
# pip install streamlit pandas numpy yahooquery statsmodels fredapi plotly

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from yahooquery import Ticker
import statsmodels.api as sm
from fredapi import Fred
import plotly.graph_objects as go

# ---------------------------
# Config / Constants
# ---------------------------
FRED_API_KEY = "91bb2c5920fb8f843abdbbfdfcab5345"
FED_FUNDS_SYMBOL = "ZQ=F"
TREASURY_SYMBOL = "^TNX"
FORECAST_HOURS = 24*7  # hourly forecast for 7 days

fred = Fred(api_key=FRED_API_KEY)

# ---------------------------
# Data fetch / caching
# ---------------------------
@st.cache_data
def fetch_sofr_from_fred(start_date, end_date):
    series = fred.get_series('SOFR', observation_start=start_date, observation_end=end_date)
    df = series.reset_index()
    df.columns = ['Date', 'Close']
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def fetch_yahoo_data(symbol, start_date, end_date):
    t = Ticker(symbol)
    df = t.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return None
    df = df.reset_index()
    if 'date' in df.columns:
        df.rename(columns={'date':'Date'}, inplace=True)
    if 'close' in df.columns:
        df.rename(columns={'close':'Close'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df[['Date','Close']].dropna()

# ---------------------------
# Helper functions
# ---------------------------
def calc_implied_volatility(series, window=20):
    return series.rolling(window=window).std().fillna(method='bfill')

def prepare_data(sofr_df, fedfund_df, treasury_df):
    fed_copy = fedfund_df.copy()
    fed_copy['InterestRate'] = 100 - fed_copy['Close']
    merged = pd.merge(sofr_df, fed_copy[['Date','InterestRate']], on='Date', how='inner')
    merged['Combined'] = (merged['Close'] + merged['InterestRate']) / 2
    combined_df = merged[['Date','Combined']].copy()

    treas = treasury_df.copy()
    if treas.index.name in ['Date','date']:
        treas = treas.reset_index()
    if 'Date' not in treas.columns:
        treas['Date'] = treas.index
    treas['ImpliedVol'] = calc_implied_volatility(treas['Close'])
    exog_df = treas[['Date','ImpliedVol']].copy()
    return combined_df, exog_df

def train_arima_model(target_df, exog_df, order=(1,1,1)):
    y = target_df.set_index('Date')['Combined']
    ex = exog_df.set_index('Date')['ImpliedVol']
    combined = pd.concat([y, ex], axis=1).dropna()
    model = sm.tsa.ARIMA(combined['Combined'], exog=combined[['ImpliedVol']], order=order)
    return model.fit()

def generate_hourly_exog_forecast(exog_hist, forecast_hours=FORECAST_HOURS):
    last_vol = exog_hist['ImpliedVol'].iloc[-1]
    rolling_std = exog_hist['ImpliedVol'].rolling(20).std().iloc[-1]
    forecast_index = pd.date_range(start=exog_hist['Date'].iloc[-1] + pd.Timedelta(hours=1),
                                   periods=forecast_hours, freq='H')
    # Generate hourly exog with small random variation based on rolling volatility
    np.random.seed(42)
    variations = np.random.normal(0, rolling_std/2, size=forecast_hours)
    exog_forecast = pd.DataFrame({'ImpliedVol': last_vol + variations}, index=forecast_index)
    return exog_forecast

def forecast_arima_hourly(model_fit, exog_forecast):
    forecast_res = model_fit.get_forecast(steps=len(exog_forecast), exog=exog_forecast)
    forecast_df = forecast_res.summary_frame()
    forecast_df.index = exog_forecast.index
    return forecast_df

# ---------------------------
# Plot helpers
# ---------------------------
def plot_time_series_with_p25(dates, values, p25_value, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name=y_label))
    fig.add_shape(type="line", x0=min(dates), x1=max(dates),
                  y0=p25_value, y1=p25_value,
                  line=dict(color="yellow", width=2, dash="dash"))
    fig.add_annotation(x=dates[int(len(dates)*0.02)],
                       y=p25_value,
                       text=f"25th percentile ({p25_value:.4f})",
                       showarrow=False, font=dict(color="yellow"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_label, template='plotly_white')
    return fig

def plot_forecast_with_p25(forecast_index, forecast_mean, p25_value, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines+markers', name='Forecast Mean'))
    fig.add_shape(type="line", x0=min(forecast_index), x1=max(forecast_index),
                  y0=p25_value, y1=p25_value,
                  line=dict(color="yellow", width=2, dash="dash"))
    fig.add_annotation(x=forecast_index[int(len(forecast_index)*0.02)],
                       y=p25_value,
                       text=f"25th percentile ({p25_value:.4f})",
                       showarrow=False, font=dict(color="yellow"))
    fig.update_layout(title=title, xaxis_title="Datetime", yaxis_title="Rate", template='plotly_white')
    return fig

def get_last_complete_hour_for_date(end_date):
    now = datetime.utcnow()
    if end_date >= now.date():
        last_complete_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    else:
        last_complete_hour = datetime.combine(end_date, datetime.min.time()) + timedelta(hours=23)
    return last_complete_hour

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("Interest Rate Forecasting Dashboard (Hourly ARIMA Forecast)")
    st.markdown("""
    **Summary:**
    Forecasts short-term interest rates using ARIMA(1,1,1) with Treasury implied volatility as exogenous.
    Generates hourly 7-day forecasts and plots combined SOFR + Fed Funds rate.
    25th percentile historical rate displayed in yellow.
    """)

    # Date range input
    col1, col2 = st.columns(2)
    today = datetime.utcnow().date()
    with col1:
        start_date = st.date_input("Start Date", today - timedelta(days=365))
    with col2:
        selected_end = st.date_input("End Date", today)
        end_date = get_last_complete_hour_for_date(selected_end).date()

    st.markdown(f"**Data range:** {start_date} to {end_date}")

    if st.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            sofr = fetch_sofr_from_fred(start_date, end_date)
            fedfund = fetch_yahoo_data(FED_FUNDS_SYMBOL, start_date, end_date)
            treasury = fetch_yahoo_data(TREASURY_SYMBOL, start_date, end_date)
            if sofr is None or fedfund is None or treasury is None:
                st.error("Failed to fetch one or more datasets.")
                st.stop()
            st.session_state['sofr'] = sofr
            st.session_state['fedfund'] = fedfund
            st.session_state['treasury'] = treasury
            st.success("Data fetched successfully.")

    if all(k in st.session_state for k in ['sofr','fedfund','treasury']):
        combined_df, exog_df = prepare_data(st.session_state['sofr'],
                                            st.session_state['fedfund'],
                                            st.session_state['treasury'])
        st.session_state['combined_df'] = combined_df
        st.session_state['exog_df'] = exog_df
        p25_val = float(combined_df['Combined'].quantile(0.25))
        st.session_state['p25_combined'] = p25_val

        # Historical combined rate
        fig_combined = plot_time_series_with_p25(pd.to_datetime(combined_df['Date']),
                                                 combined_df['Combined'],
                                                 p25_val,
                                                 "Combined Interest Rate",
                                                 "Rate")
        st.plotly_chart(fig_combined, use_container_width=True)

        # Treasury implied vol
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=pd.to_datetime(exog_df['Date']),
                                     y=exog_df['ImpliedVol'],
                                     mode='lines', name='ImpliedVol'))
        fig_vol.update_layout(title="Treasury Implied Volatility", xaxis_title="Date",
                              yaxis_title="ImpliedVol", template='plotly_white')
        st.plotly_chart(fig_vol, use_container_width=True)

    # Train ARIMA
    if 'combined_df' in st.session_state and 'exog_df' in st.session_state:
        if st.button("Train ARIMA(1,1,1)"):
            with st.spinner("Training ARIMA model..."):
                model_fit = train_arima_model(st.session_state['combined_df'], st.session_state['exog_df'])
                st.session_state['model_fit'] = model_fit
            st.success("Model trained successfully.")

    # Forecast
    if 'model_fit' in st.session_state:
        if st.button("Generate Hourly Forecast"):
            exog_forecast = generate_hourly_exog_forecast(st.session_state['exog_df'])
            forecast_df = forecast_arima_hourly(st.session_state['model_fit'], exog_forecast)
            st.session_state['forecast_df'] = forecast_df

            fig_forecast = plot_forecast_with_p25(forecast_df.index,
                                                  forecast_df['mean'],
                                                  st.session_state['p25_combined'],
                                                  "7-Day Hourly Forecast")
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.dataframe(forecast_df[['mean','mean_ci_lower','mean_ci_upper']].round(4))

if __name__ == "__main__":
    main()
