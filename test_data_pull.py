# requirements (install if not present):
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
FRED_API_KEY = "91bb2c5920fb8f843abdbbfdfcab5345"  # hardcoded per request
FED_FUNDS_SYMBOL = "ZQ=F"
TREASURY_SYMBOL = "^TNX"
FORECAST_DAYS = 7

fred = Fred(api_key=FRED_API_KEY)

# ---------------------------
# Data fetch / caching
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_sofr_from_fred(start_date, end_date):
    """Fetch SOFR from FRED as a DataFrame with Date, Close columns (Date as date)."""
    series = fred.get_series('SOFR', observation_start=start_date, observation_end=end_date)
    df = series.reset_index()
    df.columns = ['Date', 'Close']
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

@st.cache_data(show_spinner=False)
def fetch_yahoo_data(symbol, start_date, end_date):
    """Fetch close price history for given symbol via yahooquery."""
    t = Ticker(symbol)
    df = t.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        return None
    df = df.reset_index()
    # Normalize columns
    if 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    if 'close' in df.columns:
        df.rename(columns={'close': 'Close'}, inplace=True)
    df = df[['Date', 'Close']].dropna()
    # Ensure Date is a date (not datetime)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

# ---------------------------
# Preprocessing & features
# ---------------------------
def calc_implied_volatility(series, window=20):
    return series.rolling(window=window).std().fillna(method='bfill')

def prepare_data(sofr_df, fedfund_df, treasury_df):
    """
    Returns:
      combined_df: DataFrame with columns ['Date','Combined'] (historical)
      exog_df: DataFrame with columns ['Date','ImpliedVol'] for exogenous modeling
    """
    # build interest rate from fed funds futures (price -> implied rate)
    fed_copy = fedfund_df.copy()
    fed_copy['InterestRate'] = 100 - fed_copy['Close']

    # merge SOFR and implied fed funds rate on Date
    merged = pd.merge(sofr_df, fed_copy[['Date','InterestRate']], on='Date', how='inner')
    merged['Combined'] = (merged['Close'] + merged['InterestRate']) / 2
    combined_df = merged[['Date','Combined']].copy()

    # prepare treasury exog
    treas = treasury_df.copy()
    # ensure Date exists as a column (defensive)
    if treas.index.name in ['Date', 'date']:
        treas = treas.reset_index()
    if 'Date' not in treas.columns:
        # fallback: create from index (may be integers)
        treas['Date'] = treas.index
    treas['Date'] = pd.to_datetime(treas['Date']).dt.date
    treas['ImpliedVol'] = calc_implied_volatility(treas['Close'])
    exog_df = treas[['Date','ImpliedVol']].copy()

    return combined_df, exog_df

# ---------------------------
# Modeling
# ---------------------------
def train_arima_model(target_df, exog_df, order=(1,1,1)):
    # expects target_df['Date','Combined'] and exog_df['Date','ImpliedVol']
    y = target_df.set_index('Date')['Combined']
    ex = exog_df.set_index('Date')['ImpliedVol']
    combined = pd.concat([y, ex], axis=1).dropna()
    model = sm.tsa.ARIMA(combined['Combined'], exog=combined[['ImpliedVol']], order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima_model(model_fit, exog_forecast):
    # exog_forecast must be a DataFrame/Series indexed by dates matching forecast horizon
    steps = len(exog_forecast)
    forecast_res = model_fit.get_forecast(steps=steps, exog=exog_forecast)
    forecast_df = forecast_res.summary_frame()
    # ensure index is datetime index
    if not isinstance(exog_forecast.index, pd.DatetimeIndex):
        # try to convert
        exog_forecast = exog_forecast.copy()
        exog_forecast.index = pd.to_datetime(exog_forecast.index)
    # align forecast_df index to exog_forecast index
    forecast_df.index = exog_forecast.index
    return forecast_df

# ---------------------------
# Plot helpers (Plotly)
# ---------------------------
def plot_time_series_with_p25(dates, values, p25_value, title, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name=y_label))
    # Add 25th percentile horizontal line (yellow, dashed)
    fig.add_shape(type="line",
                  x0=min(dates), x1=max(dates),
                  y0=p25_value, y1=p25_value,
                  line=dict(color="yellow", width=2, dash="dash"),
                  xref='x', yref='y')
    fig.add_annotation(x=dates[int(len(dates)*0.02)] if len(dates)>0 else dates[0],
                       y=p25_value,
                       text=f"25th percentile ({p25_value:.4f})",
                       showarrow=False,
                       font=dict(color="yellow"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_label, template='plotly_white')
    return fig

def plot_forecast_with_p25(forecast_index, forecast_mean, p25_value, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_mean, mode='lines+markers', name='Forecast Mean'))
    # Add 25th percentile horizontal line (yellow, dashed)
    fig.add_shape(type="line",
                  x0=min(forecast_index), x1=max(forecast_index),
                  y0=p25_value, y1=p25_value,
                  line=dict(color="yellow", width=2, dash="dash"),
                  xref='x', yref='y')
    fig.add_annotation(x=forecast_index[int(len(forecast_index)*0.02)] if len(forecast_index)>0 else forecast_index[0],
                       y=p25_value,
                       text=f"25th percentile ({p25_value:.4f})",
                       showarrow=False,
                       font=dict(color="yellow"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Rate", template='plotly_white')
    return fig

# ---------------------------
# Utils
# ---------------------------
def get_last_friday():
    today = datetime.utcnow().date()
    offset = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=offset)
    return last_friday

# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.title("Interest Rate Forecasting Dashboard (ARIMA)")

    last_friday = get_last_friday()
    start_date = last_friday - timedelta(days=365)

    st.markdown(f"**Data range:** {start_date} to {last_friday}")

    # Buttons / flow
    if st.button("Fetch Data"):
        with st.spinner("Fetching SOFR (FRED), Fed Funds, Treasury..."):
            sofr = fetch_sofr_from_fred(start_date, last_friday)
            fedfund = fetch_yahoo_data(FED_FUNDS_SYMBOL, start_date, last_friday)
            treasury = fetch_yahoo_data(TREASURY_SYMBOL, start_date, last_friday)
            if sofr is None or fedfund is None or treasury is None:
                st.error("Failed to fetch one or more data sources.")
                st.stop()
            st.session_state['sofr'] = sofr
            st.session_state['fedfund'] = fedfund
            st.session_state['treasury'] = treasury
            st.success("Data fetched and cached.")

    # Prepare & show historical charts
    if all(k in st.session_state for k in ['sofr','fedfund','treasury']):
        sofr = st.session_state['sofr']
        fedfund = st.session_state['fedfund']
        treasury = st.session_state['treasury']

        combined_df, exog_df = prepare_data(sofr, fedfund, treasury)
        st.session_state['combined_df'] = combined_df
        st.session_state['exog_df'] = exog_df

        # compute historical 25th percentile from combined historical series
        p25_combined = float(combined_df['Combined'].quantile(0.25))

        # Plot combined historical series with 25th percentile (yellow)
        combined_plot_dates = pd.to_datetime(combined_df['Date'])
        combined_plot_values = combined_df['Combined']
        fig_combined = plot_time_series_with_p25(combined_plot_dates, combined_plot_values,
                                                 p25_combined,
                                                 title="Combined Interest Rate (SOFR & Fed Funds Futures)",
                                                 y_label="Combined Rate")
        st.plotly_chart(fig_combined, use_container_width=True)

        # Plot treasury implied vol history (no percentile line here)
        exog_plot = exog_df.copy()
        exog_plot['Date'] = pd.to_datetime(exog_plot['Date'])
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=exog_plot['Date'], y=exog_plot['ImpliedVol'], mode='lines', name='ImpliedVol'))
        fig_vol.update_layout(title="Treasury Implied Volatility", xaxis_title="Date", yaxis_title="ImpliedVol",
                              template='plotly_white')
        st.plotly_chart(fig_vol, use_container_width=True)

        # save p25 for later use in forecast chart
        st.session_state['p25_combined'] = p25_combined

    # Train model
    if 'combined_df' in st.session_state and 'exog_df' in st.session_state:
        if st.button("Train ARIMA(1,1,1)"):
            with st.spinner("Training ARIMA(1,1,1) with exogenous implied vol..."):
                model_fit = train_arima_model(st.session_state['combined_df'], st.session_state['exog_df'], order=(1,1,1))
                st.session_state['model_fit'] = model_fit
            st.success("Model trained successfully.")

    # Forecast
    if 'model_fit' in st.session_state and 'exog_df' in st.session_state:
        if st.button(f"Generate {FORECAST_DAYS}-day Forecast"):
            # Prepare exogenous forecast: use last observed implied vol (historical) repeated
            exog_hist = st.session_state['exog_df'].copy()
            # Ensure Date is datetime index
            exog_hist['Date'] = pd.to_datetime(exog_hist['Date'])
            last_observed_vol = float(exog_hist.set_index('Date')['ImpliedVol'].iloc[-1])

            forecast_index = pd.date_range(start=(exog_hist['Date'].max() + pd.Timedelta(days=1)), periods=FORECAST_DAYS, freq='D')
            exog_forecast = pd.DataFrame({'ImpliedVol': [last_observed_vol]*FORECAST_DAYS}, index=forecast_index)

            with st.spinner("Forecasting..."):
                forecast_df = forecast_arima_model(st.session_state['model_fit'], exog_forecast)

            # Ensure numeric mean exists
            if 'mean' not in forecast_df.columns:
                st.error("Forecast result is missing 'mean' column.")
                st.stop()

            # Plot forecast mean with historical 25th percentile (yellow)
            p25_val = float(st.session_state.get('p25_combined', np.nan))
            fig_forecast = plot_forecast_with_p25(forecast_df.index, forecast_df['mean'], p25_val,
                                                  title=f"{FORECAST_DAYS}-Day Ahead Forecast (mean)")
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Show forecast table (mean and confidence intervals)
            display_df = forecast_df[['mean','mean_ci_lower','mean_ci_upper']].copy()
            display_df.index = display_df.index.date
            display_df = display_df.rename(columns={'mean':'Forecast','mean_ci_lower':'LowerCI','mean_ci_upper':'UpperCI'})
            st.subheader("Forecast Table (7 days)")
            st.dataframe(display_df)

    # End of app
    st.markdown("---")
    st.caption("Notes: 25th percentile line (yellow) is computed from historical combined rate and drawn on both historical and forecast views.")

if __name__ == "__main__":
    main()
