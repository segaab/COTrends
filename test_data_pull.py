import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
from datetime import datetime, timedelta
import statsmodels.api as sm
from supabase import create_client, Client

# Supabase setup — fill these with your info
SUPABASE_URL = "https://dzddytphimhoxeccxqsw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR6ZGR5dHBoaW1ob3hlY2N4cXN3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTM2Njc5NCwiZXhwIjoyMDY2OTQyNzk0fQ.ng0ST7-V-cDBD0Jc80_0DFWXylzE-gte2I9MCX7qb0Q"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Constants
FED_FUNDS_SYMBOL = "ZQ=F"
SOFR_SYMBOL = "SOFR"
TREASURY_SYMBOL = "^TNX"  # 10-year Treasury yield index on YahooFinance (daily)
FORECAST_DAYS = 7

# Utils
def get_last_friday():
    today = datetime.utcnow().date()
    offset = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=offset)
    return last_friday

def fetch_yahoo_data(symbol, start_date, end_date):
    t = Ticker(symbol)
    df = t.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if df.empty:
        st.error(f"No data for {symbol}")
        return None
    df = df.reset_index()
    df = df[['date', 'close']].dropna()
    df.rename(columns={'date':'Date', 'close':'Close'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

def calc_implied_volatility(yield_series):
    # Simple rolling std as implied vol proxy
    return yield_series.rolling(window=20).std().fillna(method='bfill')

def combined_interest_rate(sofr_df, fedfund_df):
    merged = pd.merge(sofr_df, fedfund_df, on='Date', suffixes=('_SOFR', '_Fed'))
    merged['Combined'] = (merged['Close_SOFR'] + merged['Close_Fed']) / 2
    return merged[['Date', 'Combined']]

def train_arima_with_exog(data_df, exog_df):
    data_df = data_df.set_index('Date')
    exog_df = exog_df.set_index('Date')
    # Align indices
    combined = pd.concat([data_df, exog_df], axis=1).dropna()
    model = sm.tsa.ARIMA(combined['Combined'], exog=combined[exog_df.columns], order=(1,1,1))
    model_fit = model.fit()
    return model_fit

def forecast(model_fit, exog_forecast, steps=FORECAST_DAYS):
    forecast_res = model_fit.get_forecast(steps=steps, exog=exog_forecast)
    forecast_df = forecast_res.summary_frame()
    forecast_df['Date'] = pd.date_range(start=exog_forecast.index[0], periods=steps)
    return forecast_df

def upload_to_supabase(df_inputs, df_forecast):
    try:
        # Inputs - add a flag is_forecast = false
        input_records = df_inputs.to_dict(orient='records')
        for rec in input_records:
            rec['is_forecast'] = False
        res1 = supabase.table('interest_rate_forecasts').insert(input_records).execute()

        # Forecast - add flag is_forecast = true
        forecast_records = df_forecast.to_dict(orient='records')
        for rec in forecast_records:
            rec['is_forecast'] = True
            rec['Date'] = rec['Date'].date()  # convert Timestamp to date for Supabase
        res2 = supabase.table('interest_rate_forecasts').insert(forecast_records).execute()

        st.success("Uploaded forecast and input data to Supabase")
    except Exception as e:
        st.error(f"Failed to upload to Supabase: {e}")

# Main Streamlit app
def main():
    st.title("Interest Rate Forecasting with ARIMA + Exogenous Volatility")

    last_friday = get_last_friday()
    start_date = last_friday - timedelta(days=365)

    # Fetch data
    st.info(f"Fetching data from {start_date} to {last_friday}...")

    sofr = fetch_yahoo_data(SOFR_SYMBOL, start_date, last_friday)
    fedfund = fetch_yahoo_data(FED_FUNDS_SYMBOL, start_date, last_friday)
    treasury = fetch_yahoo_data(TREASURY_SYMBOL, start_date, last_friday)

    if sofr is None or fedfund is None or treasury is None:
        st.stop()

    # Calculate combined interest rate
    combined_df = combined_interest_rate(sofr, fedfund)
    st.subheader("Combined Interest Rate (SOFR & Fed Funds Futures)")
    st.line_chart(combined_df.set_index('Date')['Combined'])

    # Treasury implied volatility
    treasury['ImpliedVol'] = calc_implied_volatility(treasury['Close'])
    st.subheader("Treasury Yield & Implied Volatility")
    st.line_chart(treasury.set_index('Date')[['Close', 'ImpliedVol']])

    # Prepare exog for ARIMA (using implied vol)
    exog_df = treasury[['Date', 'ImpliedVol']].copy()
    exog_df = exog_df.set_index('Date')

    # Train ARIMA
    st.info("Training ARIMA model with exogenous volatility...")
    model_fit = train_arima_with_exog(combined_df, exog_df)

    # Prepare exog for forecast - use last known implied vol repeated for forecast days
    last_vol = exog_df['ImpliedVol'][-1]
    forecast_index = pd.date_range(start=last_friday + timedelta(days=1), periods=FORECAST_DAYS)
    exog_forecast = pd.DataFrame({'ImpliedVol': [last_vol]*FORECAST_DAYS}, index=forecast_index)

    # Forecast
    forecast_df = forecast(model_fit, exog_forecast, steps=FORECAST_DAYS)
    st.subheader(f"{FORECAST_DAYS}-Day Ahead Forecast")
    st.line_chart(forecast_df.set_index('Date')['mean'])

    # Upload button
    if st.button("Upload data & forecast to Supabase"):
        # Prepare input data for upload
        input_data = combined_df.set_index('Date')
        input_data['ImpliedVol'] = exog_df['ImpliedVol']
        input_data.reset_index(inplace=True)

        # Prepare forecast data
        forecast_upload_df = forecast_df.rename(columns={
            'mean': 'Forecast',
            'mean_ci_lower': 'LowerCI',
            'mean_ci_upper': 'UpperCI'
        })[['Date', 'Forecast', 'LowerCI', 'UpperCI']]

        upload_to_supabase(input_data, forecast_upload_df)

if __name__ == "__main__":
    main()
