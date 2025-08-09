import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from yahooquery import Ticker
import statsmodels.api as sm
from fredapi import Fred

# Constants and API Keys
FRED_API_KEY = "91bb2c5920fb8f843abdbbfdfcab5345"
FED_FUNDS_SYMBOL = "ZQ=F"
TREASURY_SYMBOL = "^TNX"
FORECAST_DAYS = 7

# Initialize Fred API client
fred = Fred(api_key=FRED_API_KEY)

# --- Data Fetching Functions ---

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
    # Rename columns if needed
    if 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    if 'close' in df.columns:
        df.rename(columns={'close': 'Close'}, inplace=True)
    df = df[['Date', 'Close']].dropna()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

# --- Data Preparation ---

def calc_implied_volatility(series):
    return series.rolling(window=20).std().fillna(method='bfill')

def prepare_data(sofr_df, fedfund_df, treasury_df):
    # Combine SOFR and Fed Funds futures into a single rate
    fedfund_copy = fedfund_df.copy()
    fedfund_copy['InterestRate'] = 100 - fedfund_copy['Close']  # Convert futures price to rate %
    combined_df = pd.merge(sofr_df, fedfund_copy[['Date', 'InterestRate']], on='Date')
    combined_df['Combined'] = (combined_df['Close'] + combined_df['InterestRate']) / 2
    
    # Prepare Treasury data for exogenous variable
    treasury_copy = treasury_df.copy()
    if treasury_copy.index.name in ['Date', 'date']:
        treasury_copy.reset_index(inplace=True)
    if 'Date' not in treasury_copy.columns:
        treasury_copy['Date'] = treasury_copy.index
    treasury_copy['Date'] = pd.to_datetime(treasury_copy['Date']).dt.date
    treasury_copy['ImpliedVol'] = calc_implied_volatility(treasury_copy['Close'])
    
    exog_df = treasury_copy[['Date', 'ImpliedVol']].copy()
    
    return combined_df[['Date', 'Combined']], exog_df

# --- Modeling Functions ---

def train_arima_model(target_df, exog_df):
    target_df = target_df.set_index('Date')
    exog_df = exog_df.set_index('Date')
    combined = pd.concat([target_df, exog_df], axis=1).dropna()
    model = sm.tsa.ARIMA(combined['Combined'], exog=combined[['ImpliedVol']], order=(1,1,1))
    model_fit = model.fit()
    return model_fit

def forecast_arima_model(model_fit, exog_forecast):
    steps = len(exog_forecast)
    forecast_res = model_fit.get_forecast(steps=steps, exog=exog_forecast)
    forecast_df = forecast_res.summary_frame()
    forecast_df['Date'] = pd.date_range(start=exog_forecast.index[0], periods=steps)
    forecast_df.set_index('Date', inplace=True)
    return forecast_df

# --- Helper Functions ---

def get_last_friday():
    today = datetime.utcnow().date()
    offset = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=offset)
    return last_friday

# --- Streamlit App ---

def main():
    st.title("Interest Rate Forecasting Dashboard with ARIMA")

    last_friday = get_last_friday()
    start_date = last_friday - timedelta(days=365)

    st.write(f"### Data range: {start_date} to {last_friday}")

    # Step 1: Fetch data
    if st.button("Fetch Data"):
        with st.spinner("Fetching SOFR, Fed Funds Futures, Treasury data..."):
            sofr = fetch_sofr_from_fred(start_date, last_friday)
            fedfund = fetch_yahoo_data(FED_FUNDS_SYMBOL, start_date, last_friday)
            treasury = fetch_yahoo_data(TREASURY_SYMBOL, start_date, last_friday)

            if sofr is None or fedfund is None or treasury is None:
                st.error("Failed to fetch all required data. Please try again.")
                st.stop()

            st.session_state['sofr'] = sofr
            st.session_state['fedfund'] = fedfund
            st.session_state['treasury'] = treasury
            st.success("Data fetched successfully!")

    # Step 2: Prepare and show data
    if all(k in st.session_state for k in ['sofr', 'fedfund', 'treasury']):
        sofr = st.session_state['sofr']
        fedfund = st.session_state['fedfund']
        treasury = st.session_state['treasury']

        combined_df, exog_df = prepare_data(sofr, fedfund, treasury)

        st.session_state['combined_df'] = combined_df
        st.session_state['exog_df'] = exog_df

        st.subheader("Combined Interest Rate (SOFR & Fed Funds Futures)")
        st.line_chart(combined_df.set_index('Date')['Combined'])

        st.subheader("Treasury Yield & Implied Volatility")
        treasury_vol = exog_df.set_index('Date')
        st.line_chart(treasury_vol)

    # Step 3: Train model
    if 'combined_df' in st.session_state and 'exog_df' in st.session_state:
        if st.button("Train ARIMA Model"):
            with st.spinner("Training ARIMA(1,1,1) with exogenous variable..."):
                model_fit = train_arima_model(st.session_state['combined_df'], st.session_state['exog_df'])
                st.session_state['model_fit'] = model_fit
            st.success("Model training complete!")

    # Step 4: Forecast
    if 'model_fit' in st.session_state and 'exog_df' in st.session_state:
        if st.button("Generate Forecast"):
            exog_df = st.session_state['exog_df']
            last_date = exog_df['Date'].max()
            last_vol = exog_df.set_index('Date').iloc[-1]['ImpliedVol']
            forecast_start_date = last_date + timedelta(days=1)
            forecast_index = pd.date_range(start=forecast_start_date, periods=FORECAST_DAYS)

            exog_forecast = pd.DataFrame({'ImpliedVol': [last_vol]*FORECAST_DAYS}, index=forecast_index)

            with st.spinner("Generating forecast..."):
                forecast_df = forecast_arima_model(st.session_state['model_fit'], exog_forecast)

            st.subheader(f"{FORECAST_DAYS}-Day Ahead Forecast")
            st.line_chart(forecast_df['mean'])
            st.dataframe(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']])

if __name__ == "__main__":
    main()
