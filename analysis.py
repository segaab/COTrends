# streamlit_fetch_exchanges.py
import streamlit as st
import pandas as pd
import requests
from io import StringIO

# Use HTTPS URLs instead of FTP
EXCHANGE_FILES = {
    "NASDAQ": "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "NYSE/AMEX": "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
}

st.title("Exchange Ticker List Preview")

exchange = st.selectbox("Select Exchange to Preview", list(EXCHANGE_FILES.keys()))

if st.button("Fetch Tickers"):
    url = EXCHANGE_FILES[exchange]
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), sep='|')
        st.write(f"Preview of {exchange} tickers:")
        st.dataframe(df.head(20))
        st.write(f"Columns: {df.columns.tolist()}")
        st.write(f"Total rows: {len(df)}")
    except Exception as e:
        st.error(f"Error fetching {exchange}: {e}")
