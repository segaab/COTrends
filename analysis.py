# streamlit_app.py

import streamlit as st
from yahoo_finance_stock_symbols_scraper import YahooSymbolsScraper

def get_all_tickers():
    scraper = YahooSymbolsScraper()
    return scraper.get_all_symbols()  # returns a list or set of ticker strings

def main():
    st.title("Yahoo Finance Ticker List")
    st.write("Fetching all available ticker symbols from Yahoo Finance...")

    tickers = get_all_tickers()
    st.success(f"Found {len(tickers)} tickers.")

    # Optionally filter or search
    search = st.text_input("Filter tickers (substring match):").upper()
    filtered = [t for t in tickers if search in t] if search else tickers

    st.dataframe(filtered)

if __name__ == "__main__":
    main()
