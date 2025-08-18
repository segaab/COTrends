# streamlit_yahoo_json.py

import streamlit as st
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Shared counter for progress
completed_chunks = 0
lock = threading.Lock()

def call_yahoo_search(query):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=100&newsCount=0"
    confirmed = False
    while not confirmed:
        try:
            r = requests.get(url)
            r.raise_for_status()
            confirmed = True
        except Exception:
            time.sleep(1)
    return r.json()

def scrape_prefix_batch(prefix_chunk):
    global completed_chunks
    symbols = set()

    # Iterate over single + double character prefixes
    for term1 in prefix_chunk:
        for term2 in prefix_chunk:
            search_term = term1 + term2
            data = call_yahoo_search(search_term)
            quotes = data.get("quotes", [])
            for quote in quotes:
                symbol = quote.get("symbol")
                if symbol:
                    symbols.add(symbol)

    # Update progress
    with lock:
        completed_chunks += 1

    return symbols

def chunk_list(lst, n):
    k = len(lst) // n
    return [lst[i*k:(i+1)*k] for i in range(n-1)] + [lst[(n-1)*k:]]

@st.cache_data(show_spinner=False)
def threaded_scrape():
    global completed_chunks
    completed_chunks = 0

    # Letters A-Z and numbers 0-9
    search_set = [chr(x) for x in range(65, 91)] + [chr(x) for x in range(48, 58)]
    chunks = chunk_list(search_set, 5)  # 5 threads

    all_symbols = set()
    total_chunks = len(chunks)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(scrape_prefix_batch, chunk) for chunk in chunks]

        progress_bar = st.progress(0.0)
        while completed_chunks < total_chunks:
            progress_bar.progress(completed_chunks / total_chunks)
            time.sleep(0.5)

        for f in as_completed(futures):
            all_symbols.update(f.result())

        progress_bar.progress(1.0)

    return sorted(list(all_symbols))

def main():
    st.title("Yahoo Finance Ticker Scraper (JSON + Threaded)")

    if st.button("Start Scraping"):
        st.write("Scraping in progress...")
        tickers = threaded_scrape()
        st.success(f"Scraping complete! Found {len(tickers)} symbols.")

        st.dataframe(tickers)

        content = "\n".join(tickers)
        st.download_button(
            "Download as txt",
            data=content,
            file_name="yh_all_symbols.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
