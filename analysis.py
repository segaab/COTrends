# streamlit_yahoo_json_parse.py

import streamlit as st
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import json
import re

# Shared counter for progress
completed_chunks = 0
lock = threading.Lock()

hdr = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
}

def call_url(url):
    confirmed = False
    while not confirmed:
        try:
            r = requests.get(url, headers=hdr)
            r.raise_for_status()
            confirmed = True
        except Exception:
            time.sleep(1)
    return r.text

def extract_symbols_from_body(body):
    """
    Extract tickers from embedded JSON in Yahoo Finance page
    """
    symbols = set()
    # Find the script containing root.App.main
    match = re.search(r'root\.App\.main\s*=\s*({.*});', body)
    if not match:
        return symbols

    data = json.loads(match.group(1))
    # Navigate to quotes array
    try:
        quotes = data['context']['dispatcher']['stores']['QuoteLookupStore']['quotes']
        for q in quotes:
            sym = q.get('symbol')
            if sym:
                symbols.add(sym)
    except KeyError:
        pass
    return symbols

def process_block(search_term):
    """
    Process one search term with block pagination
    """
    yh_all_sym = set()
    for block in range(0, 9999, 100):
        url = f"https://finance.yahoo.com/lookup/equity?s={search_term}&t=A&b={block}&c=100"
        body = call_url(url)
        symbols = extract_symbols_from_body(body)
        if not symbols:
            break
        yh_all_sym.update(symbols)
    return yh_all_sym

def scrape_prefix_batch(prefix_chunk):
    global completed_chunks
    yh_all_sym = set()
    for term_1 in prefix_chunk:
        for term_2 in prefix_chunk:
            search_term = term_1 + term_2
            symbols = process_block(search_term)
            yh_all_sym.update(symbols)

            # Triple-letter expansion if needed
            if len(symbols) >= 100:  # If maxed out, expand
                for term_3 in prefix_chunk:
                    search_term3 = search_term + term_3
                    symbols3 = process_block(search_term3)
                    yh_all_sym.update(symbols3)

                    if len(symbols3) >= 100:
                        for term_4 in prefix_chunk:
                            search_term4 = search_term3 + term_4
                            symbols4 = process_block(search_term4)
                            yh_all_sym.update(symbols4)

    # Update progress
    with lock:
        completed_chunks += 1

    return yh_all_sym

def chunk_list(lst, n):
    k = len(lst) // n
    return [lst[i*k:(i+1)*k] for i in range(n-1)] + [lst[(n-1)*k:]]

@st.cache_data(show_spinner=False)
def threaded_scrape():
    global completed_chunks
    completed_chunks = 0

    search_set = [chr(x) for x in range(65, 91)] + [chr(x) for x in range(48, 58)]
    chunks = chunk_list(search_set, 5)

    all_symbols = set()
    total_chunks = len(chunks)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(scrape_prefix_batch, chunk) for chunk in chunks]

        # Progress bar in main thread
        progress_bar = st.progress(0.0)
        while completed_chunks < total_chunks:
            progress_bar.progress(completed_chunks / total_chunks)
            time.sleep(0.5)

        # Wait for all futures to finish
        for f in as_completed(futures):
            all_symbols.update(f.result())
        progress_bar.progress(1.0)

    return sorted(list(all_symbols))

def main():
    st.title("Yahoo Finance Ticker Scraper (JSON Embedded)")

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
