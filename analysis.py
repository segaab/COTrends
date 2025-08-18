# streamlit_app_threaded_progress.py

import streamlit as st
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# Shared progress counter
completed_chunks = 0
lock = threading.Lock()

hdr = {
    "authority": "finance.yahoo.com",
    "method": "GET",
    "scheme": "https",
    "accept": "text/html",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "dnt": "1",
    "pragma": "no-cache",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
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

def get_counts(body, srch):
    count_beg = body.find('Stocks (')
    rest = body[count_beg+8: count_beg+20]
    count_end = rest.find(')')
    return int(rest[:count_end])

def process_block(body, srch, yh_all_sym):
    for block in range(0, 9999, 100):
        url = f"https://finance.yahoo.com/lookup/equity?s={srch}&t=A&b={block}&c=100"
        body = call_url(url)
        soup = BeautifulSoup(body, 'html.parser')
        links = soup.find_all('a')
        is_empty = True
        for link in links:
            if "/quote/" in link.get('href', ''):
                symbol = link.get('data-symbol')
                if symbol:
                    is_empty = False
                    yh_all_sym.add(symbol)
        if is_empty:
            break

def scrape_prefix_batch(search_set_chunk):
    global completed_chunks
    yh_all_sym = set()
    for term_1 in search_set_chunk:
        for term_2 in search_set_chunk:
            search_term = term_1 + term_2
            url = f"https://finance.yahoo.com/lookup/equity?s={search_term}&t=A&b=0&c=25"
            body = call_url(url)
            all_num = get_counts(body, search_term)

            if all_num < 9000:
                process_block(body, search_term, yh_all_sym)
            else:
                for term_3 in search_set_chunk:
                    search_term3 = search_term + term_3
                    url3 = f"https://finance.yahoo.com/lookup/equity?s={search_term3}&t=A&b=0&c=25"
                    body3 = call_url(url3)
                    all_num3 = get_counts(body3, search_term3)

                    if all_num3 < 9000:
                        process_block(body3, search_term3, yh_all_sym)
                    else:
                        for term_4 in search_set_chunk:
                            process_block(body3, search_term3 + term_4, yh_all_sym)

    # Update progress safely
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

    yh_all_symbols = set()
    total_chunks = len(chunks)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(scrape_prefix_batch, chunk) for chunk in chunks]

        # Update progress bar in main thread
        progress_bar = st.progress(0.0)
        while completed_chunks < total_chunks:
            progress_bar.progress(completed_chunks / total_chunks)
            time.sleep(1)  # check every 1 sec
        # Ensure all threads are done
        for f in as_completed(futures):
            yh_all_symbols.update(f.result())
        progress_bar.progress(1.0)

    return sorted(list(yh_all_symbols))

def main():
    st.title("Yahoo Finance Ticker Scraper (Threaded + Progress)")

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
