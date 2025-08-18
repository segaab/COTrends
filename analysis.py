# streamlit_app.py

import streamlit as st
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

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

def call_url(url, hdr):
    confirmed = False
    while not confirmed:
        try:
            r = requests.get(url, headers=hdr)
            r.raise_for_status()
            confirmed = True
        except Exception:
            time.sleep(1)
    return r.text

def get_counts(body):
    count_beg = body.find('Stocks (')
    rest = body[count_beg+8 : count_beg+20]
    count_end = rest.find(')')
    return int(rest[:count_end])

def process_block(term, sym_set):
    for block in range(0, 9999, 100):
        url = f"https://finance.yahoo.com/lookup/equity?s={term}&t=A&b={block}&c=100"
        body = call_url(url, hdr)
        soup  = BeautifulSoup(body, "html.parser")
        links = soup.find_all("a")
        is_empty = True
        for link in links:
            if "/quote/" in link.get("href", ""):
                symbol = link.get("data-symbol")
                if symbol:
                    sym_set.add(symbol)
                    is_empty = False
        if is_empty:
            break

def scrape_prefix_range(prefix_list, progress_callback=None):
    yh_all_sym = set()
    for p1 in prefix_list:
        for p2 in prefix_list:
            term = p1 + p2
            url  = f"https://finance.yahoo.com/lookup/equity?s={term}&t=A&b=0&c=25"
            body = call_url(url, hdr)
            total = get_counts(body)

            if total < 9000:
                process_block(term, yh_all_sym)
            else:
                for p3 in prefix_list:
                    term3 = term + p3
                    url   = f"https://finance.yahoo.com/lookup/equity?s={term3}&t=A&b=0&c=25"
                    body  = call_url(url, hdr)
                    cnt3  = get_counts(body)

                    if cnt3 < 9000:
                        process_block(term3, yh_all_sym)
                    else:
                        for p4 in prefix_list:
                            process_block(term3 + p4, yh_all_sym)

    if progress_callback:
        progress_callback()
    return yh_all_sym

def chunk_list(lst, n_chunks):
    k = len(lst) // n_chunks
    return [lst[i*k:(i+1)*k] for i in range(n_chunks-1)] + [lst[(n_chunks-1)*k:]]

@st.cache_data(show_spinner=False)
def threaded_scrape():
    search_set = [chr(x) for x in range(65, 91)] + [chr(x) for x in range(48, 58)]
    chunks = chunk_list(search_set, 5)

    yh_all_symbols = set()
    progress = st.progress(0.0)

    completed = 0

    def update_progress():
        nonlocal completed
        completed += 1
        progress.progress(completed / 5.0)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(scrape_prefix_range, chunk, update_progress)
            for chunk in chunks
        ]
        for f in as_completed(futures):
            yh_all_symbols.update(f.result())

    return sorted(list(yh_all_symbols))

def main():
    st.title("Yahoo Finance Ticker Scraper (Threaded)")

    if st.button("Start Scraping"):
        st.write("Scraping in progress...")
        tickers = threaded_scrape()
        st.success(f"Done! Found {len(tickers)} tickers.")

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
