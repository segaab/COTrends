# streamlit_app.py

import streamlit as st
import requests
import time
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
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
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
    rest = body[count_beg+8: count_beg+20]
    count_end = rest.find(')')
    return int(rest[0: count_end])

def process_block(srch, yh_all_sym, hdr):
    for block in range(0, 9999, 100):
        url = f"https://finance.yahoo.com/lookup/equity?s={srch}&t=A&b={block}&c=100"
        body = call_url(url, hdr)
        soup = BeautifulSoup(body, 'html.parser')
        links = soup.find_all('a')
        is_empty = True
        for link in links:
            if "/quote/" in link.get('href'):
                symbol = link.get('data-symbol')
                if symbol:
                    is_empty = False
                    yh_all_sym.add(symbol)
        if is_empty:
            break

@st.cache_data(show_spinner=True)
def scrape_all_tickers():
    search_set = [chr(x) for x in range(65, 91)] + [chr(x) for x in range(48, 58)]
    yh_all_sym = set()

    for t1 in search_set:
        for t2 in search_set:
            term = t1 + t2
            url = f"https://finance.yahoo.com/lookup/equity?s={term}&t=A&b=0&c=25"
            hdr["path"] = url
            body = call_url(url, hdr)
            total = get_counts(body)

            if total < 9000:
                process_block(term, yh_all_sym, hdr)
            else:
                for t3 in search_set:
                    term3 = term + t3
                    url = f"https://finance.yahoo.com/lookup/equity?s={term3}&t=A&b=0&c=25"
                    hdr["path"] = url
                    body = call_url(url, hdr)
                    cnt3 = get_counts(body)

                    if cnt3 < 9000:
                        process_block(term3, yh_all_sym, hdr)
                    else:
                        for t4 in search_set:
                            process_block(term3 + t4, yh_all_sym, hdr)

    return sorted(list(yh_all_sym))

def main():
    st.title("Yahoo Finance Ticker Scraper")

    if st.button("Start Scraping"):
        tickers = scrape_all_tickers()
        st.success(f"Scraping complete – found {len(tickers)} symbols.")
        st.dataframe(tickers)

        # Convert to text for download
        text_content = "\n".join(tickers)
        st.download_button(
            label="Download as .txt file",
            data=text_content,
            file_name="yh_all_symbols.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
