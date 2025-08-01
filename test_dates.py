import streamlit as st
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(page_title="EDGAR 10-Q Scraper", layout="wide")
st.title("📄 EDGAR Bank 10-Q Scraper")

BANK_CIKS = {
    "JPMorgan Chase": "19617",
    "Bank of America": "70858",
    "U.S. Bancorp": "36104",
    "Citigroup": "831001",
    "PNC Financial": "713676",
    "Wells Fargo": "72971",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

BASE_EDGAR_URL = "https://data.sec.gov/submissions/CIK{}.json"
FILINGS_URL = "https://www.sec.gov/Archives/{}"

def get_latest_10q_urls(cik, count=5):
    time.sleep(0.5)  # polite delay
    url = BASE_EDGAR_URL.format(cik.zfill(10))
    try:
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            st.warning(f"Failed to retrieve CIK {cik} data.")
            return []
        data = res.json()
        filings = data.get("filings", {}).get("recent", {})
        urls = []
        for i, form in enumerate(filings.get("form", [])):
            if form == "10-Q" and len(urls) < count:
                acc_no = filings["accessionNumber"][i].replace("-", "")
                filing_href = f"/Archives/edgar/data/{int(cik)}/{acc_no}/{filings['primaryDocument'][i]}"
                full_url = FILINGS_URL.format(filing_href.lstrip("/"))
                urls.append(full_url)
        return urls
    except Exception as e:
        st.error(f"Error: {e}")
        return []

st.subheader("Scrape 10-Q Filings via EDGAR")
for bank, cik in BANK_CIKS.items():
    st.markdown(f"### {bank}")
    urls = get_latest_10q_urls(cik)
    if urls:
        for u in urls:
            st.write(u)
    else:
        st.warning("No 10-Q filings found.")
                           
