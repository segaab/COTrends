# streamlit_app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="Sector Credit Exposure Dashboard", layout="wide")
st.title("📊 Sector Credit Exposure from J.P. Morgan's Latest 10-K Filing")

# 1. Get the latest 10-K filing URL from EDGAR
def get_latest_10k_url(cik="0000019617"):
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers)
    filings = r.json()["filings"]["recent"]
    
    for idx, form in enumerate(filings["form"]):
        if form == "10-K":
            accession = filings["accessionNumber"][idx].replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
            return filing_url
    return None

# 2. Parse the index and get the actual 10-K filing URL
def extract_10k_html_url(index_url):
    r = requests.get(index_url, headers={"User-Agent": "Mozilla/5.0"})
    filing_json = r.json()
    for file in filing_json["directory"]["item"]:
        if file["name"].endswith(".htm") and "10-k" in file["name"].lower():
            base = index_url.rsplit("/", 1)[0]
            return f"{base}/{file['name']}"
    return None

# 3. Scrape the HTML and find the Sector Exposure Table/Text
def get_sector_credit_section(html_url):
    r = requests.get(html_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.content, "html.parser")

    text_blocks = soup.get_text().split("\n")
    matched_lines = []

    for i, line in enumerate(text_blocks):
        if "Wholesale Credit Exposure" in line or "industry exposure" in line:
            matched_lines.extend(text_blocks[i:i+30])  # grab 30 lines below
            break

    return "\n".join(matched_lines) if matched_lines else "Sector credit exposure not found."

# Run all
st.subheader("Fetching from SEC EDGAR...")
index_url = get_latest_10k_url()
if index_url:
    html_url = extract_10k_html_url(index_url)
    if html_url:
        sector_data = get_sector_credit_section(html_url)
        st.code(sector_data, language="text")
    else:
        st.warning("Failed to find 10-K HTML filing.")
else:
    st.warning("Could not locate a recent 10-K filing.")

st.markdown("---")
st.caption("Data fetched live from [SEC.gov](https://www.sec.gov/). Scraper runs browserlessly using `requests`.")
