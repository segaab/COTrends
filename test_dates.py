import requests
import pandas as pd
from bs4 import BeautifulSoup
import streamlit as st

st.set_page_config(page_title="Sector Loan Table from Latest 10-Q", layout="wide")
st.title("📊 Sector-Specific Loans from Latest JPMorgan 10-Q")

CIK = "0000019617"
SEC_HEADERS = {"User-Agent": "segaab120@gmail.com"}

def get_latest_10q_url(cik):
    base_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(base_url, headers=SEC_HEADERS)
    res.raise_for_status()
    data = res.json()
    recent = data.get("filings", {}).get("recent", {})
    for i, form in enumerate(recent.get("form", [])):
        if form == "10-Q":
            accession = recent["accessionNumber"][i].replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json"
            return doc_url
    return None

def get_html_filing_url(index_json_url):
    res = requests.get(index_json_url, headers=SEC_HEADERS)
    res.raise_for_status()
    files = res.json().get("directory", {}).get("item", [])
    for file in files:
        name = file["name"]
        if name.endswith(".htm") or name.endswith(".html"):
            return index_json_url.rsplit("/", 1)[0] + "/" + name
    return None

def extract_sector_table(url):
    res = requests.get(url, headers=SEC_HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")
    tables = soup.find_all("table")
    for table in tables:
        if "Loan" in table.text and any(x in table.text for x in ["Commercial", "Real Estate", "Consumer"]):
            rows = []
            for row in table.find_all("tr"):
                cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
                if cols:
                    rows.append(cols)
            if len(rows) > 1:
                return pd.DataFrame(rows[1:], columns=rows[0])
    return None

# Run
index_json_url = get_latest_10q_url(CIK)
if index_json_url:
    filing_html_url = get_html_filing_url(index_json_url)
    if filing_html_url:
        st.markdown(f"📄 [View Filing HTML]({filing_html_url})")
        sector_df = extract_sector_table(filing_html_url)
        if sector_df is not None:
            st.subheader("📋 Sector Loan Table")
            st.dataframe(sector_df)
        else:
            st.warning("⚠️ No sector loan breakdown table found.")
    else:
        st.error("❌ Could not locate HTML document in filing.")
else:
    st.error("❌ No 10-Q filing found.")
