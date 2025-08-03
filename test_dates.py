import streamlit as st
import pandas as pd
from sec_api import QueryApi, ExtractorApi

SEC_API_KEY = "b48426e1ec0d314f153b9d1b9f0421bc1aaa6779d25ea56bfc05bf235393478c"
query_api = QueryApi(api_key=SEC_API_KEY)
extractor_api = ExtractorApi(api_key=SEC_API_KEY)

# CIK and fallback URLs
bank_info = {
    "JPM": {
        "cik": "19617",
        "fallback_url": "https://www.sec.gov/Archives/edgar/data/19617/000001961724000555/jpm-20241011.htm"
    },
    "BAC": {
        "cik": "70858",
        "fallback_url": "https://investor.bankofamerica.com/regulatory-and-other-filings/all-sec-filings/content/0000070858-25-000139/0000070858-25-000139.pdf"
    },
    "GS": {
        "cik": "886982",
        "fallback_url": "https://www.sec.gov/Archives/edgar/data/886982/000088698225000005/gs-20240331.htm"
    }
}

st.set_page_config(page_title="Sector Dashboard", layout="wide")
st.title("🏦 Sector Dashboard: Credit Exposure (Item 2)")
st.markdown("Extracts **credit exposure** content from Item 2 of the latest 10-Q filings for major U.S. banks.")

@st.cache_data
def get_10q_filing_url(cik):
    query = {
        "query": {"query_string": {"query": f"cik:{cik} AND formType:\"10-Q\""}},
        "from": "0", "size": "10",
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    result = query_api.get_filings(query)
    filings = result.get("filings", [])
    for filing in filings:
        if filing.get("filingDate", "").startswith(("2025", "2024")):
            return filing["linkToFilingDetails"]
    return None

def extract_credit_exposure(filing_url, ticker):
    try:
        section = extractor_api.get_section(filing_url, "part1item2", "text")
        lines = [line.strip() for line in section.splitlines()]
        filtered = [line for line in lines if "credit" in line.lower() or "exposure" in line.lower()]
        return "\n".join(filtered[:10]) if filtered else "No credit exposure references found."
    except Exception as e:
        return f"⚠️ Error extracting: {str(e)}"

# Display dashboard
records = []
for ticker, info in bank_info.items():
    st.subheader(f"🔍 {ticker}")
    filing_url = get_10q_filing_url(info["cik"])
    fallback_used = False

    if not filing_url:
        filing_url = info["fallback_url"]
        fallback_used = True

    st.markdown(f"[Open Filing →]({filing_url})")
    text = extract_credit_exposure(filing_url, ticker)
    st.code(text, language="markdown")
    if fallback_used:
        st.warning("⚠️ Using manually sourced fallback URL.")

    records.append({
        "Ticker": ticker,
        "Filing URL": filing_url,
        "Credit Exposure": text[:500] + "..." if len(text) > 500 else text
    })

# Download
df = pd.DataFrame(records)
st.download_button("📥 Download Data", df.to_csv(index=False), "sector_credit_exposure.csv")
