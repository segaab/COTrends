import streamlit as st
import pandas as pd
from sec_api import QueryApi, ExtractorApi

# --- Setup ---
SEC_API_KEY = "b48426e1ec0d314f153b9d1b9f0421bc1aaa6779d25ea56bfc05bf235393478c"  # Replace this with your actual key
query_api = QueryApi(api_key=SEC_API_KEY)
extractor_api = ExtractorApi(api_key=SEC_API_KEY)

# --- CIK Lookup for Banks ---
bank_ciks = {
    "JPM": "19617",
    "BAC": "70858",
    "GS": "886982"
}

# --- Streamlit UI ---
st.set_page_config(page_title="Sector Dashboard", layout="wide")
st.title("🏦 Sector Dashboard: Bank Credit Exposure")
st.markdown("Extracts **credit exposure** mentions from latest **10-Q filings** – *Item 2: Management’s Discussion and Analysis*.")

@st.cache_data
def get_latest_10q_url(cik):
    query = {
        "query": {"query_string": {"query": f"cik:{cik} AND formType:\"10-Q\""}},
        "from": "0", "size": "5",
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    response = query_api.get_filings(query)
    filings = response.get("filings", [])
    urls = [f["linkToFilingDetails"] for f in filings if f.get("filingDate", "").startswith(("2025", "2024"))]
    return urls[0] if urls else None

@st.cache_data
def extract_credit_exposure(filing_url):
    try:
        section_text = extractor_api.get_section(filing_url, "part1item2", "text")
        lines = [line.strip() for line in section_text.splitlines()]
        credit_mentions = [line for line in lines if "credit" in line.lower() or "exposure" in line.lower()]
        return "\n".join(credit_mentions[:10]) if credit_mentions else "No relevant mentions found."
    except Exception as e:
        return f"Error extracting: {str(e)}"

# --- Run Extraction ---
records = []
for ticker, cik in bank_ciks.items():
    st.subheader(f"🔍 {ticker}")
    filing_url = get_latest_10q_url(cik)
    if filing_url:
        st.markdown(f"[Open Filing →]({filing_url})")
        snippet = extract_credit_exposure(filing_url)
        st.code(snippet, language="markdown")
        records.append({
            "Ticker": ticker,
            "Filing URL": filing_url,
            "Credit Exposure": snippet
        })
    else:
        st.warning("⚠️ No 10-Q filing found for 2024 or 2025.")
        records.append({
            "Ticker": ticker,
            "Filing URL": "Not found",
            "Credit Exposure": "No filing data available."
        })

# --- Optional: Download CSV ---
if records:
    df = pd.DataFrame(records)
    st.download_button("📥 Download Data as CSV", df.to_csv(index=False), file_name="credit_exposure.csv")
