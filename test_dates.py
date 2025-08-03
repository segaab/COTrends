import streamlit as st
import pandas as pd
from sec_api import QueryApi, ExtractorApi

SEC_API_KEY = "b48426e1ec0d314f153b9d1b9f0421bc1aaa6779d25ea56bfc05bf235393478c"  # Replace this
query_api = QueryApi(api_key=SEC_API_KEY)
extractor_api = ExtractorApi(api_key=SEC_API_KEY)

st.set_page_config(page_title="Sector Dashboard", layout="wide")
st.title("🏦 Sector Dashboard: Credit Exposure (Item 2 - 10-Q)")
st.markdown("Extracts **credit exposure** references from Item 2 of latest 10-Q filings for selected U.S. banks.")

# List of banks
bank_info = {
    "JPM": {"cik": "19617"},
    "BAC": {"cik": "70858"},
    "GS": {"cik": "886982"},
    "MS": {"cik": "895421"},
    "WFC": {"cik": "72971"},
}

@st.cache_data
def get_latest_10q_url(cik):
    query = {
        "query": {
            "query_string": {
                "query": f'cik:{cik} AND formType:"10-Q"'
            }
        },
        "from": "0",
        "size": "10",
        "sort": [{"filedAt": {"order": "desc"}}]
    }
    result = query_api.get_filings(query)
    filings = result.get("filings", [])
    
    for filing in filings:
        form_type = filing.get("formType", "").upper()
        if form_type == "10-Q" and filing.get("filingDate", "").startswith(("2025", "2024")):
            return filing["linkToFilingDetails"]
    return None

def extract_item2_credit_mentions(filing_url):
    try:
        section = extractor_api.get_section(filing_url, "part1item2", "text")
        lines = [line.strip() for line in section.splitlines()]
        credit_lines = [line for line in lines if any(word in line.lower() for word in ["credit", "exposure", "lending", "default", "counterparty"])]
        return "\n".join(credit_lines[:12]) if credit_lines else "No credit exposure references found."
    except Exception as e:
        return f"⚠️ Extraction failed: {e}"

# Collect records
records = []

for ticker, info in bank_info.items():
    st.subheader(f"🔍 {ticker}")
    filing_url = get_latest_10q_url(info["cik"])
    
    if filing_url:
        st.markdown(f"[📄 View Filing →]({filing_url})")
        content = extract_item2_credit_mentions(filing_url)
        st.code(content, language="text")
    else:
        content = "⚠️ No recent 10-Q filing found for 2024 or 2025."
        st.warning(content)
    
    records.append({
        "Ticker": ticker,
        "Filing URL": filing_url or "N/A",
        "Summary": content[:500] + "..." if len(content) > 500 else content
    })

# Download results
df = pd.DataFrame(records)
st.download_button("📥 Download Extracted Data", df.to_csv(index=False), "sector_credit_summary.csv")
