import streamlit as st
import pandas as pd
from sec_api import QueryApi, ExtractorApi

# --- Configuration ---
SEC_API_KEY = "b48426e1ec0d314f153b9d1b9f0421bc1aaa6779d25ea56bfc05bf235393478c"  # Replace with your actual SEC API key

query_api = QueryApi(api_key=SEC_API_KEY)
extractor_api = ExtractorApi(api_key=SEC_API_KEY)

# --- Bank Ticker to CIK Mapping ---
bank_ciks = {
    "JPM": "0000019617",
    "BAC": "0000070858",
    "GS": "0000886982"
}

# --- Functions ---

@st.cache_data
def get_latest_10q_url(cik):
    query = {
        "query": {
            "query_string": {
                "query": f"cik:{cik} AND formType:\"10-Q\""
            }
        },
        "from": "0",
        "size": "1",
        "sort": [{ "filedAt": { "order": "desc" } }]
    }
    response = query_api.get_filings(query)
    try:
        return response["filings"][0]["linkToFilingDetails"]
    except IndexError:
        return None

@st.cache_data
def extract_credit_exposure_paragraph(filing_url):
    try:
        section_text = extractor_api.get_section(filing_url, "part2item7", "text")
        lines = section_text.splitlines()
        credit_related = [line.strip() for line in lines if "credit" in line.lower() or "exposure" in line.lower()]
        return "\n".join(credit_related[:8]) if credit_related else "No credit exposure data found."
    except Exception as e:
        return f"Error extracting data: {str(e)}"

# --- Streamlit UI ---
st.title("📄 Bank Wholesale Credit Exposure Dashboard")

st.markdown("This dashboard extracts **credit exposure mentions** from the latest 10-Q filings (Item 7) for major U.S. banks.")

results = []

for ticker, cik in bank_ciks.items():
    st.subheader(f"🔍 {ticker}")
    filing_url = get_latest_10q_url(cik)
    if filing_url:
        snippet = extract_credit_exposure_paragraph(filing_url)
        st.markdown(f"**Filing URL:** [View Filing]({filing_url})")
        st.code(snippet, language="markdown")
        results.append({"Ticker": ticker, "Snippet": snippet, "URL": filing_url})
    else:
        st.warning("No 10-Q filing found.")

# Optional: Export as DataFrame
if results:
    df = pd.DataFrame(results)
    st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="credit_exposure_snippets.csv")
