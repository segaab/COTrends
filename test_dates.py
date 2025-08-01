import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Define headers for EDGAR API requests (per SEC guidelines)
headers = {
    "User-Agent": "Bank10QFetcher/1.0 (contact: segaab120@gmail.com; author: Tsegaab G)"
}

# List of banks and their corresponding 10-digit CIKs
banks = {
    "JPMorgan Chase": "0000019617",
    "Bank of America": "0000070858",
    "Citigroup": "0000831001",
    "Goldman Sachs": "0000886982",
    "Morgan Stanley": "0000895421"
}

# Streamlit UI
st.set_page_config(page_title="EDGAR 10-Q Dashboard", layout="wide")
st.title("📄 EDGAR 10-Q Filings Fetcher")

bank_choice = st.selectbox("Select a bank:", list(banks.keys()))

# Fetch EDGAR submissions for selected bank
cik = banks[bank_choice]
submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"

try:
    response = requests.get(submissions_url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()

    # Filter for 10-Q filings only
    filings = data.get("filings", {}).get("recent", {})
    if not filings:
        st.warning("No filings found in EDGAR data.")
    else:
        df = pd.DataFrame(filings)
        df = df[df["form"] == "10-Q"]

        # Prepare and display the latest 5 10-Q filings
        df_result = pd.DataFrame({
            "Accession Number": df["accessionNumber"].head(5),
            "Filing Date": df["filingDate"].head(5),
            "Report URL": [f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm" for acc in df["accessionNumber"].head(5)]
        })

        st.subheader(f"Latest 10-Q Filings for {bank_choice}")
        st.dataframe(df_result, use_container_width=True)
except requests.exceptions.RequestException as e:
    st.error(f"Error fetching data: {e}")
