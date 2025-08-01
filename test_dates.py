import streamlit as st
import requests
import datetime
import pandas as pd

# -------------------------------
# CONFIGURATION
# -------------------------------
HEADERS = {
    "User-Agent": "Tsegaab G segaab120@gmail.com"
}

# Company CIKs — use 10-digit format with leading zeros
BANKS = {
    "JPMorgan Chase": "0000019617",
    "Bank of America": "0000070858",
    "U.S. Bank": "0000036104",
    "Citigroup": "0000831001",
    "PNC Financial": "0000713676",
    "Wells Fargo": "0000072971"
}

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def get_filings(cik: str, form_type="10-Q"):
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        data = res.json()
        filings = data.get("filings", {}).get("recent", {})
        df = pd.DataFrame(filings)
        if not df.empty:
            df = df[df["form"] == form_type]
            df["filingDate"] = pd.to_datetime(df["filingDate"])
            df = df[["filingDate", "form", "accessionNumber", "primaryDocument"]]
            df["fullURL"] = df["accessionNumber"].apply(
                lambda acc: f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{df.loc[df['accessionNumber'] == acc, 'primaryDocument'].values[0]}"
            )
            return df.sort_values("filingDate", ascending=False).head(5)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch filings for CIK {cik}: {e}")
        return pd.DataFrame()

# -------------------------------
# STREAMLIT DASHBOARD
# -------------------------------

st.set_page_config(page_title="EDGAR 10-Q Dashboard", layout="wide")
st.title("📑 SEC EDGAR 10-Q Report Dashboard")

bank_selected = st.selectbox("Choose a bank", list(BANKS.keys()))
cik = BANKS[bank_selected]

st.info(f"Showing latest 10-Q filings for: **{bank_selected}**")

df_filings = get_filings(cik)

if not df_filings.empty:
    df_display = df_filings[["filingDate", "form", "fullURL"]].rename(columns={
        "filingDate": "Filing Date",
        "form": "Form Type",
        "fullURL": "Link to Report"
    })
    st.dataframe(df_display, use_container_width=True)
else:
    st.warning("No 10-Q filings found.")
