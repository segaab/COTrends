import streamlit as st
import requests
import datetime
import pandas as pd
import pdfkit
import os
from bs4 import BeautifulSoup

# -------------------------------
# CONFIGURATION
# -------------------------------
HEADERS = {
    "User-Agent": "Tsegaab G segaab120@gmail.com"
}
PDF_OPTIONS = {
    'quiet': '',
    'enable-local-file-access': None,
    'page-size': 'Letter',
    'encoding': 'UTF-8',
}

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
            df["fullURL"] = df.apply(
                lambda row: f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{row['accessionNumber'].replace('-', '')}/{row['primaryDocument']}", axis=1
            )
            return df.sort_values("filingDate", ascending=False).head(5)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch filings for CIK {cik}: {e}")
        return pd.DataFrame()

def download_and_export(link: str, export_type: str, filename: str):
    try:
        response = requests.get(link, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        clean_html = soup.prettify()

        if export_type == "PDF":
            pdfkit.from_string(clean_html, f"{filename}.pdf", options=PDF_OPTIONS)
            return f"{filename}.pdf"
        elif export_type == "JSON":
            with open(f"{filename}.json", "w", encoding="utf-8") as f:
                f.write(response.text)
            return f"{filename}.json"
    except Exception as e:
        st.error(f"Error downloading {link}: {e}")
        return None

# -------------------------------
# STREAMLIT APP
# -------------------------------

st.set_page_config(page_title="EDGAR 10-Q Dashboard", layout="wide")
st.title("📑 SEC EDGAR 10-Q Report Dashboard + Exporter")

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

    st.subheader("📤 Export Options")
    export_type = st.radio("Choose export type:", ["PDF", "JSON"])
    export_indices = st.multiselect("Select rows to export", df_display.index)

    if st.button("Download Selected"):
        for idx in export_indices:
            row = df_filings.loc[idx]
            date_str = row["filingDate"].strftime("%Y%m%d")
            filename = f"{bank_selected.replace(' ', '_')}_{date_str}"
            file_path = download_and_export(row["fullURL"], export_type, filename)
            if file_path:
                with open(file_path, "rb") as file:
                    st.download_button(
                        label=f"📎 Download {filename}.{export_type.lower()}",
                        data=file,
                        file_name=f"{filename}.{export_type.lower()}",
                        mime="application/pdf" if export_type == "PDF" else "application/json"
                    )
else:
    st.warning("No 10-Q filings found.")
