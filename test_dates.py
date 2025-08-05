import streamlit as st
from edgar import EdgarClient
from supabase import create_client, Client
from dotenv import load_dotenv
import pandas as pd
import uuid
import os
import re

# --- Load environment variables ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Initialize Supabase ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Sector keywords to detect ---
SECTORS = [
    "energy", "real estate", "consumer discretionary", "industrials",
    "financials", "technology", "utilities", "materials", "health care"
]

# --- Streamlit App ---
st.set_page_config(page_title="Sector Credit Exposure Scraper", layout="wide")
st.title("📊 Sector Credit Exposure from SEC Filings")

# --- Sidebar inputs ---
tickers = st.sidebar.multiselect("Select tickers (investment banks):", ["GS", "MS", "JPM", "BAC"], default=["GS", "JPM"])
filing_type = st.sidebar.selectbox("Filing type:", ["10-Q", "10-K"])
max_filings = st.sidebar.slider("Number of filings per ticker:", 1, 5, 2)
enable_upload = st.sidebar.checkbox("Upload to Supabase", value=False)

# --- Edgar client ---
client = EdgarClient()

# --- Extract credit exposure paragraphs ---
@st.cache_data(show_spinner=False)
def extract_sector_paragraphs(text, cik, ticker, form_type, filing_date, filing_url):
    results = []
    paragraphs = text.split('\n')
    for para in paragraphs:
        for sector in SECTORS:
            if sector in para.lower() and re.search(r'\$[\d,.]+', para):
                results.append({
                    "id": str(uuid.uuid4()),
                    "cik": cik,
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "form_type": form_type,
                    "sector": sector.title(),
                    "exposure_text": para.strip(),
                    "source_url": filing_url
                })
    return results

# --- Scrape filings and show results ---
if st.button("🔍 Run Scraper"):
    all_results = []

    with st.spinner("Fetching and parsing filings..."):
        for ticker in tickers:
            try:
                filings = client.get_filings(ticker, form=filing_type, count=max_filings)
                for filing in filings:
                    text = client.get_filing_text(filing.accession_no)
                    if text:
                        parsed = extract_sector_paragraphs(
                            text,
                            cik=filing.cik,
                            ticker=ticker,
                            form_type=filing.form,
                            filing_date=filing.filing_date,
                            filing_url=filing.primary_document_url
                        )
                        all_results.extend(parsed)
            except Exception as e:
                st.error(f"❌ Error loading filings for {ticker}: {e}")

    # Display results
    if all_results:
        df = pd.DataFrame(all_results)
        st.success(f"✅ Found {len(df)} sector exposure entries")
        st.dataframe(df)

        # Upload to Supabase
        if enable_upload and st.button("⬆️ Upload to Supabase"):
            with st.spinner("Uploading..."):
                response = supabase.table("sector_credit_exposure").insert(all_results).execute()
                if response.data:
                    st.success("✅ Upload complete!")
                else:
                    st.error("⚠️ Upload failed or no data returned.")
    else:
        st.warning("No exposures found in the selected filings.")
