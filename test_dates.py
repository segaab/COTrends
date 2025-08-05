from edgar import EdgarClient
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import uuid
import os
import re

# --- Load environment variables ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Initialize Supabase client ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Edgar client ---
client = EdgarClient()

# --- Investment banks to analyze ---
tickers = ['GS', 'MS', 'JPM', 'BAC']

# --- Sector keywords to match in filings ---
sectors = [
    "energy", "real estate", "consumer discretionary", "industrials",
    "financials", "technology", "utilities", "materials", "health care"
]

# --- Helper to extract paragraphs mentioning sector exposure ---
def extract_sector_paragraphs(text, cik, ticker, form_type, filing_date, filing_url):
    results = []
    paragraphs = text.split('\n')
    for para in paragraphs:
        for sector in sectors:
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

# --- Main scraping logic ---
all_results = []

for ticker in tickers:
    filings = client.get_filings(ticker, form='10-Q', count=2)  # You can change to '10-K'
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

# --- Upload to Supabase ---
if all_results:
    response = supabase.table("sector_credit_exposure").insert(all_results).execute()
    print("✅ Inserted records:", len(response.data))
else:
    print("⚠️ No sector credit exposures found in this batch.")
