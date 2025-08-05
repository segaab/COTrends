from edgar import EdgarClient
import re
import pandas as pd
from supabase import create_client, Client
from datetime import datetime
import uuid

# --- Supabase setup ---
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-or-service-role-key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Edgar client ---
client = EdgarClient()

# --- Investment banks to analyze ---
tickers = ['GS', 'MS', 'JPM', 'BAC']

# --- Sector keywords (extendable) ---
sectors = [
    "energy", "real estate", "consumer discretionary", "industrials",
    "financials", "technology", "utilities", "materials", "health care"
]

# --- Helper to extract paragraphs containing sector exposure ---
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
    filings = client.get_filings(ticker, form='10-Q', count=2)  # Latest 2 filings per bank
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
    print("Inserted:", response.data)
else:
    print("No relevant exposures found.")
