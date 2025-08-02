import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st

# Test: Use a specific 10-Q HTML report (replace with dynamic URL later)
report_url = "https://www.sec.gov/Archives/edgar/data/19617/000001961724000117/jpm-20240630.htm"

st.title("📊 Sector Loan Breakdown (Scraped from 10-Q)")
st.write(f"Source: [10-Q Filing]({report_url})")

# Get HTML content
res = requests.get(report_url, headers={"User-Agent": "segaab120@gmail.com"})
soup = BeautifulSoup(res.text, "html.parser")

# Look for tables with keywords in headers
tables = soup.find_all("table")

found_table = None
for table in tables:
    if "Loan" in table.text and ("Commercial" in table.text or "Real Estate" in table.text):
        found_table = table
        break

# Extract rows
if found_table:
    rows = []
    for row in found_table.find_all("tr"):
        cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
        if cols:
            rows.append(cols)

    df = pd.DataFrame(rows[1:], columns=rows[0])
    st.subheader("📋 Sector Loan Table")
    st.dataframe(df)
else:
    st.warning("⚠️ No loan sector table found in this 10-Q.")
