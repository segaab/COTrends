import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Wells Fargo 10-Q Automation", layout="wide")
st.title("🤖 Automated Wells Fargo 10‑Q Filing Scraper")

# URL source
WFC_FILINGS_URL = "https://www.wellsfargo.com/about/investor-relations/filings/"

@st.cache_data
def fetch_wfc_filings_html():
    res = requests.get(WFC_FILINGS_URL)
    res.raise_for_status()
    return res.text

def parse_wfc_filings_html(html):
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find(string="Q1 2025").find_parent("ul")
    filings = []
    for li in container.find_all("li"):
        txt = li.get_text(separator="|", strip=True).split("|")
        # Expect items like ["Filed April 29, 2025", "Q1 2025 Form 10‑Q (PDF)", ...]
        if any("10-Q" in part for part in txt):
            date_part = txt[0].replace("Filed", "").strip()
            try:
                date = datetime.strptime(date_part, "%B %d, %Y")
            except ValueError:
                continue
            quarter = int(txt[1].split()[0][1])
            # Link extraction for PDF
            link = li.find("a", string=lambda s: s and "Form 10-Q" in s)
            href = link.get("href") if link else ""
            filings.append({"bank": "Wells Fargo", "date": date, "quarter": quarter, "year": date.year, "url": href})
    return pd.DataFrame(filings)

# Fetch and parse
html = fetch_wfc_filings_html()
df = parse_wfc_filings_html(html)

if df.empty:
    st.error("No valid 10-Q entries found on the page.")
else:
    st.header("🔍 Raw HTML Extract")
    st.text_area("Filing HTML snippet", html[:2000]+"...", height=200)
    
    st.header("📄 Parsed 10‑Q Filings Table")
    st.dataframe(df.sort_values("date", ascending=False))
    
    st.header("📈 Filing Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x="date", y="quarter", color="year",
                         title="Filing Dates vs Quarter")
        st.plotly_chart(fig)
    with col2:
        counts = df.groupby(["year", "quarter"]).size().reset_index(name="count")
        fig2 = px.bar(counts, x="quarter", y="count", color="year",
                      title="Counts of 10-Q Filings by Quarter")
        st.plotly_chart(fig2)
    
    st.download_button("📥 Download Filing Metadata", df.to_csv(index=False), "wfc_filings.csv")
