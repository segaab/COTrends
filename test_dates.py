import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import plotly.express as px
import re

st.set_page_config(page_title="Wells Fargo 10-Q Filings", layout="wide")
st.title("📄 Wells Fargo 10-Q Filing Dashboard")

# URL for Wells Fargo Filings
WFC_FILINGS_URL = "https://www.wellsfargo.com/about/investor-relations/filings/"

@st.cache_data(show_spinner=True)
def fetch_wfc_html():
    res = requests.get(WFC_FILINGS_URL)
    res.raise_for_status()
    return res.text

def parse_wfc_filings_html(html):
    soup = BeautifulSoup(html, "html.parser")
    filings = []

    # Find all anchor tags that mention "10-Q"
    for link in soup.find_all("a", href=True):
        text = link.get_text(strip=True)
        href = link["href"]

        if "10-Q" in text and "Form" in text:
            parent_li = link.find_parent("li")
            if not parent_li:
                continue

            raw_text = parent_li.get_text(" ", strip=True)
            date_match = re.search(r"Filed\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", raw_text)
            quarter_match = re.search(r"Q([1-4])\s+(\d{4})", raw_text)

            try:
                if date_match:
                    date = datetime.strptime(date_match.group(1), "%B %d, %Y")
                else:
                    continue

                quarter = int(quarter_match.group(1)) if quarter_match else (date.month - 1) // 3 + 1

                filings.append({
                    "bank": "Wells Fargo",
                    "date": date,
                    "quarter": quarter,
                    "year": date.year,
                    "url": href
                })
            except Exception as e:
                print("Error parsing filing:", e)
                continue

    return pd.DataFrame(filings)

# Main execution
html = fetch_wfc_html()
df = parse_wfc_filings_html(html)

# Display HTML debug
with st.expander("🔍 Show Raw HTML"):
    st.text_area("HTML Content", html[:3000] + "...", height=300)

if df.empty:
    st.warning("No 10-Q filings found.")
else:
    st.header("🧾 Parsed 10-Q Filings Table")
    st.dataframe(df.sort_values("date", ascending=False))

    st.header("📊 Filing Analytics")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x="date", y="quarter", color="year", title="10-Q Filing Dates by Quarter")
        st.plotly_chart(fig)

    with col2:
        count_df = df.groupby(["year", "quarter"]).size().reset_index(name="count")
        fig2 = px.bar(count_df, x="quarter", y="count", color="year", title="Filing Count by Quarter & Year")
        st.plotly_chart(fig2)

    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", data=csv, file_name="wfc_10q_filings.csv")
