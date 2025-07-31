import streamlit as st
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://invest.bnpparibas"
REPORT_PAGE = f"{BASE_URL}/en/document"

def scrape_bnp_reports():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(REPORT_PAGE, headers=headers)

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    reports = []

    for link in soup.find_all("a", href=True):
        text = link.get_text(strip=True)
        href = link["href"]

        if "pillar" in text.lower() or "annual" in text.lower():
            full_link = href if href.startswith("http") else BASE_URL + href
            reports.append((text, full_link))

    return reports

# Streamlit App
st.title("📄 BNP Paribas Report Scraper")
st.write("Fetching the latest sector-related credit reports from BNP Paribas.")

with st.spinner("Scraping..."):
    report_links = scrape_bnp_reports()

if report_links:
    for title, url in report_links:
        st.markdown(f"- [{title}]({url})")
else:
    st.warning("No relevant reports found.")
