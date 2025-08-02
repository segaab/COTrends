import streamlit as st
import requests
from bs4 import BeautifulSoup
import json

# 🔐 PDF.co API key
PDFCO_API_KEY = "segaab120@gmail.com_nToGFafi2kw9Nlx5JdgtvEoDjqw06HzKczviZvlZ4V7OXv7eN1ZS6eIUIVV2JOQi"
PDFCO_HEADERS = {
    "x-api-key": PDFCO_API_KEY,
    "Content-Type": "application/json"
}

# Wells Fargo 10-Q Reports page
REPORTS_PAGE_URL = "https://www.wellsfargo.com/about/investor-relations/filings/"

# Streamlit page setup
st.set_page_config(page_title="Wells Fargo 10-Q Parser", layout="wide")
st.title("📄 Wells Fargo 10-Q Report Parser")

# ⛏️ Scrape 10-Q report links
@st.cache_data(show_spinner=False)
def fetch_10q_links():
    res = requests.get(REPORTS_PAGE_URL, headers={"User-Agent": "segaab120@gmail.com"})
    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.find_all("a", href=True)

    ten_q_links = []
    for link in links:
        href = link['href']
        if "10-q" in href.lower() or "10q" in link.text.lower():
            full_url = href if href.startswith("http") else "https://www.wellsfargo.com" + href
            ten_q_links.append({"text": link.text.strip(), "url": full_url})

    return ten_q_links

# Display links and UI
ten_q_reports = fetch_10q_links()
st.subheader("📁 Available 10-Q Reports")

if ten_q_reports:
    report_titles = [f"{r['text']}" for r in ten_q_reports]
    selected = st.selectbox("Select a report to send to PDF.co API:", options=report_titles)
    selected_url = next((r['url'] for r in ten_q_reports if r['text'] == selected), None)

    st.markdown(f"🔗 [View Selected Report]({selected_url})")

    if st.button("📤 Send to PDF.co API"):
        with st.spinner("Sending to PDF.co Document Parser..."):
            payload = {
                "url": selected_url,
                "inline": True
            }
            try:
                response = requests.post(
                    "https://api.pdf.co/v1/pdf/convert/to/text",
                    headers=PDFCO_HEADERS,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                parsed = response.json()
                if parsed.get("body"):
                    st.subheader("📄 Parsed Text Preview")
                    st.text_area("Extracted Content:", parsed["body"][:3000], height=300)
                else:
                    st.warning("⚠️ No content returned by PDF.co.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
else:
    st.warning("⚠️ No 10-Q links found on the Wells Fargo filings page.")
