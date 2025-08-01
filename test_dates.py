import streamlit as st
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="Bank 10-Q Dashboard", layout="wide")
st.title("📄 Bank 10-Q Report Fetcher with Pagination")

BANK_SOURCES = {
    "J.P. Morgan": "https://jpmorganchaseco.gcs-web.com/ir/sec-other-filings/overview",
    "BoA": "https://www.sec.gov/Archives/edgar/data/70858/000119312506105418/d10q.htm",
    "U.S. Bank": "https://ir.usbank.com/financials/sec-filings/default.aspx",
    "Citibank": "https://www.citigroup.com/global/investors/sec-filings",
    "PNC": "https://investor.pnc.com/sec-filings/all-sec-filings?form_type=10-Q&year=",
    "Wells Fargo": "https://www.wellsfargo.com/about/investor-relations/filings/"
}


def fetch_10q_links(url, max_pages=5):
    links = []
    current_url = url
    for page_num in range(max_pages):
        try:
            res = requests.get(current_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")
            a_tags = soup.find_all("a", href=True)
            found_new = False
            for tag in a_tags:
                href = tag["href"]
                text = tag.get_text(strip=True).lower()
                if "10-q" in href.lower() or "10-q" in text:
                    full_url = href if href.startswith("http") else requests.compat.urljoin(current_url, href)
                    if full_url not in links:
                        links.append(full_url)
                        found_new = True

            # Try to find a next page link — limit by max_pages
            next_link = soup.find("a", string=re.compile("next", re.IGNORECASE))
            if next_link and "href" in next_link.attrs:
                next_href = next_link["href"]
                current_url = requests.compat.urljoin(current_url, next_href)
            else:
                break  # No next page found
        except Exception as e:
            st.error(f"[Error on page {page_num+1}] {e}")
            break
    return links

for bank, url in BANK_SOURCES.items():
    st.subheader(bank)
    filings = fetch_10q_links(url, max_pages=5)
    if filings:
        for link in filings:
            st.markdown(f"[📄 View 10-Q]({link})")
    else:
        st.warning("No 10-Q filings found or website structure unsupported.")
