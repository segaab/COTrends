import streamlit as st
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="Bank 10-Q Dashboard", layout="wide")
st.title("📄 Bank 10-Q Report Fetcher with Pagination")

BANK_URLS = {
    "J.P. Morgan": "https://jpmorganchaseco.gcs-web.com/ir/sec-other-filings/overview",
    "Bank of America": "https://www.sec.gov/Archives/edgar/data/70858/000119312506105418/d10q.htm",
    "U.S. Bank": "https://ir.usbank.com/financials/sec-filings/default.aspx",
    "Citibank": "https://www.citigroup.com/global/investors/sec-filings",
    "PNC": "https://investor.pnc.com/sec-filings/all-sec-filings?form_type=10-Q&year=",
    "Wells Fargo": "https://www.wellsfargo.com/about/investor-relations/filings/"
}

def get_10q_links_jpm():
    # Static fallback, JPM requires JS rendering
    return ["https://www.sec.gov/ix?doc=/Archives/edgar/data/19617/000001961724000032/jpm-20240630.htm"]

def get_10q_links_boa():
    return ["https://www.sec.gov/Archives/edgar/data/70858/000119312506105418/d10q.htm"]

def get_10q_links_usbank():
    base_url = "https://ir.usbank.com/financials/sec-filings/default.aspx"
    response = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        if "10-q" in a.text.lower():
            link = a["href"]
            if not link.startswith("http"):
                link = "https://ir.usbank.com" + link
            links.append(link)
        if len(links) >= 5:
            break
    return links

def get_10q_links_citi():
    base_url = "https://www.citigroup.com/global/investors/sec-filings"
    links = []
    page = 1
    while len(links) < 5:
        resp = requests.get(f"{base_url}?page={page}", headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            if "10-q" in a.text.lower():
                link = a["href"]
                if not link.startswith("http"):
                    link = "https://www.citigroup.com" + link
                links.append(link)
            if len(links) >= 5:
                break
        page += 1
        if page > 10:
            break
    return links

def get_10q_links_pnc():
    base_url = "https://investor.pnc.com/sec-filings/all-sec-filings?form_type=10-Q&year=2024"
    response = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        if "10-q" in a.text.lower():
            link = a["href"]
            if not link.startswith("http"):
                link = "https://investor.pnc.com" + link
            links.append(link)
        if len(links) >= 5:
            break
    return links

def get_10q_links_wells():
    base_url = "https://www.wellsfargo.com/about/investor-relations/filings/"
    resp = requests.get(base_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        if "10-q" in a.text.lower():
            link = a["href"]
            if not link.startswith("http"):
                link = "https://www.wellsfargo.com" + link
            links.append(link)
        if len(links) >= 5:
            break
    return links

LINK_FUNCTIONS = {
    "J.P. Morgan": get_10q_links_jpm,
    "Bank of America": get_10q_links_boa,
    "U.S. Bank": get_10q_links_usbank,
    "Citibank": get_10q_links_citi,
    "PNC": get_10q_links_pnc,
    "Wells Fargo": get_10q_links_wells
}

selected_bank = st.selectbox("Select a Bank", list(BANK_URLS.keys()))

if st.button("Fetch 10-Q Reports"):
    st.info(f"Fetching 10-Q reports for {selected_bank}...")
    try:
        links = LINK_FUNCTIONS[selected_bank]()
        if links:
            for i, link in enumerate(links, 1):
                st.markdown(f"{i}. [View 10-Q Report]({link})")
        else:
            st.warning("No 10-Q reports found.")
    except Exception as e:
        st.error(f"Error fetching reports: {e}")
                           
