import streamlit as st
import requests
from bs4 import BeautifulSoup

# === Configuration ===
st.set_page_config(page_title="SEC 10‑Q Parser", layout="wide")
st.title("📄 SEC EDGAR 10‑Q HTML Parser - Tag List Only")

CIK = "0000019617"  # JPMorgan Chase
HEADERS = {
    "User-Agent": "segaab120@gmail.com"
}

# === Step 1: Fetch latest 10-Q index JSON URL from EDGAR ===
@st.cache_data(show_spinner=False)
def fetch_latest_10q_html():
    index_url = f"https://data.sec.gov/submissions/CIK{CIK}.json"
    res = requests.get(index_url, headers=HEADERS)
    res.raise_for_status()
    data = res.json()

    filings = data.get("filings", {}).get("recent", {})
    for i, form in enumerate(filings.get("form", [])):
        if form == "10-Q":
            accession = filings["accessionNumber"][i].replace("-", "")
            cik_int = int(CIK)
            return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/index.json"
    return None

# === Step 2: Get .htm or .xml report from index ===
def fetch_html_url_from_index(index_json_url):
    res = requests.get(index_json_url, headers=HEADERS)
    res.raise_for_status()
    index = res.json()
    for item in index["directory"]["item"]:
        name = item["name"].lower()
        if name.endswith(".htm") or name.endswith(".xml"):
            return index_json_url.rsplit("/", 1)[0] + "/" + item["name"]
    return None

# === Step 3: Parse and list unique XML tag names ===
def extract_unique_tags(url):
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.content, "lxml-xml")
    tag_names = set(el.name for el in soup.find_all())
    return sorted(tag_names)

# === Streamlit UI ===
st.subheader("🔍 Step 1: Fetch Latest 10‑Q Filing from SEC")
if st.button("Fetch Latest 10‑Q HTML Link"):
    index_url = fetch_latest_10q_html()
    if index_url:
        html_url = fetch_html_url_from_index(index_url)
        if html_url:
            st.success(f"✅ Found report URL: [Open Filing]({html_url})")
            st.session_state["html_url"] = html_url
        else:
            st.warning("⚠️ No HTML/XML report found in this filing.")
    else:
        st.warning("⚠️ No recent 10‑Q filing found for JPMorgan.")

# === Extract and show unique tags ===
if "html_url" in st.session_state:
    st.subheader("📤 Step 2: Display All Tag Names")
    if st.button("List All Tags"):
        tags = extract_unique_tags(st.session_state["html_url"])
        if tags:
            st.subheader(f"🏷️ Found {len(tags)} unique tags")
            st.code("\n".join(tags))
        else:
            st.warning("No tags found in the selected document.")
