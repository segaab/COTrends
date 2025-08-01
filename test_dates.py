import streamlit as st
import requests
import re
import pandas as pd

st.set_page_config(page_title="📄 JPM Sector Loan Analysis", layout="wide")
st.title("📊 JPMorgan 10-Q: Sector Exposure via PDF.co API")

PDFCO_API_KEY = "nToGFafi2kw9Nlx5JdgtvEoDjqw06HzKczviZvlZ4V7OXv7eN1ZS6eIUIVV2JOQi"
PDFCO_HEADERS = {
    "x-api-key": PDFCO_API_KEY,
    "Content-Type": "application/json"
}

# Step 1: Fetch latest PDF URL from JPMorgan filings
with st.spinner("Fetching latest 10-Q PDF from JPMorgan..."):
    try:
        res = requests.get("https://jpmorganchaseco.gcs-web.com/ir/sec-other-filings/overview", headers={"User-Agent": "Mozilla/5.0"})
        pdf_links = re.findall(r'https://[^"]+\.pdf', res.text)
        if not pdf_links:
            st.error("❌ No PDF links found.")
            st.stop()
        latest_pdf_url = pdf_links[0]
        st.success("✅ PDF Found")
        st.markdown(f"[View PDF]({latest_pdf_url})")
    except Exception as e:
        st.error(f"Error fetching PDF links: {e}")
        st.stop()

# Step 2: Send PDF to PDF.co and extract raw text
with st.spinner("Parsing PDF via PDF.co API..."):
    try:
        parse_url = "https://api.pdf.co/v1/pdf/convert/to/text"
        payload = {"url": latest_pdf_url, "inline": True, "pages": ""}
        parse_res = requests.post(parse_url, json=payload, headers=PDFCO_HEADERS)
        parse_res.raise_for_status()
        parsed_text = parse_res.json().get("body", "")
    except Exception as e:
        st.error(f"Error parsing PDF via API: {e}")
        st.stop()

# Step 3: Extract sector-level data using regex
with st.spinner("Extracting sector loan exposure from parsed text..."):
    patterns = {
        "Real Estate": r"real estate[^\n$]*?\$?([\d.,]+)",
        "Commercial": r"commercial[^\n$]*?\$?([\d.,]+)",
        "Consumer": r"consumer[^\n$]*?\$?([\d.,]+)",
        "Agriculture": r"agriculture[^\n$]*?\$?([\d.,]+)"
    }
    data = {}
    for sector, pattern in patterns.items():
        match = re.search(pattern, parsed_text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1).replace(",", ""))
                data[sector] = val
            except:
                pass

# Step 4: Visualize the result
if data:
    st.subheader("📊 Sector Loan Exposure Breakdown")
    df = pd.DataFrame.from_dict(data, orient="index", columns=["Loan Exposure (USD)"])
    st.bar_chart(df)
    st.dataframe(df)
else:
    st.warning("No sector-specific values extracted from the PDF text.")
