import requests
import re
import pandas as pd
import streamlit as st

# PDF.co API configuration
PDFCO_API_KEY = "nToGFafi2kw9Nlx5JdgtvEoDjqw06HzKczviZvlZ4V7OXv7eN1ZS6eIUIVV2JOQi"
PDFCO_HEADERS = {
    "x-api-key": PDFCO_API_KEY,
    "Content-Type": "application/json"
}

# Fetch latest 10-Q PDF link
with st.spinner("Fetching the latest 10-Q PDF from JPMorgan..."):
    try:
        res = requests.get("https://jpmorganchaseco.gcs-web.com/ir/sec-other-filings/overview", headers={"User-Agent": "Mozilla/5.0"})
        pdf_links = re.findall(r'https://[^\"]+\.pdf', res.text)
        if not pdf_links:
            st.error("❌ No PDF links found.")
            st.stop()
        latest_pdf_url = pdf_links[0]
        st.success(f"✅ PDF Found: [View PDF]({latest_pdf_url})")
    except Exception as e:
        st.error(f"Error fetching PDF links: {e}")
        st.stop()

# Parse PDF via PDF.co API
with st.spinner("Parsing PDF via PDF.co API..."):
    try:
        parse_url = "https://api.pdf.co/v1/pdf/convert/to/text"
        payload = {"url": latest_pdf_url, "inline": True, "pages": ""}
        parse_res = requests.post(parse_url, json=payload, headers=PDFCO_HEADERS)
        parse_res.raise_for_status()
        parsed_text = parse_res.json().get("body", "")
        st.text_area("Parsed PDF Text", parsed_text, height=300)  # Debug: View extracted text
    except Exception as e:
        st.error(f"Error parsing PDF via API: {e}")
        st.stop()

# Extract sector loan data from parsed text
with st.spinner("Extracting sector loan data..."):
    patterns = {
        "Real Estate": r"real estate[^\\n$]*?\\$?([\\d.,]+)",
        "Commercial": r"commercial[^\\n$]*?\\$?([\\d.,]+)",
        "Consumer": r"consumer[^\\n$]*?\\$?([\\d.,]+)",
        "Agriculture": r"agriculture[^\\n$]*?\\$?([\\d.,]+)"
    }
    data = {}
    for sector, pattern in patterns.items():
        match = re.search(pattern, parsed_text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(",", ""))
                data[sector] = value
            except ValueError:
                pass

# Display extracted data
if data:
    st.subheader("📊 Sector Loan Exposure Breakdown")
    df = pd.DataFrame.from_dict(data, orient="index", columns=["Loan Exposure (USD)"])
    st.bar_chart(df)
    st.dataframe(df)
else:
    st.warning("No sector data extracted from the PDF.")
