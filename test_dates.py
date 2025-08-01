import streamlit as st
import requests
import re
import pandas as pd
import xml.etree.ElementTree as ET

# 🔐 Replace this with your actual PDF.co API Key
PDFCO_API_KEY = "segaab120@gmail.com_nToGFafi2kw9Nlx5JdgtvEoDjqw06HzKczviZvlZ4V7OXv7eN1ZS6eIUIVV2JOQi"
PDFCO_HEADERS = {
    "x-api-key": PDFCO_API_KEY,
    "Content-Type": "application/json"
}

# Streamlit setup
st.set_page_config(page_title="Sector Loan Exposure Dashboard", layout="wide")
st.title("📊 Sector Loan Exposure from 10-Q PDF using PDF.co Document Parser")

# Input URL for 10-Q PDF
pdf_url = st.text_input(
    "Enter direct 10-Q PDF URL:",
    value="https://www.sec.gov/Archives/edgar/data/19617/000001961724000117/jpm-20240630.pdf"
)

# Input for template ID (Document Parser Template ID from your PDF.co account)
template_id = st.text_input("Enter your Document Parser Template ID:", value="REPLACE_WITH_TEMPLATE_ID")

# Function to clean XML content
def clean_extracted_text(text):
    try:
        root = ET.fromstring(f"<root>{text}</root>")
        cleaned = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                cleaned.append(elem.text.strip())
        return "\n".join(cleaned)
    except ET.ParseError:
        return text

# Button to run parser
if st.button("Parse PDF and Show Dashboard"):
    with st.spinner("🔄 Parsing PDF using PDF.co Document Parser API..."):
        try:
            # Use PDF.co Document Parser API endpoint
            endpoint = "https://api.pdf.co/v1/pdf/documentparser"
            payload = {
                "url": pdf_url,
                "templateId": template_id,
                "inline": True
            }

            response = requests.post(endpoint, headers=PDFCO_HEADERS, json=payload, timeout=30)
            response.raise_for_status()

            parsed_result = response.json()
            if not parsed_result.get("body"):
                st.warning("⚠️ No content extracted using the Document Parser.")
                st.stop()

            raw_text = parsed_result["body"]
            cleaned_text = clean_extracted_text(raw_text)

            st.text_area("📄 Cleaned Extracted Data", cleaned_text[:3000], height=300)

            # Extract loan exposure values using regex or known key patterns
            patterns = {
                "Real Estate": r"real estate[^\n$]*?\$?([\d.,]+)",
                "Commercial": r"commercial[^\n$]*?\$?([\d.,]+)",
                "Consumer": r"consumer[^\n$]*?\$?([\d.,]+)",
                "Agriculture": r"agriculture[^\n$]*?\$?([\d.,]+)"
            }

            sector_data = {}
            for sector, pattern in patterns.items():
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1).replace(",", ""))
                        sector_data[sector] = value
                    except ValueError:
                        continue

            if sector_data:
                st.subheader("📈 Sector Loan Breakdown")
                df = pd.DataFrame.from_dict(sector_data, orient="index", columns=["Loan Exposure (USD)"])
                st.bar_chart(df)
                st.dataframe(df)
            else:
                st.warning("⚠️ No sector loan data matched from the extracted content.")

        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ PDF.co request failed: {req_err}")
        except Exception as e:
            st.error(f"❌ Unexpected error occurred: {e}")
                
