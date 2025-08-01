import streamlit as st
import requests
import re
import pandas as pd
import xml.etree.ElementTree as ET

# 🔐 Replace this with your actual PDF.co API Key
PDFCO_API_KEY = "segaab120@gmail.com_nToGFafi2kw9Nlx5JdgtvEoDjqw06HzKczviZvlZ4V7OXv7eN1ZS6eIUIVV2JOQia"
PDFCO_HEADERS = {
    "x-api-key": PDFCO_API_KEY,
    "Content-Type": "application/json"
}

# Streamlit UI
st.set_page_config(page_title="PDF to Clean JSON", layout="wide")
st.title("📄 Parse, Clean, and Convert PDF to JSON (via PDF.co)")

# User inputs
pdf_url = st.text_input(
    "Enter direct PDF URL (e.g., 10-Q from SEC):",
    value="https://www.sec.gov/Archives/edgar/data/19617/000001961724000117/jpm-20240630.pdf"
)
template_id = st.text_input(
    "Enter your Document Parser Template ID:",
    value="REPLACE_WITH_TEMPLATE_ID"
)

# Clean HTML/XML content
def clean_xbrl_value(value: str) -> str:
    """Remove XML/HTML tags and entities from parsed PDF text."""
    if not value:
        return ""
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", value).strip()

# Button logic
if st.button("Parse PDF and Generate JSON"):
    with st.spinner("🔄 Parsing PDF via PDF.co..."):
        try:
            # Call PDF.co Document Parser
            endpoint = "https://api.pdf.co/v1/pdf/documentparser"
            payload = {
                "url": pdf_url,
                "templateId": template_id,
                "inline": True
            }

            response = requests.post(endpoint, headers=PDFCO_HEADERS, json=payload, timeout=30)
            response.raise_for_status()
            parsed = response.json()

            # Check and clean
            if not parsed.get("body"):
                st.warning("⚠️ No content extracted.")
                st.stop()

            cleaned_text = clean_xbrl_value(parsed["body"])
            st.text_area("📄 Cleaned PDF Text", cleaned_text[:3000], height=300)

            # JSON output
            output = {
                "PDF Source URL": pdf_url,
                "CleanedText": cleaned_text
            }
            json_str = json.dumps(output, indent=4)
            st.download_button("📥 Download Cleaned JSON", json_str, file_name="cleaned_pdf_data.json")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ PDF.co API error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
