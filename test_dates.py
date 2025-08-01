import streamlit as st
import requests
import re
import pandas as pd

# CONFIG
PDFCO_API_KEY = "nToGFafi2kw9Nlx5JdgtvEoDjqw06HzKczviZvlZ4V7OXv7eN1ZS6eIUIVV2JOQi"
PDFCO_HEADERS = {
    "x-api-key": PDFCO_API_KEY,
    "Content-Type": "application/json"
}

st.set_page_config(page_title="Test PDF Parse", layout="wide")
st.title("🧪 Test PDF Parsing with PDF.co")

# Use a working direct PDF URL (SEC.gov or public CDN)
pdf_url = st.text_input("Enter direct PDF URL:", value="https://www.sec.gov/Archives/edgar/data/19617/000001961724000117/jpm-20240630.pdf")

if st.button("Parse PDF via PDF.co"):
    with st.spinner("Parsing PDF..."):
        try:
            payload = {
                "url": pdf_url,
                "inline": True,
                "pages": ""
            }
            r = requests.post("https://api.pdf.co/v1/pdf/convert/to/text", json=payload, headers=PDFCO_HEADERS, timeout=30)
            r.raise_for_status()

            result = r.json()
            if "body" in result:
                text = result["body"]
                st.success("✅ PDF parsed successfully.")
                st.text_area("Parsed Text", text[:3000])  # show first 3,000 chars
            else:
                st.error("❌ No body in response.")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ PDF.co API request failed: {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
