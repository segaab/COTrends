import streamlit as st
import requests
import re
import pandas as pd

# 🔐 Replace this with your actual PDF.co API Key
PDFCO_API_KEY = "segaab120@gmail.com_nToGFafi2kw9Nlx5JdgtvEoDjqw06HzKczviZvlZ4V7OXv7eN1ZS6eIUIVV2JOQi"

PDFCO_HEADERS = {
    "x-api-key": PDFCO_API_KEY,
    "Content-Type": "application/json"
}

# Set up the Streamlit app
st.set_page_config(page_title="Sector Loan Exposure Dashboard", layout="wide")
st.title("📊 10-Q Sector Loan Exposure Dashboard (via PDF.co)")

# Input: PDF URL
pdf_url = st.text_input(
    label="Enter direct 10-Q PDF URL (from SEC or other public source):",
    value="https://www.sec.gov/Archives/edgar/data/19617/000001961724000117/jpm-20240630.pdf"
)

# Trigger PDF parsing
if st.button("Parse PDF and Show Dashboard"):
    with st.spinner("🔄 Parsing PDF via PDF.co..."):
        try:
            # Send request to PDF.co
            payload = {
                "url": pdf_url,
                "inline": True,
                "pages": ""
            }

            response = requests.post(
                "https://api.pdf.co/v1/pdf/convert/to/text",
                headers=PDFCO_HEADERS,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            parsed_text = result.get("body", "")

            if not parsed_text:
                st.warning("⚠️ No content was extracted from the PDF.")
                st.stop()

            st.text_area("📄 Parsed Text Preview", parsed_text[:3000], height=300)

            # Extract sector-specific loan figures using regex
            patterns = {
                "Real Estate": r"real estate[^\\n$]*?\\$?([\d.,]+)",
                "Commercial": r"commercial[^\\n$]*?\\$?([\d.,]+)",
                "Consumer": r"consumer[^\\n$]*?\\$?([\d.,]+)",
                "Agriculture": r"agriculture[^\\n$]*?\\$?([\d.,]+)"
            }

            sector_data = {}
            for sector, pattern in patterns.items():
                match = re.search(pattern, parsed_text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1).replace(",", ""))
                        sector_data[sector] = value
                    except ValueError:
                        continue

            if sector_data:
                # Display dashboard
                st.subheader("📈 Loan Exposure by Sector")
                df = pd.DataFrame.from_dict(sector_data, orient="index", columns=["Loan Exposure (USD)"])
                st.bar_chart(df)
                st.dataframe(df)
            else:
                st.warning("⚠️ No sector loan data was extracted from the text.")

        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ Request to PDF.co API failed: {req_err}")
        except Exception as e:
            st.error(f"❌ Unexpected error occurred: {e}")
