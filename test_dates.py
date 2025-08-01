import streamlit as st
import requests
import re
import pandas as pd

# PDF.co API key (replace this!)
PDFCO_API_KEY = "segaab120@gmail.com_nToGFafi2kw9Nlx5JdgtvEoDjqw06HzKczviZvlZ4V7OXv7eN1ZS6eIUIVV2JOQi"
PDFCO_HEADERS = {
    "x-api-key": PDFCO_API_KEY,
    "Content-Type": "application/json"
}

# Streamlit app setup
st.set_page_config(page_title="PDF Sector Exposure Dashboard", layout="wide")
st.title("📊 Sector Loan Exposure from 10-Q PDF (via PDF.co)")

# PDF URL input
pdf_url = st.text_input(
    "Enter direct PDF URL:",
    value="https://www.sec.gov/Archives/edgar/data/19617/000001961724000117/jpm-20240630.pdf"
)

# Button to trigger parsing and dashboard display
if st.button("Parse PDF and Show Dashboard"):
    with st.spinner("Calling PDF.co API..."):
        try:
            # Step 1: Send PDF to PDF.co
            payload = {
                "url": pdf_url,
                "inline": True,
                "pages": ""
            }
            response = requests.post(
                "https://api.pdf.co/v1/pdf/convert/to/text",
                json=payload,
                headers=PDFCO_HEADERS,
                timeout=30
            )
            response.raise_for_status()
            parsed_text = response.json().get("body", "")

            st.success("✅ PDF parsed successfully.")
            st.text_area("📝 Parsed Text Preview", parsed_text[:3000], height=300)

            # Step 2: Extract sector data via regex
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

            # Step 3: Show results in dashboard
            if sector_data:
                st.subheader("📈 Dashboard: Sector Loan Breakdown")
                df = pd.DataFrame.from_dict(sector_data, orient="index", columns=["Loan Exposure (USD)"])
                st.bar_chart(df)
                st.dataframe(df)
            else:
                st.warning("⚠️ No recognizable sector data found in the PDF.")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ PDF.co API request failed: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
