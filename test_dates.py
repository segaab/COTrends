import streamlit as st
import requests
import pandas as pd

# Configuration
CIK = "0000019617"  # JPMorgan Chase
API_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
HEADERS = {
    "User-Agent": "Tsegaab G (segaab120@gmail.com)"  # Replace with your email
}

# Initialize Streamlit App
st.set_page_config(page_title="SEC 10-Q Dashboard", layout="wide")
st.title("📊 SEC 10-Q Financials: JPMorgan Chase")

# Fetch SEC data
with st.spinner("📡 Fetching data from SEC..."):
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        st.stop()

# Select key metrics
metrics = {
    "Revenue": "Revenues",
    "Interest Expense": "InterestExpense",
    "Net Income": "NetIncomeLoss",
    "Total Loans": "LoansReceivableNet"
}

us_gaap = data.get("facts", {}).get("us-gaap", {})

st.subheader("📂 Extracted Financial Metrics")

# Build time series
chart_data = {}

for label, tag in metrics.items():
    tag_data = us_gaap.get(tag)
    if tag_data and "USD" in tag_data.get("units", {}):
        values = tag_data["units"]["USD"]
        df = pd.DataFrame(values)
        df = df[df["form"].isin(["10-Q", "10-K"])]
        df["date"] = pd.to_datetime(df["end"], errors="coerce")
        df = df[["date", "val"]].dropna()
        df = df.drop_duplicates(subset="date")
        df.set_index("date", inplace=True)

        if not df.empty:
            chart_data[label] = df["val"]

# Merge and display
if chart_data:
    combined = pd.concat(chart_data.values(), axis=1)
    combined.columns = list(chart_data.keys())
    combined = combined.sort_index()

    st.line_chart(combined)
    st.dataframe(combined.tail(6))
else:
    st.warning("⚠️ No matching metrics found for this company.")
