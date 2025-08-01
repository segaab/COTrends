import streamlit as st
import requests
import pandas as pd

# SEC API CIK for JPMorgan Chase
CIK = "0000019617"
API_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
HEADERS = {
    "User-Agent": "Tsegaab G (segaab120@gmail.com)"
}

# Set up Streamlit
st.set_page_config(page_title="SEC 10-Q Test Dashboard", layout="wide")
st.title("📊 SEC 10-Q Financials: JPMorgan Chase")

# Fetch JSON data from SEC
with st.spinner("📡 Fetching financial facts from SEC API..."):
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        st.stop()

# Define tags to extract
metrics = {
    "Revenue": "Revenues",
    "Interest Expense": "InterestExpense",
    "Net Income": "NetIncomeLoss",
    "Total Loans": "LoansReceivableNet"
}

us_gaap = data.get("facts", {}).get("us-gaap", {})

st.subheader("📂 Extracted Key Financial Metrics")

# Loop over each metric and extract data
chart_data = {}

for label, tag in metrics.items():
    tag_data = us_gaap.get(tag)
    if tag_data and "USD" in tag_data.get("units", {}):
        values = tag_data["units"]["USD"]
        df = pd.DataFrame(values)
        df = df[df["form"].isin(["10-Q", "10-K"])]
        df["date"] = pd.to_datetime(df["end"], errors="coerce")
        df = df.sort_values("date", ascending=True)
        df = df[["date", "val"]].dropna()

        if not df.empty:
            df.set_index("date", inplace=True)
            chart_data[label] = df["val"]

# Combine all into one DataFrame
if chart_data:
    combined = pd.DataFrame(chart_data)
    st.line_chart(combined)
    st.dataframe(combined.tail(6))
else:
    st.warning("⚠️ No matching financial metrics found.")
