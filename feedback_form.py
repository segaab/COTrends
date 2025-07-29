import streamlit as st
import requests
import datetime
from yahooquery import Ticker
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# -------------------------
# Validate token setup
# -------------------------
if not HUGGINGFACE_TOKEN:
    st.error("❌ Hugging Face token missing in `.env` (HF_TOKEN)")
if not FRED_API_KEY:
    st.warning("⚠️ FRED API key missing in `.env` (FRED_API_KEY)")

# -------------------------
# Initialize session state
# -------------------------
if "snapshot_text" not in st.session_state:
    st.session_state.snapshot_text = ""
if "snapshot_lines" not in st.session_state:
    st.session_state.snapshot_lines = []

# -------------------------
# Helper functions
# -------------------------
def fetch_fred_data(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json().get("observations", [])
        latest = next((d for d in reversed(data) if d["value"] != "."), None)
        prev = next((d for d in reversed(data[:-1]) if d["value"] != "."), None)
        return float(latest["value"]), float(prev["value"]), latest["date"]
    return None, None, None

def fetch_yahooquery_price(ticker):
    try:
        tq = Ticker(ticker)
        hist = tq.history(period="2d", interval="1d")
        if isinstance(hist, dict) or hist.empty or len(hist) < 2:
            raise ValueError("No valid data found")
        latest = hist["close"].iloc[-1]
        previous = hist["close"].iloc[-2]
        return latest, previous
    except Exception as e:
        st.error(f"YahooQuery error for {ticker}: {e}")
        return None, None

def generate_hf_analysis(snapshot_text):
    try:
        client = InferenceClient(
            provider="novita",
            api_key=HUGGINGFACE_TOKEN,
        )

        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=[
                {
                    "role": "user",
                    "content": f"""Below is a snapshot of macroeconomic and market data.

{snapshot_text}

Generate a brief macroeconomic analysis. Focus on monetary policy, inflation trend, and market implications for investors."""
                }
            ],
        )

        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error during Hugging Face inference: {str(e)}"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config("📊 Macro Snapshot", layout="centered")
st.title("📌 Macro Snapshot Dashboard")

with st.expander("ℹ️ How it Works", expanded=False):
    st.markdown("""
This dashboard:
- Fetches macro data from **FRED** (e.g., Fed Rate, CPI)
- Fetches market data from **YahooQuery** (S&P500, 10Y, Gold)
- Generates a short AI commentary using **MiniMax AI** on Hugging Face
""")

# -------------------------
# Snapshot Fetch Button
# -------------------------
if st.button("🚀 Run Macro Snapshot Fetch"):
    with st.spinner("Fetching market and macro data..."):
        today = datetime.date.today()
        st.subheader(f"🗓 Snapshot Date: {today}")
        snapshot_lines = []

        # 🏦 Monetary Policy
        st.markdown("### 🏦 Monetary Policy")
        fed_rate, _, rate_date = fetch_fred_data("FEDFUNDS")
        if fed_rate is not None:
            fed_line = f"Fed Funds Rate: {fed_rate:.2f}% (as of {rate_date}) | Stance: Neutral-Hawkish"
            st.write(f"**{fed_line}**")
            snapshot_lines.append(fed_line)
        else:
            st.warning("❗ Fed Funds Rate unavailable")

        # 📊 Economic Indicator
        st.markdown("### 📊 Inflation Indicator")
        cpi_now, cpi_prev, cpi_date = fetch_fred_data("CPILFESL")
        if cpi_now is not None and cpi_prev is not None:
            trend = "Cooling" if cpi_now < cpi_prev else "Heating"
            cpi_line = f"Core CPI YoY: {cpi_now:.2f}% (Prev: {cpi_prev:.2f}%) — {trend} trend"
            st.write(f"**{cpi_line}**")
            snapshot_lines.append(cpi_line)
        else:
            st.warning("❗ CPI data unavailable")

        # 📈 Market Snapshot
        st.markdown("### 📈 Markets Overview")
        sp_now, sp_prev = fetch_yahooquery_price("^GSPC")
        bond_now, bond_prev = fetch_yahooquery_price("^TNX")
        gold_now, gold_prev = fetch_yahooquery_price("GC=F")

        if sp_now and sp_prev:
            sp_line = f"S&P500: {sp_now:.2f} ({sp_now - sp_prev:+.2f})"
            st.write(f"**{sp_line}**")
            snapshot_lines.append(sp_line)
        else:
            snapshot_lines.append("S&P500: Unavailable")

        if bond_now and bond_prev:
            bond_line = f"10Y Yield: {bond_now:.2f} ({bond_now - bond_prev:+.2f})"
            st.write(f"**{bond_line}**")
            snapshot_lines.append(bond_line)
        else:
            snapshot_lines.append("10Y Yield: Unavailable")

        if gold_now and gold_prev:
            gold_line = f"Gold: ${gold_now:.2f} ({gold_now - gold_prev:+.2f})"
            st.write(f"**{gold_line}**")
            snapshot_lines.append(gold_line)
        else:
            snapshot_lines.append("Gold: Unavailable")

        # 🌍 Global Risks
        st.markdown("### 🌍 Global Macro Risks")
        risk_line = "Risks: US-China tension, oil shocks, elections, commodity volatility"
        st.write(risk_line)
        snapshot_lines.append(risk_line)

        st.session_state.snapshot_lines = snapshot_lines
        st.session_state.snapshot_text = "\n".join(snapshot_lines)
        st.success("✅ Snapshot Ready!")

# -------------------------
# Hugging Face Button
# -------------------------
st.markdown("---")
st.markdown("### 🧠 Generate Commentary")

if not st.session_state.snapshot_text.strip():
    st.info("📋 Please run snapshot first.")
else:
    if st.button("🧠 Generate with Hugging Face"):
        response = generate_hf_analysis(st.session_state.snapshot_text)
        st.markdown("### 🧠 AI Commentary")
        st.write(response)
