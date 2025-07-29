import streamlit as st
import requests
import datetime
from yahooquery import Ticker

# ---------------------
# Configuration
# ---------------------
FRED_API_KEY = 'bb26c399cb6892eece681374de6d370e'
HUGGINGFACE_TOKEN = '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'
HF_MODEL_ID = "deepseek-ai/DeepSeek-R1"  # R1 model
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# ---------------------
# Session State
# ---------------------
if "snapshot_text" not in st.session_state:
    st.session_state.snapshot_text = ""
if "snapshot_lines" not in st.session_state:
    st.session_state.snapshot_lines = []

# ---------------------
# Helper Functions
# ---------------------
def fetch_fred_data(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json().get('observations', [])
        latest = next((item for item in reversed(data) if item['value'] not in ['.']), None)
        prev = next((item for item in reversed(data[:-1]) if item['value'] not in ['.']), None)
        return float(latest['value']), float(prev['value']), latest['date']
    else:
        st.error(f"Error fetching {series_id} from FRED.")
        return None, None, None

def fetch_yahooquery_price(ticker):
    try:
        tq = Ticker(ticker)
        hist = tq.history(period='2d', interval='1d')
        if isinstance(hist, dict) or hist.empty or len(hist) < 2:
            raise ValueError("Insufficient or invalid data")
        latest_close = hist['close'].iloc[-1]
        previous_close = hist['close'].iloc[-2]
        return latest_close, previous_close
    except Exception as e:
        st.error(f"Error fetching {ticker} via yahooquery: {e}")
        return None, None

def generate_hf_analysis(snapshot_text):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"""Below is a snapshot of macroeconomic data. Generate a brief investment-oriented analysis:

{snapshot_text}
""",
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 300
        }
    }
    with st.spinner("Calling Hugging Face model..."):
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"❌ HF API error: {response.status_code} — {response.text}"

# ---------------------
# UI Layout
# ---------------------
st.set_page_config(page_title="Macro Snapshot Dashboard", layout="centered")
st.title("📌 Macro Snapshot Dashboard")

with st.expander("ℹ️ How it Works", expanded=False):
    st.markdown("""
- Pulls **macroeconomic indicators** (Fed Funds Rate, CPI) using **FRED API**
- Pulls **market prices** (S&P 500, 10Y yield, Gold) using **YahooQuery**
- Uses Hugging Face's **R1 model** to generate investment-style analysis
""")

# ---------------------
# FETCH Button
# ---------------------
if st.button("🚀 Run Macro Snapshot Fetch"):
    with st.spinner("Fetching data..."):

        today = datetime.date.today()
        st.subheader(f"🗓 Snapshot Date: {today}")

        snapshot_lines = []

        # 🏦 Monetary Policy
        st.markdown("### 🏦 Monetary Policy")
        fed_rate, _, rate_date = fetch_fred_data("FEDFUNDS")
        fed_line = f"Fed Funds Rate: {fed_rate:.2f}% (as of {rate_date})"
        st.write(f"**{fed_line}**")
        st.write("**Stance:** Neutral-Hawkish")
        snapshot_lines.append(f"{fed_line} | Stance: Neutral-Hawkish")

        # 📊 Economic Indicator
        st.markdown("### 📊 Key Economic Indicator")
        cpi_now, cpi_prev, cpi_date = fetch_fred_data("CPILFESL")
        trend = "Cooling" if cpi_now < cpi_prev else "Heating"
        cpi_line = f"Core CPI YoY: {cpi_now:.2f}% (Prev: {cpi_prev:.2f}%) — {trend} trend"
        st.write(f"**{cpi_line}**")
        snapshot_lines.append(cpi_line)

        # 📈 Market Snapshot
        st.markdown("### 📈 Market Snapshot")
        sp_now, sp_prev = fetch_yahooquery_price("^GSPC")
        bond_now, bond_prev = fetch_yahooquery_price("^TNX")
        gold_now, gold_prev = fetch_yahooquery_price("GC=F")

        sp_line = f"S&P500: {sp_now:.2f} ({sp_now - sp_prev:+.2f})"
        bond_line = f"10Y Yield: {bond_now:.2f} ({bond_now - bond_prev:+.2f})"
        gold_line = f"Gold: ${gold_now:.2f} ({gold_now - gold_prev:+.2f})"

        st.write(f"**{sp_line}**")
        st.write(f"**{bond_line}**")
        st.write(f"**{gold_line}**")

        snapshot_lines += [sp_line, bond_line, gold_line]

        # 🌍 Global Risks
        st.markdown("### 🌍 Global Macro Risks")
        risks = "• U.S.–China tension\n• Oil/geopolitical risks\n• Elections\n• Commodity volatility"
        st.write(risks)
        snapshot_lines.append("Risks: U.S.–China, oil shocks, elections, volatility")

        # Store snapshot
        snapshot_text = "\n".join(snapshot_lines)
        st.session_state.snapshot_text = snapshot_text
        st.session_state.snapshot_lines = snapshot_lines

        st.success("Snapshot data fetched and ready for analysis!")

# ---------------------
# Hugging Face Button
# ---------------------
st.markdown("---")
st.markdown("### 🧠 Generate Commentary")

snapshot_ready = st.session_state.snapshot_text.strip() != ""
hf_btn_disabled = not snapshot_ready
hf_clicked = st.button("🧠 Generate with Hugging Face", disabled=hf_btn_disabled)

if not snapshot_ready:
    st.info("Please run the macro snapshot first.")

if hf_clicked and snapshot_ready:
    commentary = generate_hf_analysis(st.session_state.snapshot_text)
    st.markdown("### 🧠 AI Commentary")
    st.write(commentary)
