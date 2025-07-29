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

# ---------------------
# Helpers
# ---------------------
def fetch_fred_data(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()['observations']
        latest = next((item for item in reversed(data) if item['value'] not in ['.']), None)
        prev = next((item for item in reversed(data[:-1]) if item['value'] not in ['.']), None)
        return float(latest['value']), float(prev['value']), latest['date']
    else:
        st.error(f"Error fetching {series_id} from FRED: {resp.status_code}")
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
        "inputs": f"""Below is a snapshot of current macroeconomic data. Please generate a brief and insightful commentary suitable for an investment research note:

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
            generated = response.json()[0]["generated_text"]
            return generated
        else:
            return f"Error from HF API: {response.status_code} - {response.text}"

# ---------------------
# UI
# ---------------------
st.title("📌 Macro Snapshot Dashboard")

with st.expander("📝 How it works", expanded=False):
    st.markdown("""
    - 📊 Fetches macroeconomic indicators (FRED + YahooQuery)
    - 🧠 Optionally sends results to Hugging Face R1 model for commentary
    """)

# ---------------------
# Fetch Button
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
        snapshot_lines.append(fed_line + " | Stance: Neutral-Hawkish")

        # 📊 Key Economic Indicator
        st.markdown("### 📊 Key Economic Indicator")
        cpi_now, cpi_prev, cpi_date = fetch_fred_data("CPILFESL")
        inflation_trend = "Cooling" if cpi_now < cpi_prev else "Heating"
        cpi_line = f"Core CPI YoY: {cpi_now:.2f}% (Previous: {cpi_prev:.2f}%) – {inflation_trend} trend"
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

        # 🌍 Global Risks (Static)
        st.markdown("### 🌍 Global Macro Risks")
        risks = "• U.S.–China tensions\n• Commodity volatility\n• Election cycles\n• Oil/geopolitical risk"
        st.write(risks)
        snapshot_lines.append("Key Risks: U.S.–China, oil supply shocks, elections, commodity volatility")

        # Store snapshot for HF
        st.session_state.snapshot_text = "\n".join(snapshot_lines)

        st.success("Snapshot complete and stored for analysis.")

# ---------------------
# Hugging Face Inference Button
# ---------------------
if st.button("🧠 Generate Commentary with Hugging Face"):
    if st.session_state.snapshot_text.strip() == "":
        st.warning("Run the macro snapshot fetch first!")
    else:
        commentary = generate_hf_analysis(st.session_state.snapshot_text)
        st.markdown("### 🧠 AI Commentary")
        st.write(commentary)
        
st.set_page_config(page_title="Macro Snapshot Dashboard", layout="centered")
st.title("📌 Macro Snapshot Dashboard")

# ---------------------
# Helpers
# ---------------------
def fetch_fred_data(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()['observations']
        latest = next((item for item in reversed(data) if item['value'] not in ['.']), None)
        prev = next((item for item in reversed(data[:-1]) if item['value'] not in ['.']), None)
        return float(latest['value']), float(prev['value']), latest['date']
    else:
        st.error(f"Error fetching {series_id} from FRED: {resp.status_code}")
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

# ---------------------
# UI Layout
# ---------------------
with st.expander("📝 How it works", expanded=False):
    st.markdown("""
    - Fetches macroeconomic indicators using **FRED API**
    - Pulls S&P500, 10Y Yield, and Gold prices using **yahooquery**
    - Displays snapshot + logs any fetch issues
    """)

if st.button("🚀 Run Macro Snapshot Fetch"):
    with st.spinner("Fetching data..."):

        today = datetime.date.today()
        st.subheader(f"🗓 Snapshot Date: {today}")

        # 🏦 Monetary Policy
        st.markdown("### 🏦 Monetary Policy")
        fed_rate, _, rate_date = fetch_fred_data("FEDFUNDS")
        st.write(f"**Fed Funds Rate:** {fed_rate:.2f}% (as of {rate_date})")
        st.write("**Stance:** Neutral-Hawkish")
        st.write("**Tools:** Fed Funds Rate, Balance Sheet Runoff")

        # 📊 Key Economic Indicator
        st.markdown("### 📊 Key Economic Indicator")
        cpi_now, cpi_prev, cpi_date = fetch_fred_data("CPILFESL")
        inflation_trend = "Cooling" if cpi_now < cpi_prev else "Heating"
        st.write(f"**Core CPI YoY**: {cpi_now:.2f}% (Previous: {cpi_prev:.2f}%) – *{inflation_trend} Inflation*")

        # 📈 Market Outlook
        st.markdown("### 📈 Market Snapshot")
        sp_now, sp_prev = fetch_yahooquery_price("^GSPC")
        bond_now, bond_prev = fetch_yahooquery_price("^TNX")
        gold_now, gold_prev = fetch_yahooquery_price("GC=F")

        st.write(f"**S&P500**: {sp_now:.2f} ({sp_now - sp_prev:+.2f})")
        st.write(f"**10Y Yield**: {bond_now:.2f} ({bond_now - bond_prev:+.2f})")
        st.write(f"**Gold**: ${gold_now:.2f} ({gold_now - gold_prev:+.2f})")

        # 🌍 Global Risks (Static)
        st.markdown("### 🌍 Global Macro Risks")
        st.write("• U.S.–China tensions\n• Commodity volatility\n• Election cycles\n• Oil/geopolitical risk")

        # Logs
        st.markdown("### 🪵 Fetch Log")
        st.success("Data fetch completed successfully!")

else:
    st.info("Click the button above to fetch macroeconomic snapshot data.")
