import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sector Wave & Negative Space Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sector Wave & Negative Space Dashboard")

# -----------------------------
# 1. Sector & Ticker Mapping
# -----------------------------
SECTORS = {
    "Information Technology": {
        "AI": ["NVDA", "AMD", "PLTR", "SMCI", "SNOW", "AVGO"],
        "Cloud Computing": ["AMZN", "MSFT", "NET", "ORCL", "SMCI", "PLTR"],
        "Cybersecurity": ["CRWD", "ZS", "FTNT", "CYBR", "FFIV", "AVGO"]
    },
    "Healthcare": {
        "Telemedicine": ["TDOC", "DOCS", "AMWL", "PINC", "PHR", "IRTC"],
        "Biotechnology": ["MRNA", "REGN", "VRTX", "BTAI", "TNXP", "SNGX"],
        "Medical Devices": ["BSX", "MDT", "ABT", "SNN", "ZBH", "MASI"]
    },
    "Energy": {
        "Renewable Energy": ["NEE", "FSLR", "BEP", "ENLT", "RNW", "CEG"],
        "Energy Storage": ["STEM", "BE", "AEE", "CMS", "CEG", "ENPH"]
    },
    "Financials": {
        "Fintech": ["SQ", "SOFI", "PYPL", "TW", "HOOD", "IBKR"]
    },
    "Consumer Discretionary": {
        "E-commerce": ["AMZN", "SHOP", "WMT", "PYPL", "CVNA", "GRPN"],
        "Electric Vehicles": ["TSLA", "BYDDY", "NIO", "VWAGY", "RIVN", "LCID"]
    },
    "Industrials": {
        "Robotics & Automation": ["ISRG", "TXN", "ABB", "TER", "SYM", "PATH"],
        "Aerospace Technology": ["BA", "LMT", "RTX", "GD", "TDY", "HII"]
    },
    "Materials": {
        "Advanced Materials": ["EMN", "HTGC", "KMT", "RYAM", "HUN", "ASIX"],
        "Nanotechnology": ["ZTEK", "NNOMF", "NNXPF", "ATOM", "AVAV", "ONTO"],
        "Sustainable & Recycled Materials": ["EMN", "ECL", "SMG", "PKG", "WRK", "SEE"]
    },
    "Utilities": {
        "Smart Grid Technology": ["ITRI", "GRID", "IPWR", "BMI", "HASI", "ARQ"],
        "Renewable Electricity Integration": ["PEG", "NEE", "ED", "SO", "DUK", "ES"]
    },
    "Consumer Staples": {
        "Sustainable & Organic Food": ["NSRGY", "TSN", "CLX", "MO", "OLLI", "COCO"],
        "Household & Personal Care": ["PG", "CLX", "KMB", "UL", "CL", "EL"]
    },
    "Communication Services": {
        "Social Media & Interactive Media": ["META", "SNAP", "PINS", "SPOT", "BILI", "TWTR"],
        "Telecom Services": ["VZ", "T", "TMUS", "CMCSA", "DISH", "NOVN"]
    },
    "Real Estate": {
        "Proptech & Real Estate Tech": ["Z", "OPEN", "RDFN", "AURUMPROP", "COMP", "DOC"],
        "REITs (Retail & Industrial)": ["PLD", "AMT", "SPG", "PSA", "EQR", "DLR"]
    },
    "Basic Materials": {
        "Chemicals & Specialty Chemicals": ["DOW", "LYB", "SHW", "HUN", "ECL", "PPG"],
        "Metals & Mining": ["BHP", "RIO", "FCX", "NEM", "GOLD", "VALE"]
    },
    "Oil & Gas": {
        "Major Integrated Oil Companies": ["XOM", "CVX", "COP", "SLB", "WMB", "EOG"],
        "Oilfield Services & Equipment": ["SLB", "HAL", "BKR", "NOV", "FTI", "COS"],
        "Canadian Oil & Gas": ["MEG", "FO", "ATH", "HWX", "CNQ", "SU"]
    },
    "Gold & Precious Metals": {
        "Gold Mining": ["NEM", "GOLD", "KL", "AEM", "FNV", "ABX"],
        "Precious Metals Streaming & Royalty": ["FNV", "WPM", "RGLD", "HL", "PAAS", "AUY"]
    },
    "Arms/Defense": ["LMT", "RTX", "BA", "HON", "GD", "NOC"],
    "Cryptocurrency": ["COIN", "MSTR", "RIOT", "MARA", "BLOK", "CORE", "CLSK", "HUT", "HOOD", "TERA"]
}

# -----------------------------
# 2. Sidebar
# -----------------------------
sector = st.sidebar.selectbox("Select Sector", list(SECTORS.keys()))
subsector_list = list(SECTORS[sector].keys()) if isinstance(SECTORS[sector], dict) else [sector]
subsector = st.sidebar.selectbox("Select Niche/Subsector", subsector_list)
tickers = SECTORS[sector][subsector] if isinstance(SECTORS[sector], dict) else SECTORS[sector]
leader = st.sidebar.selectbox("Select Leader", tickers)
start_date = st.sidebar.date_input("Start Date", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2024, 7, 1))

roc_window = st.sidebar.slider("ROC Window (days)", min_value=5, max_value=30, value=14)
neg_space_threshold = st.sidebar.slider("Negative Space Threshold", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

# -----------------------------
# 3. Data Fetching
# -----------------------------
@st.cache_data
def fetch_stock_data(symbols, start_date, end_date):
    ticker = Ticker(symbols)
    data = ticker.history(start=start_date, end=end_date, interval='1d')
    
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index().pivot(index="date", columns="symbol", values="adjclose")
    return data.fillna(method='ffill').dropna()

data = fetch_stock_data(tickers, start_date, end_date)

# -----------------------------
# 4. Metrics & Phase Detection
# -----------------------------
def normalize_prices(prices):
    return (prices / prices.iloc[0] - 1) * 100

def calculate_negative_space(leader_prices, follower_prices):
    leader_norm = normalize_prices(leader_prices)
    follower_norms = pd.concat([normalize_prices(follower_prices[c]) for c in follower_prices.columns], axis=1)
    avg_follower_norm = follower_norms.mean(axis=1)
    negative_space = leader_norm - avg_follower_norm
    return negative_space, leader_norm, avg_follower_norm

def calculate_roc(series, window):
    return series.pct_change(window) * 100

def identify_phases(negative_space, roc_neg_space, acc_neg_space):
    phases = []
    for i in range(len(negative_space)):
        if pd.isna(roc_neg_space.iloc[i]) or pd.isna(acc_neg_space.iloc[i]):
            phases.append("Inactive")
        elif roc_neg_space.iloc[i] > 0 and negative_space.iloc[i] > 0:
            phases.append("Initiation")
        elif roc_neg_space.iloc[i] < 0 and acc_neg_space.iloc[i] < 0:
            phases.append("Early Inflection")
        elif roc_neg_space.iloc[i] < 0 and acc_neg_space.iloc[i] >= 0:
            phases.append("Mid Inflection")
        elif roc_neg_space.iloc[i] >= 0 and negative_space.iloc[i] < 0:
            phases.append("Late Inflection")
        elif roc_neg_space.iloc[i] > 0 and negative_space.iloc[i] > 0:
            phases.append("Interruption")
        else:
            phases.append("Inactive")
    return phases

leader_prices = data[leader]
follower_prices = data[[c for c in tickers if c != leader]]
negative_space, leader_norm, avg_follower_norm = calculate_negative_space(leader_prices, follower_prices)
roc_neg_space = calculate_roc(negative_space, roc_window)
acc_neg_space = calculate_roc(roc_neg_space, roc_window)
phases = identify_phases(negative_space, roc_neg_space, acc_neg_space)

# -----------------------------
# 5. Main Display
# -----------------------------
st.subheader("Normalized Prices & Metrics")
st.line_chart(pd.concat([leader_norm, avg_follower_norm], axis=1))

st.subheader("Negative Space & ROC")
fig_neg = go.Figure()
fig_neg.add_trace(go.Scatter(x=data.index, y=negative_space, name='Negative Space', line=dict(color='red')))
fig_neg.add_trace(go.Scatter(x=data.index, y=roc_neg_space, name='ROC Negative Space', line=dict(color='purple')))
st.plotly_chart(fig_neg, use_container_width=True)

st.subheader("Current Phase")
st.markdown(f"**{phases[-1]}**")

# -----------------------------
# 6. Dot Plot: Progression Toward Inflection
# -----------------------------
st.subheader("Stock Progression Toward Leader Inflection")

# Compute % progress: normalize follower vs leader max
progress = pd.DataFrame({c: normalize_prices(data[c]) / normalize_prices(data[leader]).max() for c in tickers})
latest_progress = progress.iloc[-1]

fig_dot = go.Figure()
for stock in tickers:
    fig_dot.add_trace(go.Scatter(
        x=[stock],
        y=[latest_progress[stock]],
        mode='markers+lines',
        name=stock,
        line=dict(width=2),
        marker=dict(size=12, color='blue'),
        text=[f"Phase: {phases[-1]}<br>Progress: {latest_progress[stock]:.2f}"],
        hoverinfo="text"
    ))

fig_dot.update_layout(
    yaxis_title="Progress to Leader Inflection",
    xaxis_title="Stocks",
    yaxis=dict(range=[0, 1]),
    height=500
)
st.plotly_chart(fig_dot, use_container_width=True)

# -----------------------------
# 7. Download Metrics
# -----------------------------
metrics_df = pd.DataFrame({
    "Date": data.index,
    "Leader": leader_norm,
    "Followers_Avg": avg_follower_norm,
    "Negative_Space": negative_space,
    "ROC_Negative_Space": roc_neg_space,
    "ACC_Negative_Space": acc_neg_space,
    "Phase": phases
})
st.download_button("Download Metrics CSV", metrics_df.to_csv(index=False), file_name=f"{sector}_{subsector}_metrics.csv")
