import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Leader-Follower Backtest",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Leader-Follower Backtesting Dashboard")

# -----------------------------
# Sidebar: Parameters
# -----------------------------
st.sidebar.header("Simulation Parameters")
starting_balance = st.sidebar.number_input("Starting Balance ($)", value=600, min_value=100, step=100)
start_date = st.sidebar.date_input("Start Date", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2024, 7, 1))

st.sidebar.header("Strategy Settings")
roc_window = st.sidebar.slider("ROC Window (days)", min_value=5, max_value=30, value=14)

# -----------------------------
# Stock List
# -----------------------------
tickers = ['NVDA', 'AMD', 'MRVL', 'ASML']
leader = 'NVDA'
followers = [t for t in tickers if t != leader]

# -----------------------------
# Fetch Data
# -----------------------------
@st.cache_data
def fetch_stock_data(symbols, start_date, end_date):
    try:
        ticker = Ticker(symbols)
        data = ticker.history(start=start_date, end=end_date, interval='1d')
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index().pivot(index='date', columns='symbol', values='adjclose')
        return data.fillna(method='ffill').dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

data = fetch_stock_data(tickers, start_date, end_date)
if data is None:
    st.stop()

# -----------------------------
# Metrics
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
            phases.append("Early Inflection")
        elif roc_neg_space.iloc[i] < 0 and acc_neg_space.iloc[i] < 0:
            phases.append("Late Inflection")
        elif roc_neg_space.iloc[i] < 0 and acc_neg_space.iloc[i] >= 0:
            phases.append("Mid Inflection")
        else:
            phases.append("Inactive")
    return phases

negative_space, leader_norm, avg_follower_norm = calculate_negative_space(data[leader], data[followers])
roc_neg_space = calculate_roc(negative_space, roc_window)
acc_neg_space = calculate_roc(roc_neg_space, roc_window)
phases = identify_phases(negative_space, roc_neg_space, acc_neg_space)

# -----------------------------
# Backtest
# -----------------------------
def backtest_strategy(data, phases, starting_balance):
    balance = starting_balance
    shares = {symbol: 0 for symbol in data.columns}
    portfolio_values = []
    trades = []
    in_position = False

    for i, (date, row) in enumerate(data.iterrows()):
        phase = phases[i]
        prices = row.to_dict()
        portfolio_value = balance + sum(shares[s] * prices[s] for s in data.columns)
        portfolio_values.append(portfolio_value)

        if phase == "Early Inflection" and not in_position:
            allocation = balance / len(data.columns)
            for s in data.columns:
                shares[s] = allocation / prices[s]
            trades.append({'date': date, 'action': 'BUY', 'phase': phase})
            balance = 0
            in_position = True

        elif phase == "Late Inflection" and in_position:
            balance = sum(shares[s] * prices[s] for s in data.columns)
            trades.append({'date': date, 'action': 'SELL', 'phase': phase})
            shares = {s: 0 for s in data.columns}
            in_position = False

    return portfolio_values, trades

portfolio_values, trades = backtest_strategy(data, phases, starting_balance)

# -----------------------------
# Display Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Starting Balance", f"${starting_balance:,.2f}")
col2.metric("Final Value", f"${portfolio_values[-1]:,.2f}")
col3.metric("Total Return", f"{(portfolio_values[-1]-starting_balance)/starting_balance*100:.1f}%")
col4.metric("Profit/Loss", f"${portfolio_values[-1]-starting_balance:,.2f}")

# -----------------------------
# Charts
# -----------------------------
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=['Normalized Prices', 'Negative Space & ROC', 'Portfolio Value'],
    specs=[[{}], [{}], [{}]]
)

# Normalized Prices
for t in tickers:
    fig.add_trace(go.Scatter(x=data.index, y=normalize_prices(data[t]), name=t), row=1, col=1)

# Negative Space & ROC
fig.add_trace(go.Scatter(x=data.index, y=negative_space, name='Negative Space', line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=roc_neg_space, name='ROC', line=dict(color='purple')), row=2, col=1)

# Portfolio
fig.add_trace(go.Scatter(x=data.index, y=portfolio_values, name='Portfolio Value', line=dict(color='green')), row=3, col=1)

# Add trade markers
for t in trades:
    y_val = portfolio_values[data.index.get_loc(t['date'])]
    color = 'green' if t['action']=='BUY' else 'red'
    symbol = 'triangle-up' if t['action']=='BUY' else 'triangle-down'
    fig.add_trace(go.Scatter(x=[t['date']], y=[y_val], mode='markers', marker=dict(color=color, symbol=symbol, size=12), name=f"{t['action']}"))

fig.update_layout(height=800)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Trade History
# -----------------------------
st.subheader("Trade History")
if trades:
    trades_df = pd.DataFrame(trades)
    st.dataframe(trades_df)
else:
    st.info("No trades executed.")

# -----------------------------
# Download Results
# -----------------------------
results_df = pd.DataFrame({
    "Date": data.index,
    "Leader_Normalized": leader_norm,
    "Followers_Avg": avg_follower_norm,
    "Negative_Space": negative_space,
    "ROC_Negative_Space": roc_neg_space,
    "ACC_Negative_Space": acc_neg_space,
    "Phase": phases,
    "Portfolio_Value": portfolio_values
})

st.download_button("Download Backtest CSV", results_df.to_csv(index=False), file_name="backtest_results.csv")
