import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(page_title="AI Sector Phase Detection Backtest", page_icon="📈", layout="wide")
st.title("📈 AI Sector Dynamic Phase Detection Backtest")
st.markdown("**Leader-Follower Strategy on AI Sector (NVDA, AMD, MRVL, ASML)**")

# ------------------------------
# Sidebar Parameters
# ------------------------------
st.sidebar.header("Simulation Parameters")
starting_balance = st.sidebar.number_input("Starting Balance ($)", value=600, min_value=100, step=100)
start_date = st.sidebar.date_input("Start Date", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2024, 7, 1))

st.sidebar.header("Strategy Settings")
roc_window = st.sidebar.slider("ROC Window (days)", min_value=5, max_value=30, value=14)
neg_space_threshold = st.sidebar.slider("Negative Space Threshold", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

# ------------------------------
# AI Sector Stocks
# ------------------------------
ai_symbols = ['NVDA', 'AMD', 'MRVL', 'ASML']

# ------------------------------
# Fetch stock data
# ------------------------------
@st.cache_data
def fetch_stock_data(symbols, start, end):
    ticker = Ticker(symbols)
    data = ticker.history(start=start, end=end, interval='1d')
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
        data = data.pivot(index='date', columns='symbol', values='adjclose')
    return data.fillna(method='ffill').dropna()

# ------------------------------
# Normalization and metrics
# ------------------------------
def normalize_prices(prices):
    return (prices / prices.iloc[0] - 1) * 100

def calculate_negative_space(leader_prices, follower_prices):
    leader_norm = normalize_prices(leader_prices)
    follower_norms = [normalize_prices(follower_prices[col]) for col in follower_prices.columns]
    avg_follower_norm = pd.concat(follower_norms, axis=1).mean(axis=1)
    negative_space = leader_norm - avg_follower_norm
    return negative_space, leader_norm, avg_follower_norm

def calculate_roc(series, window):
    return series.pct_change(window) * 100

def identify_phases(neg_space, roc_neg_space, acc_neg_space):
    phases = []
    for i in range(len(neg_space)):
        if pd.isna(roc_neg_space.iloc[i]) or pd.isna(acc_neg_space.iloc[i]):
            phases.append("Inactive")
        elif roc_neg_space.iloc[i] > 0 and neg_space.iloc[i] > 0:
            phases.append("Initiation")
        elif roc_neg_space.iloc[i] < 0 and acc_neg_space.iloc[i] < 0:
            phases.append("Early Inflection")
        elif roc_neg_space.iloc[i] < 0 and acc_neg_space.iloc[i] >= 0:
            phases.append("Mid Inflection")
        elif roc_neg_space.iloc[i] >= 0 and neg_space.iloc[i] < 0:
            phases.append("Late Inflection")
        elif roc_neg_space.iloc[i] > 0 and neg_space.iloc[i] > 0:
            phases.append("Interruption")
        else:
            phases.append("Inactive")
    return phases

# ------------------------------
# Backtesting logic
# ------------------------------
def backtest_strategy(data, phases, starting_balance):
    symbols = data.columns.tolist()
    leader = 'NVDA'
    followers = [s for s in symbols if s != leader]

    balance = starting_balance
    shares = {symbol: 0 for symbol in symbols}
    portfolio_values = []
    trades = []
    in_position = False

    for i, (date, row) in enumerate(data.iterrows()):
        phase = phases[i]
        current_prices = row.to_dict()
        portfolio_value = balance + sum(shares[symbol]*current_prices[symbol] for symbol in symbols)
        portfolio_values.append(portfolio_value)

        # Entry
        if phase == "Early Inflection" and not in_position and balance>0:
            allocation_per_stock = balance/len(symbols)
            for s in symbols:
                if current_prices[s] > 0:
                    shares[s] = allocation_per_stock / current_prices[s]
            trades.append({'date': date, 'action': 'BUY', 'phase': phase, 'balance_before': balance,
                           'allocation_per_stock': allocation_per_stock, 'prices': current_prices.copy(),
                           'shares': shares.copy()})
            balance = 0
            in_position = True
        # Exit
        elif phase in ["Late Inflection","Interruption"] and in_position:
            balance = sum(shares[s]*current_prices[s] for s in symbols)
            trades.append({'date': date, 'action': 'SELL', 'phase': phase, 'balance_after': balance,
                           'prices': current_prices.copy(), 'shares_sold': shares.copy()})
            shares = {symbol:0 for symbol in symbols}
            in_position = False

    return portfolio_values, trades

# ------------------------------
# Main app
# ------------------------------
if st.button("Run Backtest") or 'data' not in st.session_state:
    with st.spinner("Fetching stock data..."):
        data = fetch_stock_data(ai_symbols, start_date, end_date)
        if data is None:
            st.error("Failed to fetch data. Check your connection.")
            st.stop()
        st.session_state.data = data

if 'data' in st.session_state:
    data = st.session_state.data

    leader_prices = data['NVDA']
    follower_prices = data[['AMD','MRVL','ASML']]

    neg_space, leader_norm, avg_follower_norm = calculate_negative_space(leader_prices, follower_prices)
    roc_neg_space = calculate_roc(neg_space, roc_window)
    acc_neg_space = calculate_roc(roc_neg_space, roc_window)
    phases = identify_phases(neg_space, roc_neg_space, acc_neg_space)
    portfolio_values, trades = backtest_strategy(data, phases, starting_balance)

    # ------------------------------
    # Performance Metrics
    # ------------------------------
    final_value = portfolio_values[-1]
    total_return = (final_value-starting_balance)/starting_balance*100
    st.metric("Starting Balance", f"${starting_balance:,.2f}")
    st.metric("Final Value", f"${final_value:,.2f}")
    st.metric("Total Return", f"{total_return:.2f}%")
    st.metric("Profit/Loss", f"${final_value-starting_balance:,.2f}")

    # ------------------------------
    # Buy & Hold Comparison
    # ------------------------------
    initial_shares_bh = {s: (starting_balance/len(ai_symbols))/data[s].iloc[0] for s in ai_symbols}
    buy_hold_values = [sum(initial_shares_bh[s]*row[s] for s in ai_symbols) for _,row in data.iterrows()]
    bh_return = (buy_hold_values[-1]-starting_balance)/starting_balance*100
    st.metric("Buy & Hold Return", f"{bh_return:.2f}%")

    # ------------------------------
    # Visualizations
    # ------------------------------
    normalized_data = pd.DataFrame({s: (data[s]/data[s].iloc[0]-1)*100 for s in ai_symbols})

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Normalized Prices & Trade Markers","Portfolio Value vs Buy & Hold"],
        vertical_spacing=0.1
    )

    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    for i, s in enumerate(ai_symbols):
        fig.add_trace(go.Scatter(x=data.index, y=normalized_data[s], name=f"{s} (Normalized)", line=dict(color=colors[i], width=2)), row=1, col=1)

    for trade in trades:
        color = 'green' if trade['action']=='BUY' else 'red'
        symbol_marker = 'triangle-up' if trade['action']=='BUY' else 'triangle-down'
        fig.add_trace(go.Scatter(x=[trade['date']], y=[normalized_data.mean(axis=1).loc[trade['date']]],
                                 mode='markers', marker=dict(color=color, size=12, symbol=symbol_marker),
                                 name=f"{trade['action']} - {trade['phase']}"), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=portfolio_values, name='Strategy Portfolio', line=dict(color='green', width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=buy_hold_values, name='Buy & Hold', line=dict(color='orange', width=3, dash='dot')), row=2, col=1)

    fig.update_layout(height=800, title_text="AI Sector Phase Detection Strategy", showlegend=True)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="% Change from Start", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Trade History
    # ------------------------------
    st.subheader("Trade History")
    if trades:
        trades_display = []
        for t in trades:
            info = {'Date': t['date'].strftime('%Y-%m-%d'), 'Action': t['action'], 'Phase': t['phase']}
            if t['action']=='BUY':
                info['Balance Before'] = f"${t['balance_before']:,.2f}"
                info['Allocation/Stock'] = f"${t['allocation_per_stock']:,.2f}"
                for s in ai_symbols:
                    info[f'{s} Price'] = f"${t['prices'][s]:.2f}"
                    info[f'{s} Shares'] = f"{t['shares'][s]:.4f}"
            else:
                info['Balance After'] = f"${t['balance_after']:,.2f}"
                for s in ai_symbols:
                    info[f'{s} Price'] = f"${t['prices'][s]:.2f}"
                    info[f'{s} Shares Sold'] = f"{t['shares_sold'][s]:.4f}"
            trades_display.append(info)
        st.dataframe(pd.DataFrame(trades_display), use_container_width=True)
