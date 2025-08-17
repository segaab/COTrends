import streamlit as st
import pandas as pd
import numpy as np
from yahooquery import Ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Leader-Follower Trading Strategy Backtest",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Leader-Follower Trading Strategy Backtest")
st.markdown("**AI Sector Wave Trading Analysis (NVDA, AMD, MRVL, ASML)**")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")
starting_balance = st.sidebar.number_input("Starting Balance ($)", value=600, min_value=100, step=100)
start_date = st.sidebar.date_input("Start Date", value=datetime(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2024, 7, 1))

# Strategy parameters
st.sidebar.header("Strategy Settings")
roc_window = st.sidebar.slider("ROC Window (days)", min_value=5, max_value=30, value=14)
neg_space_threshold = st.sidebar.slider("Negative Space Threshold", min_value=0.01, max_value=0.1, value=0.05, step=0.01)

@st.cache_data
def fetch_stock_data(symbols, start_date, end_date):
    """Fetch stock data using yahooquery"""
    try:
        ticker = Ticker(symbols)
        data = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if isinstance(data.index, pd.MultiIndex):
            # Reset index to get symbol as column
            data = data.reset_index()
            data = data.pivot(index='date', columns='symbol', values='adjclose')
        
        return data.fillna(method='ffill').dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def normalize_prices(prices):
    """Normalize prices to percentage returns from the first day"""
    return (prices / prices.iloc[0] - 1) * 100

def calculate_negative_space(leader_prices, follower_prices):
    """Calculate negative space between leader and followers using normalized prices"""
    # Normalize all prices to percentage returns from start
    leader_norm = normalize_prices(leader_prices)
    
    # Normalize each follower and calculate average
    follower_norms = []
    for col in follower_prices.columns:
        follower_norm = normalize_prices(follower_prices[col])
        follower_norms.append(follower_norm)
    
    # Average of normalized follower performances
    avg_follower_norm = pd.concat(follower_norms, axis=1).mean(axis=1)
    
    # Negative space = leader normalized performance - average follower normalized performance
    negative_space = leader_norm - avg_follower_norm
    
    return negative_space, leader_norm, avg_follower_norm

def calculate_roc(series, window):
    """Calculate Rate of Change"""
    return series.pct_change(window) * 100

def identify_phases(negative_space, roc_neg_space, acc_neg_space):
    """Identify trading phases based on negative space dynamics"""
    phases = []
    
    for i in range(len(negative_space)):
        if pd.isna(roc_neg_space.iloc[i]) or pd.isna(acc_neg_space.iloc[i]):
            phases.append("Inactive")
        elif roc_neg_space.iloc[i] > 0 and negative_space.iloc[i] > 0:
            # Negative space is positive and increasing (leader pulling away)
            phases.append("Initiation")
        elif roc_neg_space.iloc[i] < 0 and acc_neg_space.iloc[i] < 0:
            # Negative space is shrinking and acceleration is negative (early convergence)
            phases.append("Early Inflection")
        elif roc_neg_space.iloc[i] < 0 and acc_neg_space.iloc[i] >= 0:
            # Negative space still shrinking but deceleration starting
            phases.append("Mid Inflection")
        elif roc_neg_space.iloc[i] >= 0 and negative_space.iloc[i] < 0:
            # Negative space stopped shrinking, followers caught up
            phases.append("Late Inflection")
        elif roc_neg_space.iloc[i] > 0 and negative_space.iloc[i] > 0:
            # Divergence resuming
            phases.append("Interruption")
        else:
            phases.append("Inactive")
    
    return phases

def backtest_strategy(data, phases, starting_balance):
    """Backtest the trading strategy using actual prices but normalized analysis"""
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
        
        # Calculate current portfolio value using actual prices
        portfolio_value = balance + sum(shares[symbol] * current_prices[symbol] for symbol in symbols)
        portfolio_values.append(portfolio_value)
        
        # Trading logic
        if phase == "Early Inflection" and not in_position and balance > 0:
            # Enter position - equal weight allocation using actual prices
            allocation_per_stock = balance / len(symbols)
            for symbol in symbols:
                if current_prices[symbol] > 0:
                    shares[symbol] = allocation_per_stock / current_prices[symbol]
            
            trades.append({
                'date': date,
                'action': 'BUY',
                'phase': phase,
                'balance_before': balance,
                'allocation_per_stock': allocation_per_stock,
                'prices': current_prices.copy(),
                'shares': shares.copy()
            })
            
            balance = 0
            in_position = True
            
        elif phase in ["Late Inflection", "Interruption"] and in_position:
            # Exit position - sell all shares using actual prices
            balance = sum(shares[symbol] * current_prices[symbol] for symbol in symbols)
            
            trades.append({
                'date': date,
                'action': 'SELL',
                'phase': phase,
                'balance_after': balance,
                'prices': current_prices.copy(),
                'shares_sold': shares.copy()
            })
            
            shares = {symbol: 0 for symbol in symbols}
            in_position = False
    
    return portfolio_values, trades

# Main app
if st.button("Run Backtest") or 'data' not in st.session_state:
    with st.spinner("Fetching stock data..."):
        symbols = ['NVDA', 'AMD', 'MRVL', 'ASML']
        data = fetch_stock_data(symbols, start_date, end_date)
        
        if data is not None:
            st.session_state.data = data
            st.session_state.symbols = symbols
        else:
            st.error("Failed to fetch stock data. Please check your internet connection and try again.")
            st.stop()

if 'data' in st.session_state:
    data = st.session_state.data
    symbols = st.session_state.symbols
    
    # Calculate normalized metrics
    leader_prices = data['NVDA']
    follower_prices = data[['AMD', 'MRVL', 'ASML']]
    
    # Get normalized data and negative space
    negative_space, leader_norm, avg_follower_norm = calculate_negative_space(leader_prices, follower_prices)
    roc_neg_space = calculate_roc(negative_space, roc_window)
    acc_neg_space = calculate_roc(roc_neg_space, roc_window)
    
    phases = identify_phases(negative_space, roc_neg_space, acc_neg_space)
    
    # Run backtest
    portfolio_values, trades = backtest_strategy(data, phases, starting_balance)
    
    # Create normalized price data for all stocks for visualization
    normalized_data = pd.DataFrame()
    for symbol in symbols:
        normalized_data[symbol] = normalize_prices(data[symbol])
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Date': data.index,
        'Phase': phases,
        'NVDA_Normalized': leader_norm.values,
        'Followers_Avg_Normalized': avg_follower_norm.values,
        'Negative_Space': negative_space.values,
        'ROC_Negative_Space': roc_neg_space.values,
        'ACC_Negative_Space': acc_neg_space.values,
        'Portfolio_Value': portfolio_values
    })
    
    # Performance metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - starting_balance) / starting_balance * 100
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Starting Balance", f"${starting_balance:,.2f}")
    with col2:
        st.metric("Final Value", f"${final_value:,.2f}")
    with col3:
        st.metric("Total Return", f"{total_return:.1f}%")
    with col4:
        st.metric("Profit/Loss", f"${final_value - starting_balance:,.2f}")
    
    # Create visualizations
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Normalized Stock Prices (% from Start)', 'Leader vs Followers (Normalized)', 'Negative Space & ROC', 'Portfolio Value'],
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": False}]]
    )
    
    # Normalized stock prices for comparison
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, symbol in enumerate(symbols):
        fig.add_trace(
            go.Scatter(x=data.index, y=normalized_data[symbol], name=f'{symbol} (Normalized)', 
                      line=dict(width=2, color=colors[i])),
            row=1, col=1
        )
    
    # Leader vs average followers (normalized)
    fig.add_trace(
        go.Scatter(x=data.index, y=leader_norm, name='NVDA (Normalized)', 
                  line=dict(color='blue', width=3)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=avg_follower_norm, name='Followers Avg (Normalized)', 
                  line=dict(color='orange', width=3)),
        row=2, col=1
    )
    
    # Negative space and ROC
    fig.add_trace(
        go.Scatter(x=data.index, y=negative_space, name='Negative Space', 
                  line=dict(color='red', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=roc_neg_space, name='ROC Negative Space', 
                  line=dict(color='purple', width=2)),
        row=3, col=1, secondary_y=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(x=data.index, y=portfolio_values, name='Portfolio Value', 
                  line=dict(color='green', width=3)),
        row=4, col=1
    )
    fig.add_hline(y=starting_balance, line_dash="dash", line_color="gray", row=4, col=1)
    
    # Add trade markers
    for trade in trades:
        color = 'green' if trade['action'] == 'BUY' else 'red'
        symbol_marker = 'triangle-up' if trade['action'] == 'BUY' else 'triangle-down'
        
        # Get the corresponding y-values for each chart at the trade date
        trade_idx = data.index.get_loc(trade['date'])
        
        y_values = [
            normalized_data.iloc[trade_idx].mean(),  # Average normalized price
            (leader_norm.iloc[trade_idx] + avg_follower_norm.iloc[trade_idx]) / 2,  # Average of leader and followers
            negative_space.iloc[trade_idx],  # Negative space value
            portfolio_values[trade_idx]  # Portfolio value
        ]
        
        for row_idx, y_val in enumerate(y_values, 1):
            fig.add_trace(
                go.Scatter(x=[trade['date']], y=[y_val], mode='markers',
                          marker=dict(color=color, size=15, symbol=symbol_marker),
                          name=f"{trade['action']} - {trade['phase']}", 
                          showlegend=(row_idx == 1)),
                row=row_idx, col=1
            )
    
    fig.update_layout(height=900, title_text="Leader-Follower Strategy Analysis (Normalized)")
    
    # Update axis labels
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="% Change from Start", row=1, col=1)
    fig.update_yaxes(title_text="% Change from Start", row=2, col=1)
    fig.update_yaxes(title_text="Negative Space (%)", row=3, col=1)
    fig.update_yaxes(title_text="ROC (%)", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=4, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Phase distribution
    st.subheader("Phase Distribution")
    phase_counts = pd.Series(phases).value_counts()
    phase_colors = {
        'Inactive': '#gray',
        'Initiation': '#yellow', 
        'Early Inflection': '#green',
        'Mid Inflection': '#blue',
        'Late Inflection': '#orange',
        'Interruption': '#red'
    }
    colors_list = [phase_colors.get(phase, '#gray') for phase in phase_counts.index]
    
    fig_phases = go.Figure(data=[go.Bar(
        x=phase_counts.index, 
        y=phase_counts.values,
        marker_color=colors_list
    )])
    fig_phases.update_layout(title="Trading Phase Distribution", xaxis_title="Phase", yaxis_title="Days")
    st.plotly_chart(fig_phases, use_container_width=True)
    
    # Trading history
    st.subheader("Trading History")
    if trades:
        # Create a more readable trades dataframe
        trades_display = []
        for trade in trades:
            trade_info = {
                'Date': trade['date'].strftime('%Y-%m-%d'),
                'Action': trade['action'],
                'Phase': trade['phase'],
            }
            
            if trade['action'] == 'BUY':
                trade_info['Balance Before'] = f"${trade['balance_before']:,.2f}"
                trade_info['Allocation per Stock'] = f"${trade['allocation_per_stock']:,.2f}"
                for symbol in symbols:
                    trade_info[f'{symbol} Price'] = f"${trade['prices'][symbol]:.2f}"
                    trade_info[f'{symbol} Shares'] = f"{trade['shares'][symbol]:.4f}"
            else:
                trade_info['Balance After'] = f"${trade['balance_after']:,.2f}"
                for symbol in symbols:
                    trade_info[f'{symbol} Price'] = f"${trade['prices'][symbol]:.2f}"
                    trade_info[f'{symbol} Shares Sold'] = f"{trade['shares_sold'][symbol]:.4f}"
            
            trades_display.append(trade_info)
        
        trades_df = pd.DataFrame(trades_display)
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No trades were executed during this period.")
    
    # Performance comparison
    st.subheader("Performance Comparison")
    
    # Buy and hold comparison
    buy_hold_values = []
    initial_shares_bh = {symbol: (starting_balance / len(symbols)) / data[symbol].iloc[0] for symbol in symbols}
    
    for i, (date, row) in enumerate(data.iterrows()):
        bh_value = sum(initial_shares_bh[symbol] * row[symbol] for symbol in symbols)
        buy_hold_values.append(bh_value)
    
    bh_return = (buy_hold_values[-1] - starting_balance) / starting_balance * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Strategy Return", f"{total_return:.1f}%")
    with col2:
        st.metric("Buy & Hold Return", f"{bh_return:.1f}%")
    
    # Detailed results
    st.subheader("Detailed Results")
    # Add buy and hold comparison to results
    results_df['Buy_Hold_Value'] = buy_hold_values
    st.dataframe(results_df.round(4), use_container_width=True)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"leader_follower_backtest_{start_date}_{end_date}.csv",
        mime="text/csv"
    )

# Strategy explanation
with st.expander("Strategy Explanation"):
    st.markdown("""
    ### Leader-Follower Trading Strategy (Normalized Analysis)
    
    **Core Concept**: This strategy identifies wave-like patterns in sector movements by analyzing the relationship between a sector leader (NVDA) and its followers (AMD, MRVL, ASML) using normalized price data.
    
    **Normalization**: All stock prices are normalized to percentage returns from the starting date to ensure fair comparison regardless of absolute price levels.
    
    **Key Metrics**:
    - **Normalized Prices**: All stock prices converted to % change from the first day
    - **Negative Space**: The performance gap between the normalized leader and average normalized follower performance
    - **ROC Negative Space**: Rate of change in the negative space (momentum)
    - **ACC Negative Space**: Acceleration of the negative space change
    
    **Trading Phases**:
    1. **Inactive**: No clear trend, stay in cash
    2. **Initiation**: Leader starts outperforming, but followers haven't caught up yet
    3. **Early Inflection**: Followers start catching up, negative space shrinking (BUY signal)
    4. **Mid Inflection**: Continued convergence, hold positions
    5. **Late Inflection**: Convergence slowing down (SELL signal)
    6. **Interruption**: Divergence resuming, avoid positions
    
    **Entry**: Buy all stocks with equal weight when "Early Inflection" is detected
    **Exit**: Sell all positions when "Late Inflection" or "Interruption" is detected
    
    **Important**: While analysis uses normalized data for signal generation, actual trading uses real prices for position sizing and profit calculation.
    """)
