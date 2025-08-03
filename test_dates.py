import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from yahooquery import Ticker
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from seleniumwire import webdriver
from bs4 import BeautifulSoup
import requests
import re
import xml.etree.ElementTree as ET
from sec_api import QueryApi
import tabula
import numpy as np

# SEC API configuration
SEC_API_KEY = "b48426e1ec0d314f153b9d1b9f0421bc1aaa6779d25ea56bfc05bf235393478c"
query_api = QueryApi(api_key=SEC_API_KEY)

# Major Investment Banks and their CIK numbers
BANKS = {
    'JPMorgan Chase': '0000019617',
    'Goldman Sachs': '0000886982',
    'Morgan Stanley': '0000895421',
    'Bank of America': '0000070858',
    'Citigroup': '0000831001'
}

# Define major sectors and their corresponding ETFs
SECTOR_ETFS = {
    'Financial Services': 'XLF',
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Consumer Discretionary': 'XLY',
    'Industrial': 'XLI',
    'Energy': 'XLE',
    'Materials': 'XLB',
    'Consumer Staples': 'XLP',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE'
}

class SectorAnalyzer:
    def __init__(self):
        self.etfs = SECTOR_ETFS
        
    def fetch_sector_performance(self):
        performance_data = []
        
        # Batch process ETFs for better performance
        symbols = list(self.etfs.values())
        tickers = Ticker(symbols, asynchronous=True)
        
        # Get historical data for all ETFs
        hist_data = tickers.history(period='1y')
        
        # Get additional info for all ETFs
        quotes = tickers.quotes
        
        for sector, symbol in self.etfs.items():
            try:
                # Extract single symbol data
                symbol_hist = hist_data.xs(symbol, level=0) if isinstance(hist_data, pd.DataFrame) else None
                
                if symbol_hist is not None:
                    returns = symbol_hist['close'].pct_change()
                    ytd_return = ((symbol_hist['close'][-1] / symbol_hist['close'][0]) - 1) * 100
                    
                    performance = {
                        'Sector': sector,
                        'ETF': symbol,
                        'YTD_Return': ytd_return,
                        'Volatility': returns.std() * 100,
                        'Current_Price': symbol_hist['close'][-1],
                        'Volume': symbol_hist['volume'].mean(),
                        'Market_Cap': quotes[symbol]['marketCap'] if symbol in quotes else None
                    }
                    performance_data.append(performance)
            
            except Exception as e:
                st.warning(f"Error processing {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(performance_data)

    def get_sector_holdings(self, etf_symbol):
        try:
            ticker = Ticker(etf_symbol)
            holdings = ticker.fund_holding_info
            
            if holdings and 'holdings' in holdings[etf_symbol]:
                holdings_df = pd.DataFrame(holdings[etf_symbol]['holdings'])
                # Add additional metrics
                holdings_df['marketValue'] = holdings_df['holdingPercent'] * ticker.price[etf_symbol]['regularMarketPrice']
                return holdings_df
            
            return pd.DataFrame()
            
        except Exception as e:
            st.warning(f"Error fetching holdings for {etf_symbol}: {str(e)}")
            return pd.DataFrame()

    def get_sector_fundamentals(self, etf_symbol):
        try:
            ticker = Ticker(etf_symbol)
            
            # Fetch various fundamental data
            summary = ticker.summary_detail
            profile = ticker.fund_profile
            performance = ticker.fund_performance
            
            fundamentals = {
                'PE_Ratio': summary[etf_symbol].get('trailingPE'),
                'Expense_Ratio': profile[etf_symbol].get('feesExpensesInvestment', {}).get('annualReportExpenseRatio'),
                'Beta': summary[etf_symbol].get('beta'),
                'YTD_Return': performance[etf_symbol].get('performanceOverview', {}).get('ytd'),
                'Category': profile[etf_symbol].get('categoryName')
            }
            
            return fundamentals
            
        except Exception as e:
            st.warning(f"Error fetching fundamentals for {etf_symbol}: {str(e)}")
            return {}

# [Previous EDGARScraper and CreditExposureAnalyzer classes remain the same]

def main():
    st.title("Enhanced Sector Analysis Dashboard")
    
    sector_analyzer = SectorAnalyzer()
    
    # Sidebar for filters and controls
    st.sidebar.header("Analysis Controls")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Sector Performance", "Credit Exposure", "Combined Analysis"]
    )
    
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        list(SECTOR_ETFS.keys()),
        default=list(SECTOR_ETFS.keys())[:3]
    )
    
    # Fetch and cache sector performance data
    @st.cache_data(ttl=3600)
    def get_cached_sector_data():
        return sector_analyzer.fetch_sector_performance()
    
    sector_df = get_cached_sector_data()
    filtered_df = sector_df[sector_df['Sector'].isin(selected_sectors)]
    
    if analysis_type in ["Sector Performance", "Combined Analysis"]:
        st.header("Sector Performance Overview")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Best Performing Sector",
                f"{filtered_df.iloc[filtered_df['YTD_Return'].argmax()]['Sector']}", 
                f"{filtered_df['YTD_Return'].max():.2f}%"
            )
        with col2:
            st.metric(
                "Most Volatile Sector",
                f"{filtered_df.iloc[filtered_df['Volatility'].argmax()]['Sector']}", 
                f"{filtered_df['Volatility'].max():.2f}%"
            )
        with col3:
            st.metric(
                "Largest Sector by Market Cap",
                f"{filtered_df.iloc[filtered_df['Market_Cap'].argmax()]['Sector']}", 
                f"${filtered_df['Market_Cap'].max()/1e9:.2f}B"
            )
        
        # YTD Returns Chart
        fig_returns = px.bar(
            filtered_df,
            x='Sector',
            y='YTD_Return',
            title='YTD Returns by Sector',
            color='YTD_Return',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_returns)
        
        # Risk vs Return Analysis
        fig_scatter = px.scatter(
            filtered_df,
            x='Volatility',
            y='YTD_Return',
            size='Market_Cap',
            text='Sector',
            title='Risk vs Return Analysis',
            hover_data=['ETF'],
            size_max=60
        )
        st.plotly_chart(fig_scatter)
    
    if analysis_type in ["Credit Exposure", "Combined Analysis"]:
        st.header("Credit Exposure Analysis")
        
        # Fetch credit exposure data
        with st.spinner("Fetching credit exposure data from investment banks..."):
            credit_exposure_data = fetch_credit_exposure_data()
            sector_risk = analyze_sector_credit_risk(credit_exposure_data)
        
        # Display credit exposure analysis
        fig_heatmap = px.imshow(
            credit_exposure_data.pivot(index='sector', columns='bank', values='exposure'),
            aspect='auto',
            title='Credit Exposure Heatmap by Bank and Sector'
        )
        st.plotly_chart(fig_heatmap)
    
    # Detailed Sector Analysis
    st.header("Detailed Sector Analysis")
    selected_sector = st.selectbox("Select Sector for Detailed Analysis", selected_sectors)
    selected_etf = SECTOR_ETFS[selected_sector]
    
    # Fetch and display holdings
    with st.spinner("Fetching sector holdings..."):
        holdings_df = sector_analyzer.get_sector_holdings(selected_etf)
        fundamentals = sector_analyzer.get_sector_fundamentals(selected_etf)
        
        if not holdings_df.empty:
            # Display fundamental metrics
            st.subheader("ETF Fundamentals")
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("P/E Ratio", f"{fundamentals.get('PE_Ratio', 'N/A'):.2f}")
            with metrics_cols[1]:
                st.metric("Expense Ratio", f"{fundamentals.get('Expense_Ratio', 'N/A'):.2%}")
            with metrics_cols[2]:
                st.metric("Beta", f"{fundamentals.get('Beta', 'N/A'):.2f}")
            with metrics_cols[3]:
                st.metric("Category", fundamentals.get('Category', 'N/A'))
            
            # Display holdings
            st.subheader("Top Holdings")
            fig_holdings = px.treemap(
                holdings_df.head(10),
                path=[px.Constant("All"), 'symbol'],
                values='holdingPercent',
                title='Top 10 Holdings Distribution'
            )
            st.plotly_chart(fig_holdings)
            
            # Detailed holdings table
            st.dataframe(
                holdings_df.head(10).style.format({
                    'holdingPercent': '{:.2%}',
                    'marketValue': '${:,.2f}'
                })
            )

if __name__ == "__main__":
    main()
