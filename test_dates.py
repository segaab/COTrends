import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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

class EDGARScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Your Name yourname@email.com'
        }
        self.base_url = "https://www.sec.gov/Archives/"
        
    def get_latest_10q_filing(self, cik):
        query = {
            "query": {
                "query_string": {
                    "query": f"cik:{cik} AND formType:\"10-Q\""
                }
            },
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        response = query_api.get_filings(query)
        if response['total']['value'] > 0:
            return response['filings'][0]
        return None

    def extract_credit_exposure_table(self, filing_url):
        response = requests.get(filing_url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for tables containing credit exposure information
        tables = soup.find_all('table')
        credit_tables = []
        
        for table in tables:
            text = table.get_text().lower()
            if any(keyword in text for keyword in ['credit exposure', 'wholesale', 'credit risk']):
                df = pd.read_html(str(table))[0]
                credit_tables.append(df)
        
        return credit_tables

class CreditExposureAnalyzer:
    def __init__(self):
        self.scraper = EDGARScraper()
        
    def get_bank_credit_exposure(self, bank_name, cik):
        filing = self.scraper.get_latest_10q_filing(cik)
        if not filing:
            return None
        
        tables = self.scraper.extract_credit_exposure_table(filing['documentUrl'])
        if not tables:
            return None
        
        # Process and standardize the credit exposure data
        exposure_data = self.standardize_credit_data(tables, bank_name)
        return exposure_data
    
    def standardize_credit_data(self, tables, bank_name):
        # Implement standardization logic for different bank formats
        standardized_data = pd.DataFrame()
        
        for table in tables:
            # Clean and standardize column names
            table.columns = table.columns.str.lower().str.replace(' ', '_')
            
            # Look for relevant columns
            exposure_cols = [col for col in table.columns if 'exposure' in col]
            sector_cols = [col for col in table.columns if 'industry' in col or 'sector' in col]
            
            if exposure_cols and sector_cols:
                subset = table[[sector_cols[0], exposure_cols[0]]].copy()
                subset.columns = ['sector', 'exposure']
                subset['bank'] = bank_name
                standardized_data = pd.concat([standardized_data, subset])
        
        return standardized_data

def fetch_credit_exposure_data():
    analyzer = CreditExposureAnalyzer()
    all_exposure_data = pd.DataFrame()
    
    for bank_name, cik in BANKS.items():
        exposure_data = analyzer.get_bank_credit_exposure(bank_name, cik)
        if exposure_data is not None:
            all_exposure_data = pd.concat([all_exposure_data, exposure_data])
    
    return all_exposure_data

def analyze_sector_credit_risk(exposure_data):
    # Aggregate exposure by sector across banks
    sector_exposure = exposure_data.groupby('sector')['exposure'].sum().reset_index()
    
    # Calculate risk metrics
    sector_exposure['exposure_pct'] = sector_exposure['exposure'] / sector_exposure['exposure'].sum() * 100
    sector_exposure['risk_score'] = sector_exposure['exposure_pct'] * \
                                  sector_exposure['exposure'].rank(pct=True)
    
    return sector_exposure

def main():
    st.title("Sector Credit Exposure Analysis Dashboard")
    
    # Fetch credit exposure data
    with st.spinner("Fetching credit exposure data from investment banks..."):
        credit_exposure_data = fetch_credit_exposure_data()
        sector_risk = analyze_sector_credit_risk(credit_exposure_data)
    
    # Display overall credit exposure analysis
    st.header("Investment Bank Credit Exposure Overview")
    
    # Credit exposure by bank and sector
    fig_heatmap = px.imshow(
        credit_exposure_data.pivot(index='sector', columns='bank', values='exposure'),
        aspect='auto',
        title='Credit Exposure Heatmap by Bank and Sector'
    )
    st.plotly_chart(fig_heatmap)
    
    # Sector risk analysis
    st.header("Sector Risk Analysis")
    fig_risk = px.scatter(
        sector_risk,
        x='exposure_pct',
        y='risk_score',
        text='sector',
        size='exposure',
        title='Sector Risk Assessment',
        labels={
            'exposure_pct': 'Exposure Percentage',
            'risk_score': 'Risk Score'
        }
    )
    st.plotly_chart(fig_risk)
    
    # Detailed exposure tables
    st.header("Detailed Credit Exposure Data")
    st.dataframe(credit_exposure_data)
    
    # Download data
    csv = credit_exposure_data.to_csv(index=False)
    st.download_button(
        label="Download Credit Exposure Data",
        data=csv,
        file_name="credit_exposure_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
