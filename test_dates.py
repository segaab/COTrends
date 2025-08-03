import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Bank 10‑Q Dashboard", layout="wide")
st.title("📄 Major Bank 10-Q Filing Analysis")

# -------------------------------
# 🔍 Parsers for different banks
# -------------------------------

def parse_bac_filings(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    filings = []
    
    for row in soup.find_all('tr'):
        cells = row.find_all(['td', 'th'])
        if len(cells) >= 3:
            date_cell = cells[0].get_text(strip=True)
            form_cell = cells[1].get_text(strip=True)
            if '10-Q' in form_cell:
                try:
                    date = datetime.strptime(date_cell, '%m/%d/%y')
                    doc_links = [link.get('href', '') for link in row.find_all('a') if 'Documents' in link.get_text(strip=True)]
                    filings.append({
                        'date': date,
                        'form': form_cell,
                        'bank': 'Bank of America',
                        'doc_links': doc_links,
                        'quarter': (date.month - 1) // 3 + 1,
                        'year': date.year
                    })
                except ValueError:
                    continue
    return pd.DataFrame(filings)

def parse_jpm_filings(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    filings = []
    for row in soup.find_all('tr'):
        date_cell = row.find('td', text=re.compile(r'\d{2}/\d{2}/\d{4}'))
        form_cell = row.find('td', text='10-Q')
        if date_cell and form_cell:
            try:
                date = datetime.strptime(date_cell.get_text(strip=True), '%m/%d/%Y')
                doc_links = [link.get('href', '') for link in row.find_all('a') if 'View HTML' in link.get_text(strip=True) or '.pdf' in link.get('href', '')]
                filings.append({
                    'date': date,
                    'form': '10-Q',
                    'bank': 'JPMorgan Chase',
                    'doc_links': doc_links,
                    'quarter': (date.month - 1) // 3 + 1,
                    'year': date.year
                })
            except ValueError:
                continue
    return pd.DataFrame(filings)

def parse_wfc_filings(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    filings = []

    for row in soup.find_all('tr'):
        cells = row.find_all(['td', 'th'])
        if len(cells) >= 2:
            date_text = cells[0].get_text(strip=True)
            form_text = cells[1].get_text(strip=True)
            if '10-Q' in form_text:
                try:
                    date = datetime.strptime(date_text, '%B %d, %Y')  # Example: "April 24, 2024"
                    links = [a.get('href', '') for a in row.find_all('a') if '10-Q' in a.get_text()]
                    filings.append({
                        'date': date,
                        'form': '10-Q',
                        'bank': 'Wells Fargo',
                        'doc_links': links,
                        'quarter': (date.month - 1) // 3 + 1,
                        'year': date.year
                    })
                except ValueError:
                    continue
    return pd.DataFrame(filings)

# -------------------------------
# 📊 Analysis Function
# -------------------------------

def analyze_filings(df):
    fig_timeline = px.scatter(df, x='date', y='bank', color='bank', title='📆 10-Q Filing Timeline')
    st.plotly_chart(fig_timeline)
    
    quarterly_counts = df.groupby(['year', 'quarter', 'bank']).size().reset_index(name='count')
    fig_quarterly = px.bar(quarterly_counts, x='quarter', y='count', color='bank', facet_col='year',
                           title='📊 Quarterly Filing Patterns', labels={'quarter': 'Quarter', 'count': 'Filings'})
    st.plotly_chart(fig_quarterly)

    df_sorted = df.sort_values('date')
    df_sorted['days_between_filings'] = df_sorted.groupby('bank')['date'].diff().dt.days
    fig_intervals = px.box(df_sorted[df_sorted['days_between_filings'].notna()],
                          x='bank', y='days_between_filings',
                          title='📉 Filing Interval Distribution',
                          labels={'days_between_filings': 'Days'})
    st.plotly_chart(fig_intervals)

# -------------------------------
# 🚀 Main Entry
# -------------------------------

def main():
    st.sidebar.header("Upload HTML Filing Tables")
    bac_html = st.sidebar.file_uploader("BAC Filing Table HTML", type=["html"])
    jpm_html = st.sidebar.file_uploader("JPM Filing Table HTML", type=["html"])
    wfc_html = st.sidebar.file_uploader("WFC Filing Table HTML", type=["html"])

    bac_df = parse_bac_filings(bac_html.read()) if bac_html else pd.DataFrame()
    jpm_df = parse_jpm_filings(jpm_html.read()) if jpm_html else pd.DataFrame()
    wfc_df = parse_wfc_filings(wfc_html.read()) if wfc_html else pd.DataFrame()

    combined_df = pd.concat([bac_df, jpm_df, wfc_df], ignore_index=True)
    
    if not combined_df.empty:
        st.header("Filing Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Filings", len(combined_df))
        with col2:
            st.metric("Latest Filing", combined_df['date'].max().strftime('%Y-%m-%d'))
        with col3:
            st.metric("Banks Covered", combined_df['bank'].nunique())
        
        analyze_filings(combined_df)
        
        if st.checkbox("Show Raw Data"):
            st.dataframe(combined_df.sort_values('date', ascending=False))

        csv = combined_df.to_csv(index=False)
        st.download_button("📥 Download Filing Data", data=csv, file_name="bank_filings.csv", mime="text/csv")
    else:
        st.info("👆 Upload at least one HTML table of 10-Q filings from a bank website.")

if __name__ == "__main__":
    main()
