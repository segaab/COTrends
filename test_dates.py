import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bank 10‑Q Dashboard", layout="wide")
st.title("📄 Major Bank 10-Q Filing Analysis")

def parse_bac_filings(html_content):
    """Parse Bank of America 10-Q filings from the HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    filings = []
    
    # Find all rows containing 10-Q filings
    for row in soup.find_all('tr'):
        cells = row.find_all(['td', 'th'])
        if len(cells) >= 3:  # Ensure row has enough cells
            date_cell = cells[0].get_text(strip=True)
            form_cell = cells[1].get_text(strip=True)
            
            if '10-Q' in form_cell:
                try:
                    # Parse date
                    date = datetime.strptime(date_cell, '%m/%d/%y')
                    
                    # Get document links
                    doc_links = []
                    for link in row.find_all('a'):
                        if 'Documents' in link.get_text(strip=True):
                            doc_links.append(link.get('href', ''))
                    
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
    """Parse JP Morgan 10-Q filings from the HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    filings = []
    
    # Find all rows containing 10-Q filings
    for row in soup.find_all('tr'):
        date_cell = row.find('td', text=re.compile(r'\d{2}/\d{2}/\d{4}'))
        form_cell = row.find('td', text='10-Q')
        
        if date_cell and form_cell:
            try:
                # Parse date
                date = datetime.strptime(date_cell.get_text(strip=True), '%m/%d/%Y')
                
                # Get document links
                doc_links = []
                for link in row.find_all('a'):
                    if 'View HTML' in link.get_text(strip=True) or '.pdf' in link.get('href', ''):
                        doc_links.append(link.get('href', ''))
                
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

def analyze_filings(df):
    """Analyze filing patterns and create visualizations"""
    
    # Filing timeline
    fig_timeline = px.scatter(df, 
                            x='date', 
                            y='bank',
                            color='bank',
                            title='10-Q Filing Timeline',
                            labels={'date': 'Filing Date', 'bank': 'Bank'})
    st.plotly_chart(fig_timeline)
    
    # Quarterly filing patterns
    quarterly_counts = df.groupby(['year', 'quarter', 'bank']).size().reset_index(name='count')
    fig_quarterly = px.bar(quarterly_counts,
                          x='quarter',
                          y='count',
                          color='bank',
                          facet_col='year',
                          title='Quarterly Filing Patterns',
                          labels={'quarter': 'Quarter', 'count': 'Number of Filings'})
    st.plotly_chart(fig_quarterly)
    
    # Filing intervals
    df_sorted = df.sort_values('date')
    df_sorted['days_between_filings'] = df_sorted.groupby('bank')['date'].diff().dt.days
    
    fig_intervals = px.box(df_sorted[df_sorted['days_between_filings'].notna()],
                          x='bank',
                          y='days_between_filings',
                          title='Days Between Filings Distribution',
                          labels={'days_between_filings': 'Days', 'bank': 'Bank'})
    st.plotly_chart(fig_intervals)

def main():
    # Process Bank of America filings
    bac_df = parse_bac_filings(st.session_state.get('bac_html', ''))
    
    # Process JP Morgan filings
    jpm_df = parse_jpm_filings(st.session_state.get('jpm_html', ''))
    
    # Combine dataframes
    combined_df = pd.concat([bac_df, jpm_df], ignore_index=True)
    
    if not combined_df.empty:
        st.header("Filing Analysis")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Filings", len(combined_df))
        with col2:
            st.metric("Latest Filing", combined_df['date'].max().strftime('%Y-%m-%d'))
        with col3:
            st.metric("Banks Covered", combined_df['bank'].nunique())
        
        # Display detailed analysis
        analyze_filings(combined_df)
        
        # Raw data view
        if st.checkbox("Show Raw Data"):
            st.dataframe(combined_df.sort_values('date', ascending=False))
            
        # Download option
        csv = combined_df.to_csv(index=False)
        st.download_button(
            label="Download Filing Data",
            data=csv,
            file_name="bank_filings.csv",
            mime="text/csv"
        )
    else:
        st.warning("No filing data available. Please ensure the HTML content is properly loaded.")

if __name__ == "__main__":
    main()
