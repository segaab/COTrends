import os
import base64
import streamlit as st
import streamlit.components.v1 as components
from sodapy import Socrata
from datetime import datetime
import pandas as pd
import altair as alt

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide",
    page_title="COTrend Analysis",
    page_icon="📊"
)

# Import the actual implementations from their respective files
from data_fetcher import get_last_two_reports
from analysis import aggregate_report_data, analyze_change, analyze_positions
from feedback_form import render_feature_form

# Cache for the report data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_cot_data(_client):  # Added underscore to tell Streamlit not to hash this parameter
    """Fetch and cache COT report data"""
    raw_data = get_last_two_reports(_client)
    if not raw_data:
        st.error("Failed to fetch COT report data")
    return raw_data

def render_asset_section(assets, section_title, raw_data):
    """Helper function to render asset sections consistently"""
    st.markdown(f'<h2 class="section-header">{section_title}</h2>', unsafe_allow_html=True)
    
    if not raw_data:
        st.error("No data available. Please try again later.")
        return
    
    for asset in assets:
        short_asset = asset.split(" -")[0]
        with st.expander(short_asset):
            try:
                # Process the data using functions from analysis.py
                asset_data = aggregate_report_data(raw_data, asset)
                if asset_data.empty:
                    st.warning(f"No data available for {asset}")
                    continue
                
                # Analyze changes and positions
                analytics_df = analyze_change(asset_data)
                position_data = analyze_positions(asset_data)

                # Display the data
                # Remove columns, use vertical layout
                # Format the analytics DataFrame for display
                display_df = analytics_df.copy()
                display_df['Net Change %'] = display_df['change_in_net_pct'].apply(
                    lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A"
                )
                display_df = display_df[['group', 'Net Change %']].rename(
                    columns={'group': 'Traders'}
                )
                # Show net change percentages with proper formatting
                st.table(display_df)

                # Chart below the table, full width
                st.write("Trader Positions Over Time (Normalized % Long)")
                chart_df = asset_data.sort_values('report_date_as_yyyy_mm_dd').tail(3)
                chart_df = chart_df.reset_index(drop=True)
                week_labels = ['Before LW', 'Last Week', 'This Week']
                chart_df['Week'] = week_labels[-len(chart_df):]
                def pct_long(long, short):
                    total = long + short
                    return (long / total * 100) if total > 0 else 0
                smart_money_pct = [pct_long(row['comm_positions_long_all'], row['comm_positions_short_all']) for _, row in chart_df.iterrows()]
                large_spec_pct = [pct_long(row['noncomm_positions_long_all'], row['noncomm_positions_short_all']) for _, row in chart_df.iterrows()]
                retail_pct = [pct_long(row['nonrept_positions_long_all'], row['nonrept_positions_short_all']) for _, row in chart_df.iterrows()]
                plot_df = pd.DataFrame({
                    'Week': chart_df['Week'],
                    'Smart Money': smart_money_pct,
                    'Large Speculators': large_spec_pct,
                    'Retail': retail_pct
                })
                plot_df = plot_df.melt(id_vars=['Week'], var_name='Trader', value_name='Long %')
                color_scale = alt.Scale(domain=["Smart Money", "Large Speculators", "Retail"],
                                        range=["#3fa9f5", "#f25c5c", "#ffffff"])
                line = alt.Chart(plot_df).mark_line(point=alt.OverlayMarkDef(filled=True, size=100)).encode(
                    x=alt.X('Week', sort=None, axis=alt.Axis(title=None, labelFontSize=16)),
                    y=alt.Y('Long %', axis=alt.Axis(title="% Long", labelFontSize=16), scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color('Trader', scale=color_scale, legend=alt.Legend(title="")),
                    detail='Trader'
                )
                points = alt.Chart(plot_df).mark_point(filled=True, size=100).encode(
                    x='Week',
                    y='Long %',
                    color=alt.Color('Trader', scale=color_scale, legend=None),
                    shape=alt.Shape('Trader', legend=None)
                )
                chart = (line + points).properties(height=350, width=600, background="#343333")
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying {asset}: {str(e)}")

# Fixed image handling function
def get_image_base64(image_path):
    """Convert image to base64 string with error handling"""
    try:
        # Try to find the image in different possible locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__)),  # Current directory
            os.path.join(os.getcwd(), "assets"),     # Assets subdirectory
            os.path.join(os.getcwd()),               # Root directory
            "/app/streamlit-app/assets"              # Streamlit Cloud path
        ]
        
        for base_dir in possible_paths:
            full_path = os.path.join(base_dir, image_path)
            if os.path.exists(full_path):
                with open(full_path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()
        
        raise FileNotFoundError(f"Image not found at any of the searched locations: {image_path}")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return ""

# Define Ko-fi button component
def kofi_button():
    kofi_html = """
        <div style="display: flex; justify-content: center; margin-top: 1rem;">
            <style>
                .kofi-button {
                    height: 54px;
                    max-width: 100%;
                    border: 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    transition: transform 0.2s ease;
                }
                .kofi-button:hover {
                    transform: translateY(-2px);
                }
                @media (max-width: 768px) {
                    .kofi-button {
                        height: 45px;
                    }
                }
            </style>
            <a href='https://ko-fi.com/Q5Q818G1MT' target='_blank'>
                <img class='kofi-button'
                     src='https://storage.ko-fi.com/cdn/kofi1.png?v=6' 
                     alt='Buy Me a Coffee at ko-fi.com' />
            </a>
        </div>
    """
    return components.html(kofi_html, height=75)

# Complete asset lists by categories with all instruments
crypto = [
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE",
    "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE"
]

commodities = [
    "GOLD - COMMODITY EXCHANGE INC.",
    "SILVER - COMMODITY EXCHANGE INC.",
    "WTI-FINANCIAL - NEW YORK MERCANTILE EXCHANGE"
]

forex = [
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "U.S. DOLLAR INDEX - ICE FUTURES U.S.",
    "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE"
]

indices = [
    "DOW JONES REIT - CHICAGO BOARD OF TRADE",
    "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
    "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE",
    "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE"
]

# Initialize Socrata client with error handling
@st.cache_resource
def init_client():
    try:
        MyAppToken = os.getenv('SODAPY_TOKEN')
        if not MyAppToken:
            st.error("Socrata API token not found. Please set SODAPY_TOKEN in your .env file.")
            return None
        
        # Initialize client with correct domain
        client = Socrata("publicreporting.cftc.gov", MyAppToken, timeout=30)
        
        # Test connection with a simple query to the correct endpoint
        test_result = client.get("6dca-aqww", limit=1000)
        if not test_result:
            st.error("Failed to connect to CFTC API. Please check your API token and network connection.")
            return None
        return client
    except Exception as e:
        st.error(f"Failed to initialize Socrata client: {str(e)}")
        return None

# Initialize client and fetch data once
client = init_client()
if client:
    raw_data = fetch_cot_data(client)
else:
    raw_data = None

# Add custom CSS for styling
st.markdown("""
    <style>
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem;
        background: linear-gradient(145deg, #2A2A2A, #1A1A1A);
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .logo-container {
        width: 100%;
        max-width: 400px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .banner-container {
        width: 100%;
        max-width: 400px;
        text-align: center;
        padding: 1rem;
    }
    .section-header {
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #404040;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #404040;
    }
    .footer-text {
        color: #808080;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header Component with error handling
try:
    logo_b64 = get_image_base64("assets/logo.png")
    banner_b64 = get_image_base64("assets/banner.png")
    
    st.markdown(f"""
        <div class="header-container">
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_b64}" alt="COTrend Logo">
            </div>
            <div class="banner-container">
                <img src="data:image/png;base64,{banner_b64}" alt="Find Your Edge">
            </div>
        </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not load header images: {str(e)}")
    # Fallback header
    st.title("COTrend Analysis")
    st.subheader("Find Your Edge in the Commitments of Traders Reports")

# Add Ko-fi button
kofi_button()

# Create two columns for layout
col1, col2 = st.columns(2)

# Display Crypto and Forex data
with col1:
    render_asset_section(crypto, "Crypto", raw_data)
    render_asset_section(forex, "Forex", raw_data)

# Display Commodities and Indices data
with col2:
    render_asset_section(commodities, "Commodities", raw_data)
    render_asset_section(indices, "Indices", raw_data)

# Add Feature Request Form
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
render_feature_form()

# Add Footer
st.markdown("""
    <div class="footer">
        <span class="footer-text">Powered by BiltP2P • Data from CFTC • Last updated: {}</span>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

# Add report dates to debug info
with st.expander("Debug Info", expanded=False):
    st.write("Environment Variables:", {k: v for k, v in os.environ.items() if "TOKEN" not in k})
    st.write("Current Directory:", os.getcwd())
    st.write("Files in Directory:", os.listdir())
    if 'client' in locals():
        st.write("Socrata Client:", "Initialized" if client else "Not Initialized")
    if raw_data:
        dates = {r['report_date_as_yyyy_mm_dd'] for r in raw_data}
        st.write("Report Dates:", sorted(list(dates)))