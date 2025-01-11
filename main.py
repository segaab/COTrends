import os
import base64
import streamlit as st
import streamlit.components.v1 as components
from sodapy import Socrata
from data_fetcher import get_last_two_reports
from analysis import aggregate_report_data, analyze_change, analyze_positions
from feedback_form import render_feature_form

def get_image_base64(image_path):
    """Convert image to base64 string"""
    with open(os.path.join(os.path.dirname(__file__), image_path), "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Define Ko-fi button component
def kofi_button():
    kofi_html = """
        <div style="display: flex; justify-content: center; margin-top: 1rem;">
            <style>
                .kofi-button {
                    height: 54px; /* 1.5x the original height */
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
                        height: 45px; /* Slightly smaller on mobile */
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
    return components.html(kofi_html, height=75)  # Increased height to accommodate larger button

# Asset lists by categories
crypto = ["BITCOIN - CHICAGO MERCANTILE EXCHANGE", "ETHER - CHICAGO MERCANTILE EXCHANGE", "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE", "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE"]
commodities = ["GOLD - COMMODITY EXCHANGE INC.", "SILVER - COMMODITY EXCHANGE INC.", "WTI-FINANCIAL - NEW YORK MERCANTILE EXCHANGE"]
forex = ["JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE", "EURO FX - CHICAGO MERCANTILE EXCHANGE", "U.S. DOLLAR INDEX - ICE FUTURES U.S.", "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE", "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE"]
indices = ["DOW JONES REIT - CHICAGO BOARD OF TRADE", "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE", "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE"]

# Initialize Socrata client
MyAppToken = os.getenv('SODAPY_TOKEN')
client = Socrata("publicreporting.cftc.gov", MyAppToken if MyAppToken else None)

try:
    # Pull COT data 
    cot_data = get_last_two_reports(client)
    if not cot_data:
        st.error("Unable to fetch data from CFTC. Please try again later.")
        st.stop()
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()

# Setup Streamlit page
st.set_page_config(layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0.5rem;
    }
    
    /* Header Styles */
    .header-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem;
        background: linear-gradient(145deg, #2A2A2A, #1A1A1A);
        border: 1px solid #333333;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        overflow: hidden;
        max-width: 100%;
        width: 100%;
        position: relative;
    }
    
    /* Add subtle shine effect */
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            120deg,
            transparent,
            rgba(255, 255, 255, 0.03),
            transparent
        );
        transition: 0.5s;
    }
    
    .header-container:hover::before {
        left: 100%;
    }
    
    .logo-container {
        width: 100%;
        max-width: 400px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .logo-container img {
        width: 100%;
        height: auto;
        max-width: 300px;
        object-fit: contain;
    }
    
    .banner-container {
        width: 100%;
        max-width: 400px;
        text-align: center;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1.5rem;
    }
    
    .banner-container img {
        width: 100%;
        height: auto;
        max-width: 250px;
        object-fit: contain;
    }
    
    /* Ko-fi button container */
    .kofi-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        max-width: 200px;
        margin: 1rem auto;
    }
    
    .kofi-container img {
        width: 100%;
        height: auto;
        max-width: 150px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    
    .kofi-container img:hover {
        transform: translateY(-2px);
    }
    
    /* Responsive styles */
    @media (min-width: 768px) {
        .header-container {
            flex-direction: row;
            justify-content: space-between;
            padding: 3rem;
            min-height: 300px;
            align-items: center;
        }
        
        .logo-container {
            width: 45%;
            max-width: 500px;
            margin-bottom: 0;
            justify-content: flex-start;
        }
        
        .banner-container {
            width: 45%;
            max-width: 500px;
            align-items: flex-end;
            text-align: right;
            padding-left: 2rem;
        }
        
        .logo-container img {
            max-width: 400px;
        }
        
        .banner-container img {
            max-width: 350px;
        }
    }
    
    @media (min-width: 1200px) {
        .logo-container img {
            max-width: 500px;
        }
        
        .banner-container img {
            max-width: 400px;
        }
    }
    
    @media (max-width: 480px) {
        .header-container {
            padding: 1.5rem;
        }
        
        .logo-container {
            padding: 0.5rem;
        }
        
        .banner-container {
            padding: 0.5rem;
        }
        
        .logo-container img {
            max-width: 250px;
        }
        
        .banner-container img {
            max-width: 200px;
        }
    }
    
    /* Section Styles */
    .section-header {
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #404040;
    }
    
    /* Expander Styles */
    .stExpander {
        background-color: #2D2D2D !important;
        border: 1px solid #404040 !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stExpander:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    /* Table Styles */
    .stTable {
        background-color: #2D2D2D !important;
        border-radius: 8px !important;
    }
    
    .dataframe {
        font-size: 0.9rem !important;
    }
    
    /* Chart Styles */
    .stChart {
        background-color: #2D2D2D !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Footer Styles */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #404040;
    }
    
    .footer-text {
        color: #808080;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: color 0.2s ease;
    }
    
    .footer-text:hover {
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Header Component
st.markdown("""
    <div class="header-container">
        <div class="logo-container">
            <img src="data:image/png;base64,{}" alt="COTrend Logo">
        </div>
        <div class="banner-container">
            <img src="data:image/png;base64,{}" alt="Find Your Edge">
        </div>
    </div>
    """.format(
        get_image_base64("assets/logo.png"),
        get_image_base64("assets/banner.png")
    ), unsafe_allow_html=True)

# Add Ko-fi button below the header
kofi_button()

# Create two columns for layout
col1, col2 = st.columns(2)

def render_asset_section(assets, section_title):
    """Helper function to render asset sections consistently"""
    st.markdown(f'<h2 class="section-header">{section_title}</h2>', unsafe_allow_html=True)
    for asset in assets:
        short_asset = asset.split(" -")[0]
        with st.expander(short_asset):
            asset_data = aggregate_report_data(cot_data, asset)
            analytics_df = analyze_change(asset_data)
            chart_data = analyze_positions(analytics_df)
            analytics_df = analytics_df.rename(columns={'group':'Traders','change_in_net_pct':"Net Change %"})
            
            _col = st.columns(2)
            with _col[0]:
                st.table(analytics_df[['Traders', 'Net Change %']])
            with _col[1]:
                st.bar_chart(
                    chart_data,
                    color=["#FFFFFF", "#808080"],
                    use_container_width=True
                )

# Display Crypto and Forex data
with col1:
    render_asset_section(crypto, "Crypto")
    render_asset_section(forex, "Forex")

# Display Commodities and Indices data
with col2:
    render_asset_section(commodities, "Commodities")
    render_asset_section(indices, "Indices")

# Add Feature Request Form
st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
render_feature_form()

# Add Footer
st.markdown("""
    <div class="footer">
        <span class="footer-text">Powered by BiltP2P</span>
    </div>
    """, unsafe_allow_html=True)