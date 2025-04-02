import os
import base64
import streamlit as st
import streamlit.components.v1 as components
from sodapy import Socrata
from datetime import datetime

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide",
    page_title="COTrend Analysis",
    page_icon="📊"
)

# For demo purposes, we'll include mock implementations of your modules
# In your actual code, you would import these from separate files
from data_fetcher import get_last_two_reports
from analysis import aggregate_report_data, analyze_change, analyze_positions
from feedback_form import render_feature_form

# Mock implementations for demo purposes
def get_last_two_reports(client):
    # This would normally fetch from the CFTC API
    # For demo, we return mock data for all instruments
    return [
        # Crypto instruments
        {"market_and_exchange_names": "BITCOIN - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "1000", "noncomm_positions_short_all": "500", "comm_positions_long_all": "800", "comm_positions_short_all": "600"},
        {"market_and_exchange_names": "BITCOIN - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-08", "noncomm_positions_long_all": "1200", "noncomm_positions_short_all": "400", "comm_positions_long_all": "900", "comm_positions_short_all": "700"},
        {"market_and_exchange_names": "ETHER - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "800", "noncomm_positions_short_all": "400", "comm_positions_long_all": "600", "comm_positions_short_all": "500"},
        {"market_and_exchange_names": "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "500", "noncomm_positions_short_all": "250", "comm_positions_long_all": "400", "comm_positions_short_all": "300"},
        {"market_and_exchange_names": "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "400", "noncomm_positions_short_all": "200", "comm_positions_long_all": "300", "comm_positions_short_all": "250"},
        
        # Commodity instruments
        {"market_and_exchange_names": "GOLD - COMMODITY EXCHANGE INC.", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "2000", "noncomm_positions_short_all": "1000", "comm_positions_long_all": "1500", "comm_positions_short_all": "1200"},
        {"market_and_exchange_names": "SILVER - COMMODITY EXCHANGE INC.", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "1500", "noncomm_positions_short_all": "750", "comm_positions_long_all": "1200", "comm_positions_short_all": "900"},
        {"market_and_exchange_names": "WTI-FINANCIAL - NEW YORK MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "3000", "noncomm_positions_short_all": "1500", "comm_positions_long_all": "2500", "comm_positions_short_all": "2000"},
        
        # Forex instruments
        {"market_and_exchange_names": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "2500", "noncomm_positions_short_all": "1250", "comm_positions_long_all": "2000", "comm_positions_short_all": "1600"},
        {"market_and_exchange_names": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "1800", "noncomm_positions_short_all": "900", "comm_positions_long_all": "1500", "comm_positions_short_all": "1200"},
        {"market_and_exchange_names": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "2000", "noncomm_positions_short_all": "1000", "comm_positions_long_all": "1700", "comm_positions_short_all": "1300"},
        {"market_and_exchange_names": "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "2200", "noncomm_positions_short_all": "1100", "comm_positions_long_all": "1800", "comm_positions_short_all": "1400"},
        {"market_and_exchange_names": "EURO FX - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "3000", "noncomm_positions_short_all": "1500", "comm_positions_long_all": "2500", "comm_positions_short_all": "2000"},
        {"market_and_exchange_names": "U.S. DOLLAR INDEX - ICE FUTURES U.S.", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "2800", "noncomm_positions_short_all": "1400", "comm_positions_long_all": "2300", "comm_positions_short_all": "1800"},
        {"market_and_exchange_names": "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "1200", "noncomm_positions_short_all": "600", "comm_positions_long_all": "1000", "comm_positions_short_all": "800"},
        {"market_and_exchange_names": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "1500", "noncomm_positions_short_all": "750", "comm_positions_long_all": "1300", "comm_positions_short_all": "1000"},
        
        # Index instruments
        {"market_and_exchange_names": "DOW JONES REIT - CHICAGO BOARD OF TRADE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "1000", "noncomm_positions_short_all": "500", "comm_positions_long_all": "800", "comm_positions_short_all": "600"},
        {"market_and_exchange_names": "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "5000", "noncomm_positions_short_all": "2500", "comm_positions_long_all": "4000", "comm_positions_short_all": "3000"},
        {"market_and_exchange_names": "NASDAQ-100 STOCK INDEX (MINI) - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "4000", "noncomm_positions_short_all": "2000", "comm_positions_long_all": "3500", "comm_positions_short_all": "2800"},
        {"market_and_exchange_names": "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE", "report_date_as_yyyy_mm_dd": "2023-01-01", "noncomm_positions_long_all": "1500", "noncomm_positions_short_all": "750", "comm_positions_long_all": "1200", "comm_positions_short_all": "900"}
    ]

def aggregate_report_data(cot_data, asset):
    # Filter data for the specific asset
    return [d for d in cot_data if d["market_and_exchange_names"] == asset]

def analyze_change(asset_data):
    # Calculate percentage changes between reports
    if len(asset_data) < 2:
        return {
            "group": ["Non-Commercial", "Commercial"],
            "change_in_net_pct": [0, 0]
        }
    
    # Calculate net positions and changes
    noncomm_net1 = int(asset_data[0]["noncomm_positions_long_all"]) - int(asset_data[0]["noncomm_positions_short_all"])
    noncomm_net2 = int(asset_data[1]["noncomm_positions_long_all"]) - int(asset_data[1]["noncomm_positions_short_all"])
    noncomm_change = ((noncomm_net2 - noncomm_net1) / noncomm_net1 * 100) if noncomm_net1 != 0 else 0
    
    comm_net1 = int(asset_data[0]["comm_positions_long_all"]) - int(asset_data[0]["comm_positions_short_all"])
    comm_net2 = int(asset_data[1]["comm_positions_long_all"]) - int(asset_data[1]["comm_positions_short_all"])
    comm_change = ((comm_net2 - comm_net1) / comm_net1 * 100) if comm_net1 != 0 else 0
    
    return {
        "group": ["Non-Commercial", "Commercial"],
        "change_in_net_pct": [noncomm_change, comm_change]
    }

def analyze_positions(analytics_df):
    # Prepare data for visualization
    return {
        "Non-Commercial": [100, 100 + analytics_df["change_in_net_pct"][0]],
        "Commercial": [100, 100 + analytics_df["change_in_net_pct"][1]]
    }

def render_feature_form():
    with st.form("feature_request"):
        st.write("Suggest a new feature:")
        feature = st.text_area("Describe your feature request")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("Thank you for your feedback!")

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
    "ETHER - CHICAGO MERCANTILE EXCHANGE",
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
            st.warning("Running without Socrata API token - using mock data")
            return None
        
        client = Socrata("publicreporting.cftc.gov", MyAppToken, timeout=30)
        # Test connection with a simple query
        client.get("publicreporting.cftc.gov", limit=1)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Socrata client: {str(e)}")
        return None

client = init_client()

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

def render_asset_section(assets, section_title):
    """Helper function to render asset sections consistently"""
    st.markdown(f'<h2 class="section-header">{section_title}</h2>', unsafe_allow_html=True)
    
    for asset in assets:
        short_asset = asset.split(" -")[0]
        with st.expander(short_asset):
            try:
                asset_data = aggregate_report_data(get_last_two_reports(client), asset)
                
                if not asset_data:
                    st.warning(f"No data available for {asset}")
                    continue
                    
                analytics_df = analyze_change(asset_data)
                chart_data = analyze_positions(analytics_df)
                
                # Convert to DataFrame
                import pandas as pd
                df = pd.DataFrame({
                    'Traders': analytics_df['group'],
                    'Net Change %': analytics_df['change_in_net_pct']
                })

                _col = st.columns(2)
                with _col[0]:
                    st.table(df)
                with _col[1]:
                    st.bar_chart(
                        chart_data,
                        color=["#FFFFFF", "#808080"],
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error displaying {asset}: {str(e)}")

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
        <span class="footer-text">Powered by BiltP2P • Data from CFTC • Last updated: {}</span>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

# Debug information (can be removed in production)
with st.expander("Debug Info", expanded=False):
    st.write("Environment Variables:", {k: v for k, v in os.environ.items() if "TOKEN" not in k})
    st.write("Current Directory:", os.getcwd())
    st.write("Files in Directory:", os.listdir())
    if 'client' in locals():
        st.write("Socrata Client:", "Initialized" if client else "Not Initialized")