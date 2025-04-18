import os
from datetime import datetime, timedelta
from sodapy import Socrata
import pandas as pd

def init_client():
    """Initialize and test the Socrata client"""
    try:
        MyAppToken = os.getenv('SODAPY_TOKEN')
        if not MyAppToken:
            print("Error: SODAPY_TOKEN not found in environment variables")
            return None
        
        print("Initializing Socrata client...")
        client = Socrata("publicreporting.cftc.gov", MyAppToken, timeout=30)
        return client
    except Exception as e:
        print(f"Error initializing client: {str(e)}")
        return None

def get_bitcoin_data(client):
    """Get the last two reports for Bitcoin"""
    try:
        # Get current time in EDT
        edt_now = datetime.utcnow() - timedelta(hours=4)
        
        # Find the most recent Friday
        days_since_friday = (edt_now.weekday() - 4) % 7
        last_friday = edt_now - timedelta(days=days_since_friday)
        
        # If it's Friday but before 3:30 PM EDT, use the previous Friday
        if edt_now.weekday() == 4:
            current_time = edt_now.time()
            cutoff_time = datetime.strptime("15:30", "%H:%M").time()
            if current_time < cutoff_time:
                last_friday = last_friday - timedelta(weeks=1)
        
        # Go back one more Friday if we're on a weekend
        if edt_now.weekday() >= 5:
            last_friday = last_friday - timedelta(weeks=1)
        
        # The report is for the previous Tuesday
        latest_tuesday = last_friday - timedelta(days=3)
        previous_tuesday = latest_tuesday - timedelta(weeks=1)
        
        # Format dates as strings
        latest_tuesday_str = latest_tuesday.strftime('%Y-%m-%d')
        previous_tuesday_str = previous_tuesday.strftime('%Y-%m-%d')
        
        print(f"Fetching data for dates: {latest_tuesday_str} and {previous_tuesday_str}")

        # Define required fields
        required_fields = [
            "market_and_exchange_names",
            "report_date_as_yyyy_mm_dd",
            "noncomm_positions_long_all",
            "noncomm_positions_short_all",
            "comm_positions_long_all",
            "comm_positions_short_all",
            "nonrept_positions_long_all",
            "nonrept_positions_short_all"
        ]

        # Build the select query
        select_query = ",".join(required_fields)
        
        # Fetch data for both dates
        latest_result = client.get(
            "6dca-aqww",
            where=f"report_date_as_yyyy_mm_dd = '{latest_tuesday_str}' AND market_and_exchange_names = 'BITCOIN - CHICAGO MERCANTILE EXCHANGE'",
            select=select_query,
            limit=1
        )
        
        previous_result = client.get(
            "6dca-aqww",
            where=f"report_date_as_yyyy_mm_dd = '{previous_tuesday_str}' AND market_and_exchange_names = 'BITCOIN - CHICAGO MERCANTILE EXCHANGE'",
            select=select_query,
            limit=1
        )
        
        return latest_result[0] if latest_result else None, previous_result[0] if previous_result else None
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None, None

def calculate_net_changes(latest, previous):
    """Calculate net changes for all trader groups"""
    if not latest or not previous:
        print("Missing data for one or both reports")
        return
    
    print("\nLatest Report Data:")
    print(f"Date: {latest['report_date_as_yyyy_mm_dd']}")
    print(f"Non-Commercial Long: {latest['noncomm_positions_long_all']}")
    print(f"Non-Commercial Short: {latest['noncomm_positions_short_all']}")
    print(f"Commercial Long: {latest['comm_positions_long_all']}")
    print(f"Commercial Short: {latest['comm_positions_short_all']}")
    print(f"Non-Reportable Long: {latest['nonrept_positions_long_all']}")
    print(f"Non-Reportable Short: {latest['nonrept_positions_short_all']}")
    
    print("\nPrevious Report Data:")
    print(f"Date: {previous['report_date_as_yyyy_mm_dd']}")
    print(f"Non-Commercial Long: {previous['noncomm_positions_long_all']}")
    print(f"Non-Commercial Short: {previous['noncomm_positions_short_all']}")
    print(f"Commercial Long: {previous['comm_positions_long_all']}")
    print(f"Commercial Short: {previous['comm_positions_short_all']}")
    print(f"Non-Reportable Long: {previous['nonrept_positions_long_all']}")
    print(f"Non-Reportable Short: {previous['nonrept_positions_short_all']}")
    
    # Calculate net positions using the new ratio formula
    def calculate_net_change(latest_long, latest_short, prev_long, prev_short):
        # Convert strings to integers
        latest_long = int(latest_long)
        latest_short = int(latest_short)
        prev_long = int(prev_long)
        prev_short = int(prev_short)
        
        # Calculate total positions for each period
        latest_total = latest_long + latest_short
        prev_total = prev_long + prev_short
        
        # Calculate the ratios
        latest_ratio = (latest_long - latest_short) / latest_total if latest_total != 0 else 0
        prev_ratio = (prev_long - prev_short) / prev_total if prev_total != 0 else 0
        
        print(f"\nDetailed calculation:")
        print(f"Latest: ({latest_long} - {latest_short}) ÷ {latest_total}")
        print(f"Previous: ({prev_long} - {prev_short}) ÷ {prev_total}")
        print(f"Latest ratio: {latest_ratio}")
        print(f"Previous ratio: {prev_ratio}")
        
        # Calculate the difference in ratios
        ratio_change = latest_ratio - prev_ratio
        
        return ratio_change * 100  # Convert to percentage
    
    print("\nCalculated Net Changes:")
    print("\nNon-Commercial:")
    noncomm_change = calculate_net_change(
        latest['noncomm_positions_long_all'],
        latest['noncomm_positions_short_all'],
        previous['noncomm_positions_long_all'],
        previous['noncomm_positions_short_all']
    )
    print(f"Final Change: {noncomm_change:.2f}%")
    
    print("\nCommercial:")
    comm_change = calculate_net_change(
        latest['comm_positions_long_all'],
        latest['comm_positions_short_all'],
        previous['comm_positions_long_all'],
        previous['comm_positions_short_all']
    )
    print(f"Final Change: {comm_change:.2f}%")
    
    print("\nNon-Reportable:")
    nonrept_change = calculate_net_change(
        latest['nonrept_positions_long_all'],
        latest['nonrept_positions_short_all'],
        previous['nonrept_positions_long_all'],
        previous['nonrept_positions_short_all']
    )
    print(f"Final Change: {nonrept_change:.2f}%")

def main():
    print("Starting Bitcoin Position Change Test...")
    client = init_client()
    if client:
        latest, previous = get_bitcoin_data(client)
        if latest and previous:
            calculate_net_changes(latest, previous)
        else:
            print("Failed to fetch data for both reports")
    else:
        print("Failed to initialize client")

if __name__ == "__main__":
    main() 