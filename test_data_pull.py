import os
import time
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
        
        # Test connection
        print("Testing API connection...")
        test_result = client.get("6dca-aqww", limit=1)
        if not test_result:
            print("Error: Failed to connect to CFTC API")
            return None
            
        print("API connection successful!")
        return client
    except Exception as e:
        print(f"Error initializing client: {str(e)}")
        return None

def get_last_two_reports(client):
    """Test function to fetch and log the last two reports"""
    if not client:
        print("Error: No Socrata client available")
        return []
        
    try:
        start_time = time.time()
        print("\nStarting data fetch...")
        
        edt_now = datetime.utcnow() - timedelta(hours=4)
        
        # Find the most recent Friday (going backwards)
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
        
        print("\nFetching latest report...")
        latest_result = client.get(
            "6dca-aqww",
            where=f"report_date_as_yyyy_mm_dd = '{latest_tuesday_str}'",
            select=select_query,
            limit=1000
        )
        print(f"Latest report fetched: {len(latest_result) if latest_result else 0} records")
        
        print("\nFetching previous report...")
        previous_result = client.get(
            "6dca-aqww",
            where=f"report_date_as_yyyy_mm_dd = '{previous_tuesday_str}'",
            select=select_query,
            limit=1000
        )
        print(f"Previous report fetched: {len(previous_result) if previous_result else 0} records")
        
        # Combine results
        results = []
        if latest_result:
            results.extend(latest_result)
        if previous_result:
            results.extend(previous_result)
            
        # Log sample data
        if results:
            print("\nSample data from first record:")
            sample = results[0]
            for field in required_fields:
                print(f"{field}: {sample.get(field, 'N/A')}")
            
            # Check for specific assets
            test_assets = [
                "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
                "ETHER - CHICAGO MERCANTILE EXCHANGE",
                "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE",
                "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE",
                "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE"
            ]
            
            print("\nChecking for specific assets:")
            for asset in test_assets:
                asset_data = [r for r in results if r["market_and_exchange_names"] == asset]
                print(f"{asset}: {len(asset_data)} records found")
                if asset_data:
                    print(f"Sample data for {asset}:")
                    for field in required_fields:
                        print(f"{field}: {asset_data[0].get(field, 'N/A')}")
        
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        
        return results
            
    except Exception as e:
        print(f"Error in data fetching: {str(e)}")
        return []

def main():
    print("Starting CFTC API Test...")
    client = init_client()
    if client:
        data = get_last_two_reports(client)
        if data:
            print(f"\nTotal records fetched: {len(data)}")
        else:
            print("\nNo data was fetched")
    else:
        print("\nTest failed: Could not initialize client")

if __name__ == "__main__":
    main() 