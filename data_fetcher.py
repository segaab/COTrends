import os
from datetime import datetime, timedelta
import pandas as pd
from sodapy import Socrata

def get_last_two_reports(client):
    """
    Fetches the last two COT reports from the CFTC database.
    
    Parameters:
    client (Socrata): Initialized Socrata client
    
    Returns:
    list: List of dictionaries containing the latest and previous report data
    """
    try:
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

        # Retrieve the latest and previous reports
        latest_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{latest_tuesday_str}'")
        previous_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{previous_tuesday_str}'")

        if not latest_result or not previous_result:
            print("Warning: No data received for one or both dates")
            return []

        # Convert the results to DataFrames
        latest_df = pd.DataFrame.from_records(latest_result)
        previous_df = pd.DataFrame.from_records(previous_result)

        # Debug prints
        print("Latest DataFrame columns:", latest_df.columns.tolist() if not latest_df.empty else "Empty DataFrame")
        print("Previous DataFrame columns:", previous_df.columns.tolist() if not previous_df.empty else "Empty DataFrame")
        print("Latest DataFrame shape:", latest_df.shape)
        print("Previous DataFrame shape:", previous_df.shape)

        # Print some sample market names
        print("\nSample market names from latest data:")
        print(latest_df['market_and_exchange_names'].head().tolist())
        print("\nSample market names from previous data:")
        print(previous_df['market_and_exchange_names'].head().tolist())

        # Ensure we have the required columns
        required_columns = [
            'market_and_exchange_names',
            'open_interest_all',
            'noncomm_positions_long_all',
            'noncomm_positions_short_all',
            'noncomm_postions_spread_all',
            'comm_positions_long_all',
            'comm_positions_short_all',
            'tot_rept_positions_long_all',
            'tot_rept_positions_short',
            'nonrept_positions_long_all',
            'nonrept_positions_short_all'
        ]

        missing_columns = [col for col in required_columns if col not in latest_df.columns or col not in previous_df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return []

        # Create final data structure
        final_data = []
        for market_name in set(latest_df['market_and_exchange_names']).union(set(previous_df['market_and_exchange_names'])):
            latest_row = latest_df[latest_df['market_and_exchange_names'] == market_name]
            previous_row = previous_df[previous_df['market_and_exchange_names'] == market_name]
            
            latest_report = latest_row[required_columns].iloc[0].to_dict() if not latest_row.empty else None
            previous_report = previous_row[required_columns].iloc[0].to_dict() if not previous_row.empty else None
            
            if latest_report:
                latest_report.pop('market_and_exchange_names', None)
            if previous_report:
                previous_report.pop('market_and_exchange_names', None)
            
            final_data.append({
                'market_and_exchange_names': market_name,
                'latest_report': latest_report,
                'previous_report': previous_report
            })

        return final_data

    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return []

def asset_name_filter(data, asset_name=None):
    """
    Filters the report data based on the provided asset name.
    """
    if not asset_name:
        return data
    
    # Handle exact match
    filtered_data = [item for item in data if asset_name == item['market_and_exchange_names']]
    if filtered_data:
        return filtered_data
    
    # If no exact match, try case-insensitive partial match
    asset_name_lower = asset_name.lower()
    return [item for item in data if asset_name_lower in item['market_and_exchange_names'].lower()]

def filter_results(data, asset_name=None):
    """
    Returns a list of filtered market names.
    """
    filtered_data = asset_name_filter(data, asset_name)
    return [item['market_and_exchange_names'] for item in filtered_data] 