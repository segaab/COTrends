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
    if not client:
        print("Error: No Socrata client available")
        return []
        
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

        # Retrieve the latest and previous reports using the correct endpoint
        try:
            # Build the select query to get only required fields
            select_query = ",".join(required_fields)
            
            latest_result = client.get(
                "6dca-aqww",
                where=f"report_date_as_yyyy_mm_dd = '{latest_tuesday_str}'",
                select=select_query
            )
            
            previous_result = client.get(
                "6dca-aqww",
                where=f"report_date_as_yyyy_mm_dd = '{previous_tuesday_str}'",
                select=select_query
            )
            
            if not latest_result:
                print(f"Warning: No data received for latest date: {latest_tuesday_str}")
            if not previous_result:
                print(f"Warning: No data received for previous date: {previous_tuesday_str}")
                
            if not latest_result and not previous_result:
                print("Error: No data received for either date")
                return []
                
            # Combine results if we have data for at least one date
            results = []
            if latest_result:
                results.extend(latest_result)
            if previous_result:
                results.extend(previous_result)
                
            # Validate the data structure
            for item in results:
                for field in required_fields:
                    if field not in item:
                        print(f"Warning: Missing field {field} in data")
                        item[field] = "0"  # Default to 0 if field is missing
                
            return results
            
        except Exception as e:
            print(f"Error fetching data from CFTC API: {str(e)}")
            return []
            
    except Exception as e:
        print(f"Error in date calculation or data fetching: {str(e)}")
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