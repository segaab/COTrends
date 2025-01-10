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
    edt_now = datetime.utcnow() - timedelta(hours=4)

    # Find last Friday's date
    last_friday = edt_now - timedelta(days=(edt_now.weekday() - 4) % 7)

    # If it's Friday and before 3:30 PM ET, use the previous Friday
    report_time = last_friday.replace(hour=15, minute=30, second=0)
    if edt_now.weekday() == 4 and edt_now < report_time:
        last_friday = last_friday - timedelta(weeks=1)

    # Calculate the latest Tuesday and previous Tuesday
    latest_tuesday = last_friday - timedelta(days=3)
    previous_tuesday = latest_tuesday - timedelta(weeks=1)

    # Format dates as strings
    latest_tuesday_str = latest_tuesday.strftime('%Y-%m-%d')
    previous_tuesday_str = previous_tuesday.strftime('%Y-%m-%d')

    # Retrieve the latest and previous reports
    latest_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{latest_tuesday_str}'")
    previous_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{previous_tuesday_str}'")

    # Convert the results to DataFrames
    latest_df = pd.DataFrame.from_records(latest_result) if latest_result else pd.DataFrame()
    previous_df = pd.DataFrame.from_records(previous_result) if previous_result else pd.DataFrame()

    # Merging the two DataFrames
    merged_data = pd.merge(
        previous_df, 
        latest_df, 
        on='market_and_exchange_names', 
        how='outer', 
        suffixes=('_previous', '_latest')
    )

    # Create final data structure
    final_data = []
    for _, row in merged_data.iterrows():
        market_name = row['market_and_exchange_names']
        
        previous_report = row.filter(like='_previous').dropna().to_dict()
        latest_report = row.filter(like='_latest').dropna().to_dict()
        
        previous_report = {key.replace('_previous', ''): value for key, value in previous_report.items()}
        latest_report = {key.replace('_latest', ''): value for key, value in latest_report.items()}
        
        final_data.append({
            "market_and_exchange_names": market_name,
            "previous_report": previous_report,
            "latest_report": latest_report
        })

    return final_data

def asset_name_filter(data, asset_name=None):
    """
    Filters the report data based on the provided asset name.
    """
    if asset_name:
        return [item for item in data if asset_name == item['market_and_exchange_names']]
    return data

def filter_results(data, asset_name=None):
    """
    Returns a list of filtered market names.
    """
    filtered_data = asset_name_filter(data, asset_name)
    return [item['market_and_exchange_names'] for item in filtered_data] 