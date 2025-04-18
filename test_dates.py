from datetime import datetime, timedelta
import os
from sodapy import Socrata

def calculate_report_dates(current_date=None):
    """
    Calculate the expected report dates based on a given date.
    Returns the dates of the two most recent Friday reports.
    """
    # Use provided date or current date
    if current_date is None:
        current_date = datetime.now()
    
    print(f"\nTesting with current date: {current_date.strftime('%Y-%m-%d %H:%M')} EDT")
    
    # Find the most recent Friday
    days_since_friday = (current_date.weekday() - 4) % 7
    last_friday = current_date - timedelta(days=days_since_friday)
    print(f"Last Friday: {last_friday.strftime('%Y-%m-%d')}")
    
    # If it's Friday but before 3:30 PM EDT, use the previous Friday
    if current_date.weekday() == 4:  # Friday
        current_time = current_date.time()
        cutoff_time = datetime.strptime("15:30", "%H:%M").time()
        if current_time < cutoff_time:
            last_friday = last_friday - timedelta(weeks=1)
            print(f"It's Friday before 3:30 PM EDT, adjusting to previous Friday: {last_friday.strftime('%Y-%m-%d')}")
    
    # Get the previous Friday
    previous_friday = last_friday - timedelta(weeks=1)
    
    print("\nCalculated Report Dates:")
    print(f"Latest Report (Friday): {last_friday.strftime('%Y-%m-%d')}")
    print(f"Previous Report (Friday): {previous_friday.strftime('%Y-%m-%d')}")
    
    return last_friday, previous_friday

def validate_with_api(report_date):
    """
    Validate that a report exists for the given date by checking the API
    """
    try:
        MyAppToken = os.getenv('SODAPY_TOKEN')
        if not MyAppToken:
            print("Error: SODAPY_TOKEN not found")
            return False
            
        client = Socrata("publicreporting.cftc.gov", MyAppToken, timeout=30)
        
        # Format date for query
        date_str = report_date.strftime('%Y-%m-%d')
        
        # Try to get data for Bitcoin on this date
        result = client.get(
            "6dca-aqww",
            where=f"report_date_as_yyyy_mm_dd = '{date_str}' AND market_and_exchange_names = 'BITCOIN - CHICAGO MERCANTILE EXCHANGE'",
            limit=1
        )
        
        exists = len(result) > 0
        print(f"Report for {date_str}: {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            print(f"Data timestamp: {result[0].get('report_date_as_yyyy_mm_dd')}")
        
        return exists
        
    except Exception as e:
        print(f"Error checking API: {str(e)}")
        return False

def test_specific_dates():
    """Test the date logic with specific dates and times and validate with API"""
    test_cases = [
        # Current scenario
        datetime(2025, 4, 18, 14, 30),  # Friday April 18, 2025 2:30 PM EDT
        datetime(2025, 4, 18, 16, 30),  # Friday April 18, 2025 4:30 PM EDT
        
        # Previous days this week
        datetime(2025, 4, 17, 14, 30),  # Thursday
        datetime(2025, 4, 16, 14, 30),  # Wednesday
        
        # Edge cases
        datetime(2025, 4, 19, 14, 30),  # Saturday
        datetime(2025, 4, 20, 14, 30),  # Sunday
    ]
    
    for test_date in test_cases:
        print("\n" + "="*50)
        latest_friday, previous_friday = calculate_report_dates(test_date)
        
        print("\nValidating with API:")
        print("Latest report:")
        latest_exists = validate_with_api(latest_friday)
        print("\nPrevious report:")
        previous_exists = validate_with_api(previous_friday)
        
        if not latest_exists or not previous_exists:
            print("\nWARNING: One or both reports not found in API!")

if __name__ == "__main__":
    print("Testing Current Date:")
    latest_friday, previous_friday = calculate_report_dates()
    
    print("\nValidating current dates with API:")
    print("Latest report:")
    validate_with_api(latest_friday)
    print("\nPrevious report:")
    validate_with_api(previous_friday)
    
    print("\nTesting Specific Scenarios:")
    test_specific_dates() 