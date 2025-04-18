import os
from datetime import datetime, timedelta
from sodapy import Socrata
import pandas as pd
from zoneinfo import ZoneInfo

def init_client():
    """Initialize Socrata client"""
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

def get_all_available_dates(client):
    """Get all available report dates for Japanese Yen"""
    try:
        # Query for all Japanese Yen reports, selecting only the date
        results = client.get(
            "6dca-aqww",
            where="market_and_exchange_names = 'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE'",
            select="report_date_as_yyyy_mm_dd",
            limit=5000  # Get a large number to ensure we get all dates
        )
        
        # Convert to datetime objects and sort
        dates = [datetime.strptime(r['report_date_as_yyyy_mm_dd'][:10], '%Y-%m-%d') 
                for r in results]
        dates = sorted(set(dates), reverse=True)  # Remove duplicates and sort descending
        
        print("\nAPI Date Information:")
        print("-" * 30)
        print(f"Total reports available: {len(dates)}")
        print(f"Date range: {dates[-1].strftime('%Y-%m-%d')} to {dates[0].strftime('%Y-%m-%d')}")
        
        return dates
    except Exception as e:
        print(f"Error fetching dates: {str(e)}")
        return []

def get_expected_report_dates():
    """Calculate expected report dates based on current time"""
    # Get current time in ET
    now = datetime.now(ZoneInfo("America/New_York"))
    print("\nDate Calculation Details:")
    print("-" * 30)
    print(f"Current time (ET): {now}")
    print(f"Current weekday: {now.strftime('%A')}")
    
    # Find the most recent Friday
    days_since_friday = (now.weekday() - 4) % 7
    last_friday = now - timedelta(days=days_since_friday)
    print(f"Days since Friday: {days_since_friday}")
    print(f"Most recent Friday: {last_friday.strftime('%Y-%m-%d %A')}")
    
    # If it's Friday but before 3:30 PM ET, use the previous Friday
    if now.weekday() == 4 and now.time() < datetime.strptime("15:30", "%H:%M").time():
        print("It's Friday before 3:30 PM ET, adjusting to previous Friday")
        last_friday = last_friday - timedelta(weeks=1)
        print(f"Adjusted Friday: {last_friday.strftime('%Y-%m-%d %A')}")
    
    # If we're on a weekend, go back to the previous Friday
    if now.weekday() >= 5:
        print("It's weekend, adjusting to previous Friday")
        last_friday = last_friday - timedelta(weeks=1)
        print(f"Adjusted Friday: {last_friday.strftime('%Y-%m-%d %A')}")
    
    # The report is for the previous Tuesday
    latest_tuesday = last_friday - timedelta(days=3)
    previous_tuesday = latest_tuesday - timedelta(weeks=1)
    
    print(f"\nCalculated Report Dates:")
    print("-" * 30)
    print(f"Latest Tuesday: {latest_tuesday.strftime('%Y-%m-%d %A')}")
    print(f"Previous Tuesday: {previous_tuesday.strftime('%Y-%m-%d %A')}")
    
    return latest_tuesday.date(), previous_tuesday.date()

def get_yen_data(client, report_date):
    """Get Japanese Yen data for a specific date"""
    try:
        result = client.get(
            "6dca-aqww",
            where=f"report_date_as_yyyy_mm_dd = '{report_date}' AND market_and_exchange_names = 'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE'",
            limit=1
        )
        return result[0] if result else None
    except Exception as e:
        print(f"Error fetching data for {report_date}: {str(e)}")
        return None

def main():
    print("Starting Japanese Yen Report Date Test...")
    print("=" * 50)
    
    # Initialize client
    client = init_client()
    if not client:
        return
    
    # Get actual available dates
    print("\nFetching all available report dates...")
    available_dates = get_all_available_dates(client)
    if not available_dates:
        print("Failed to fetch available dates")
        return
    
    print("\nMost recent available reports:")
    print("-" * 30)
    for i, date in enumerate(available_dates[:5]):
        print(f"{i+1}. {date.strftime('%Y-%m-%d %A')}")
    
    # Calculate expected dates
    expected_latest, expected_previous = get_expected_report_dates()
    
    # Compare expected vs actual
    latest_available = available_dates[0].date()
    previous_available = available_dates[1].date()
    
    print("\nDate Validation:")
    print("-" * 30)
    print(f"Latest report:")
    print(f"  Expected: {expected_latest}")
    print(f"  Actual:   {latest_available}")
    print(f"Previous report:")
    print(f"  Expected: {expected_previous}")
    print(f"  Actual:   {previous_available}")
    
    if expected_latest != latest_available or expected_previous != previous_available:
        print("\nWARNING: Mismatch between expected and actual dates!")
        print("This suggests our date calculation formula needs to be updated.")
        
        # Calculate the delay
        delay = (expected_latest - latest_available).days
        print(f"\nDelay Analysis:")
        print("-" * 30)
        print(f"Days between expected and actual: {delay}")
        
        # Analyze the pattern
        day_differences = []
        for i in range(len(available_dates)-1):
            diff = (available_dates[i] - available_dates[i+1]).days
            day_differences.append(diff)
        
        print("\nReport Pattern Analysis:")
        print("-" * 30)
        print("Days between consecutive reports:")
        for i, diff in enumerate(day_differences[:5]):
            print(f"  {available_dates[i].strftime('%Y-%m-%d')} to {available_dates[i+1].strftime('%Y-%m-%d')}: {diff} days")
    else:
        print("\nSuccess: Expected dates match actual dates!")
    
    # Get the actual data for the latest report
    print("\nLatest Report Data:")
    print("-" * 30)
    latest_data = get_yen_data(client, latest_available)
    if latest_data:
        print(f"Report Date: {latest_data['report_date_as_yyyy_mm_dd']}")
        print("\nTrader Positions:")
        print("Non-Commercial:")
        print(f"  Long:  {int(latest_data['noncomm_positions_long_all']):,d}")
        print(f"  Short: {int(latest_data['noncomm_positions_short_all']):,d}")
        print("Commercial:")
        print(f"  Long:  {int(latest_data['comm_positions_long_all']):,d}")
        print(f"  Short: {int(latest_data['comm_positions_short_all']):,d}")
        print("Non-Reportable:")
        print(f"  Long:  {int(latest_data['nonrept_positions_long_all']):,d}")
        print(f"  Short: {int(latest_data['nonrept_positions_short_all']):,d}")

if __name__ == "__main__":
    main() 