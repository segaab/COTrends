import pandas as pd
import os
from sodapy import Socrata
from datetime import datetime, timedelta

# Required fields
required_fields = [
    "market_and_exchange_names",
    "report_date_as_yyyy_mm_dd",
    "noncomm_positions_long_all",
    "noncomm_positions_short_all"
]

# Load credentials from environment variables
MyAppToken = os.getenv("WSCAAVL_APP_TOKEN")

# Initialize Socrata client (no username/password)
client = Socrata(
    "publicreporting.cftc.gov",
    MyAppToken
)

def get_last_two_reports(client, fields):
    """Fetch the last two COT reports from Socrata API."""
    edt_now = datetime.utcnow() - timedelta(hours=4)

    # Find last Friday
    last_friday = edt_now - timedelta(days=(edt_now.weekday() - 4) % 7)

    # If it's Friday and before 3:30 PM ET, use the previous Friday
    report_time = last_friday.replace(hour=15, minute=30, second=0)
    if edt_now.weekday() == 4 and edt_now < report_time:
        last_friday -= timedelta(weeks=1)

    # Latest and previous Tuesday
    latest_tuesday = last_friday - timedelta(days=3)
    previous_tuesday = latest_tuesday - timedelta(weeks=1)

    latest_str = latest_tuesday.strftime('%Y-%m-%d')
    previous_str = previous_tuesday.strftime('%Y-%m-%d')

    # Fetch reports
    latest_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{latest_str}'", select=",".join(fields))
    previous_result = client.get("6dca-aqww", where=f"report_date_as_yyyy_mm_dd = '{previous_str}'", select=",".join(fields))

    # Convert to DataFrames
    latest_df = pd.DataFrame.from_records(latest_result) if latest_result else None
    previous_df = pd.DataFrame.from_records(previous_result) if previous_result else None

    return latest_df, previous_df

# Example usage
latest_df, previous_df = get_last_two_reports(client, required_fields)

print("Latest Report Date:", latest_df["report_date_as_yyyy_mm_dd"].iloc[0] if latest_df is not None else "No Data")
print("Previous Report Date:", previous_df["report_date_as_yyyy_mm_dd"].iloc[0] if previous_df is not None else "No Data")
