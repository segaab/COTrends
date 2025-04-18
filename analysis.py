import pandas as pd
from data_fetcher import filter_results

def aggregate_report_data(data, asset_name=None):
    """
    Aggregates data for a specific asset from both reports.
    """
    if not data:
        return pd.DataFrame()
        
    # Filter data for the specific asset if provided
    if asset_name:
        filtered_data = [item for item in data if item["market_and_exchange_names"] == asset_name]
    else:
        filtered_data = data
        
    if not filtered_data:
        return pd.DataFrame()
        
    # Convert to DataFrame
    df = pd.DataFrame(filtered_data)
    
    # Print column names for debugging
    print(f"Available columns for {asset_name}: {df.columns.tolist()}")
    
    # Define the expected column mappings
    column_mappings = {
        'noncomm_positions_long_all': ['noncomm_positions_long_all', 'noncomm_long_all', 'noncomm_long'],
        'noncomm_positions_short_all': ['noncomm_positions_short_all', 'noncomm_short_all', 'noncomm_short'],
        'comm_positions_long_all': ['comm_positions_long_all', 'comm_long_all', 'comm_long'],
        'comm_positions_short_all': ['comm_positions_short_all', 'comm_short_all', 'comm_short'],
        'nonrept_positions_long_all': ['nonrept_positions_long_all', 'nonrept_long_all', 'nonrept_long'],
        'nonrept_positions_short_all': ['nonrept_positions_short_all', 'nonrept_short_all', 'nonrept_short']
    }
    
    # Find the actual column names in the DataFrame
    actual_columns = {}
    for target_col, possible_names in column_mappings.items():
        for name in possible_names:
            if name in df.columns:
                actual_columns[target_col] = name
                break
    
    # If we couldn't find any of the required columns, return empty DataFrame
    if not actual_columns:
        print(f"Warning: No matching columns found for {asset_name}")
        return pd.DataFrame()
    
    # Convert string values to numeric where appropriate
    for target_col, actual_col in actual_columns.items():
        if actual_col in df.columns:
            df[target_col] = pd.to_numeric(df[actual_col], errors='coerce')
    
    # Ensure we have at least one report
    if len(df) < 1:
        print(f"Warning: No reports found for {asset_name}")
        return pd.DataFrame()
    
    return df

def analyze_change(df):
    """
    Analyzes position changes between reports for different trader groups.
    The formula calculates the change in net position ratios:
    ((current_long - current_short) ÷ total_current) - ((previous_long - previous_short) ÷ total_previous)
    """
    if df.empty or len(df) < 2:
        print("Warning: Not enough data for change analysis")
        return pd.DataFrame({
            'group': ['Non-Commercial', 'Commercial', 'Non-Reportable'],
            'change_in_net_pct': [0, 0, 0]
        })
    
    # Sort by date to ensure correct order
    df = df.sort_values('report_date_as_yyyy_mm_dd')
    
    # Get the latest and previous reports
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    def calculate_net_change(latest_long, latest_short, prev_long, prev_short):
        """Calculate net change using the ratio formula"""
        # Calculate total positions for each period
        latest_total = latest_long + latest_short
        prev_total = prev_long + prev_short
        
        # Calculate the ratios
        latest_ratio = (latest_long - latest_short) / latest_total if latest_total != 0 else 0
        prev_ratio = (prev_long - prev_short) / prev_total if prev_total != 0 else 0
        
        # Calculate the difference in ratios and convert to percentage
        return (latest_ratio - prev_ratio) * 100
    
    # Calculate changes for each trader group
    changes = {
        'Non-Commercial': calculate_net_change(
            latest['noncomm_positions_long_all'],
            latest['noncomm_positions_short_all'],
            previous['noncomm_positions_long_all'],
            previous['noncomm_positions_short_all']
        ),
        'Commercial': calculate_net_change(
            latest['comm_positions_long_all'],
            latest['comm_positions_short_all'],
            previous['comm_positions_long_all'],
            previous['comm_positions_short_all']
        ),
        'Non-Reportable': calculate_net_change(
            latest['nonrept_positions_long_all'],
            latest['nonrept_positions_short_all'],
            previous['nonrept_positions_long_all'],
            previous['nonrept_positions_short_all']
        )
    }
    
    return pd.DataFrame({
        'group': list(changes.keys()),
        'change_in_net_pct': list(changes.values())
    })

def analyze_positions(df):
    """
    Analyzes current positions for each trader group.
    Returns a DataFrame with the percentage of long and short positions.
    """
    if df.empty:
        print("Warning: No data available for position analysis")
        return pd.DataFrame()
        
    # Get the latest report
    latest = df.sort_values('report_date_as_yyyy_mm_dd').iloc[-1]
    
    def calculate_percentages(long_val, short_val):
        """Calculate percentage of long and short positions"""
        try:
            total = int(long_val) + int(short_val)
            if total > 0:
                return {
                    'Long (%)': (int(long_val) / total) * 100,
                    'Short (%)': (int(short_val) / total) * 100
                }
            return {'Long (%)': 0, 'Short (%)': 0}
        except (ValueError, TypeError):
            return {'Long (%)': 0, 'Short (%)': 0}
    
    # Calculate percentages for each trader group
    trader_groups = {
        'Non-Commercial': calculate_percentages(
            latest['noncomm_positions_long_all'],
            latest['noncomm_positions_short_all']
        ),
        'Commercial': calculate_percentages(
            latest['comm_positions_long_all'],
            latest['comm_positions_short_all']
        ),
        'Non-Reportable': calculate_percentages(
            latest['nonrept_positions_long_all'],
            latest['nonrept_positions_short_all']
        )
    }
    
    # Create DataFrame with proper structure for bar chart
    df_data = {
        'Long (%)': [trader_groups[group]['Long (%)'] for group in trader_groups],
        'Short (%)': [trader_groups[group]['Short (%)'] for group in trader_groups]
    }
    
    return pd.DataFrame(
        df_data,
        index=['Non-Commercial', 'Commercial', 'Non-Reportable']
    ) 