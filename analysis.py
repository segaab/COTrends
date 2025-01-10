import pandas as pd
from data_fetcher import filter_results

def aggregate_report_data(data, asset_name=None):
    """
    Aggregates and nests rows of data from both reports under corresponding market names.
    """
    filtered_market_names = filter_results(data, asset_name)

    aggregated_data = []
    for item in data:
        if item['market_and_exchange_names'] in filtered_market_names:
            aggregated_data.append({
                'market_and_exchange_names': item['market_and_exchange_names'],
                'latest_report': item['latest_report'],
                'previous_report': item['previous_report']
            })

    return pd.DataFrame(aggregated_data)

def analyze_change(aggregated_data):
    """
    Analyzes position changes and percentages for different trader groups.
    """
    analysis_results = []

    for _, row in aggregated_data.iterrows():
        market_name = row['market_and_exchange_names']
        latest_report = row['latest_report']
        previous_report = row['previous_report']

        if not latest_report or not previous_report:
            continue

        for group in ['noncomm_positions', 'comm_positions', 'nonrept_positions']:
            # Calculate positions and percentages
            latest_long = float(latest_report.get(f'{group}_long_all', 0))
            latest_short = float(latest_report.get(f'{group}_short_all', 0))
            total_latest = latest_long + latest_short

            previous_long = float(previous_report.get(f'{group}_long_all', 0))
            previous_short = float(previous_report.get(f'{group}_short_all', 0))
            total_previous = previous_long + previous_short

            # Calculate percentages and changes
            latest_long_pct = (latest_long / total_latest) * 100 if total_latest > 0 else 0
            latest_short_pct = (latest_short / total_latest) * 100 if total_latest > 0 else 0
            previous_long_pct = (previous_long / total_previous) * 100 if total_previous > 0 else 0
            previous_short_pct = (previous_short / total_previous) * 100 if total_previous > 0 else 0

            latest_net_pct = latest_long_pct - latest_short_pct
            previous_net_pct = previous_long_pct - previous_short_pct
            change_in_net_pct = latest_net_pct - previous_net_pct

            analysis_results.append({
                'market_and_exchange_names': market_name,
                'group': group,
                'latest_long_pct': latest_long_pct,
                'latest_short_pct': latest_short_pct,
                'previous_long_pct': previous_long_pct,
                'previous_short_pct': previous_short_pct,
                'latest_net_pct': latest_net_pct,
                'previous_net_pct': previous_net_pct,
                'change_in_net_pct': change_in_net_pct
            })

    df = pd.DataFrame(analysis_results)
    df_analysis = df.copy()
    df_analysis.loc[df['group']=="noncomm_positions",'group'] = "Non-Commercial"
    df_analysis.loc[df['group']=="comm_positions",'group'] = "Commercial"
    df_analysis.loc[df['group']=="nonrept_positions",'group'] = "Retail"

    return df_analysis

def analyze_positions(cot_data):
    """
    Analyzes positions by trader group and returns normalized position data.
    """
    non_commercial_long = cot_data.loc[(cot_data['group'] == 'Non-Commercial'), 'latest_long_pct'].sum()
    non_commercial_short = cot_data.loc[(cot_data['group'] == 'Non-Commercial'), 'latest_short_pct'].sum()
    
    commercial_long = cot_data.loc[(cot_data['group'] == 'Commercial'), 'latest_long_pct'].sum()
    commercial_short = cot_data.loc[(cot_data['group'] == 'Commercial'), 'latest_short_pct'].sum()
    
    non_reportable_long = cot_data.loc[(cot_data['group'] == 'Retail'), 'latest_long_pct'].sum()
    non_reportable_short = cot_data.loc[(cot_data['group'] == 'Retail'), 'latest_short_pct'].sum()

    data = {
        'Trader Group': ['Non-Commercial','Commercial', 'Retail'],
        'Long': [non_commercial_long, commercial_long, non_reportable_long],
        'Short': [non_commercial_short, commercial_short, non_reportable_short]
    }

    position_df = pd.DataFrame(data)
    position_df['Total'] = position_df['Long'] + position_df['Short']
    position_df['Long (%)'] = (position_df['Long'] / position_df['Total']) * 100
    position_df['Short (%)'] = (position_df['Short'] / position_df['Total']) * 100
    
    return position_df[['Trader Group', 'Long (%)', 'Short (%)']].set_index('Trader Group') 