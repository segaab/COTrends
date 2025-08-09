import streamlit as st
import pandas as pd
import requests
from supabase import create_client, Client
import os
from datetime import datetime, timedelta

# Environment variables or replace here:
SUPABASE_URL = "https://dzddytphimhoxeccxqsw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR6ZGR5dHBoaW1ob3hlY2N4cXN3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTM2Njc5NCwiZXh"
FUNCTION_TRIGGER_URL = "https://dzddytphimhoxeccxqsw.supabase.co/functions/v1/treasury-rates-forecast"
FUNCTION_SECRET = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR6ZGR5dHBoaW1ob3hlY2N4cXN3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTM2Njc5NCwiZXhwIjoyMDY2OTQyNzk0fQ.ng0ST7-V-cDBD0Jc80_0DFWXylzE-gte2I9MCX7qb0Q"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Treasury Rates Forecast Dashboard", layout="wide")

st.title("Treasury Rates Forecast Dashboard")

@st.cache_data(ttl=300)
def get_latest_weekly_data():
    query = (
        supabase
        .from_("hourly_forecasts")
        .select("*")
        .order("forecast_generated_at", desc=True)
        .limit(1)
        .single()
    )
    latest_run = query.execute()
    if latest_run.error or not latest_run.data:
        st.error("Failed to fetch latest forecast run metadata")
        return None

    forecast_generated_at = latest_run.data["forecast_generated_at"]
    if not forecast_generated_at:
        st.warning("No forecast generated yet")
        return None

    data_response = (
        supabase
        .from_("hourly_forecasts")
        .select("*")
        .eq("forecast_generated_at", forecast_generated_at)
        .order("forecast_time", asc=True)
    ).execute()

    if data_response.error or not data_response.data:
        st.error("Failed to fetch forecast data")
        return None

    df = pd.DataFrame(data_response.data)
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    return df

def plot_forecast(df: pd.DataFrame):
    import altair as alt

    df_actual = df[df["is_forecast"] == False]
    df_forecast = df[df["is_forecast"] == True]

    base = alt.Chart(df).encode(x="forecast_time:T")

    line_actual = base.mark_line(color="blue").encode(
        y=alt.Y("combined_rate:Q", title="Combined Interest Rate"),
        tooltip=["forecast_time:T", "combined_rate:Q"],
    ).transform_filter(alt.datum.is_forecast == False)

    line_forecast = base.mark_line(color="orange", strokeDash=[5,5]).encode(
        y="combined_rate:Q",
        tooltip=["forecast_time:T", "combined_rate:Q"],
    ).transform_filter(alt.datum.is_forecast == True)

    band = alt.Chart(df_forecast).mark_area(color="orange", opacity=0.2).encode(
        x="forecast_time:T",
        y="forecast_lower_95:Q",
        y2="forecast_upper_95:Q",
    )

    chart = (line_actual + line_forecast + band).properties(
        width=900,
        height=400,
        title="Combined Rate: Actual (Blue) and Forecast (Orange)"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

def trigger_edge_function():
    headers = {
        "Authorization": f"Bearer {FUNCTION_SECRET}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(FUNCTION_TRIGGER_URL, headers=headers, json={})
        if response.status_code == 200:
            st.success("Edge function triggered successfully!")
        else:
            st.error(f"Failed to trigger function. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error triggering function: {e}")

def main():
    st.markdown(
        """
        This dashboard displays the latest combined interest rate forecast from Supabase.
        Use the buttons below to refresh data or trigger the forecast update job.
        """
    )

    df = get_latest_weekly_data()
    if df is not None and not df.empty:
        plot_forecast(df)
    else:
        st.warning("No data available to display.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Trigger Forecast Update"):
            with st.spinner("Triggering edge function..."):
                trigger_edge_function()
    with col2:
        if st.button("Refresh Data"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
