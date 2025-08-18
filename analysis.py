# streamlit_fetch_exchanges_logging.py
import streamlit as st
import pandas as pd
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time

# Exchange CSV URLs (HTTPS)
EXCHANGE_FILES = {
    "NASDAQ": "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "NYSE/AMEX": "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
}

st.title("Exchange Ticker List Preview with Logs")

# Thread-safe log queue
log_queue = queue.Queue()

def fetch_exchange(name, url):
    try:
        log_queue.put(f"Starting download for {name}...")
        resp = requests.get(url, timeout=60)  # increase timeout
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), sep='|')
        log_queue.put(f"Successfully fetched {len(df)} tickers from {name}")
        return name, df, None
    except Exception as e:
        log_queue.put(f"Error fetching {name}: {e}")
        return name, None, str(e)

selected_exchanges = st.multiselect(
    "Select Exchanges to Fetch",
    list(EXCHANGE_FILES.keys()),
    default=list(EXCHANGE_FILES.keys())
)

if st.button("Fetch Tickers"):
    all_dfs = {}
    log_area = st.empty()
    progress_bar = st.progress(0)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_exchange, name, EXCHANGE_FILES[name]) for name in selected_exchanges]
        total = len(futures)
        completed = 0

        while completed < total:
            # Update logs
            logs = []
            while not log_queue.empty():
                logs.append(log_queue.get())
            if logs:
                log_area.text("\n".join(logs[-20:]))  # show last 20 logs

            # Update progress
            completed = sum(1 for f in futures if f.done())
            progress_bar.progress(completed / total)
            time.sleep(0.5)

        # Collect final results
        for future in as_completed(futures):
            name, df, error = future.result()
            if df is not None:
                all_dfs[name] = df

    # Show final preview of each exchange
    for name, df in all_dfs.items():
        st.write(f"Preview of {name}:")
        st.dataframe(df.head(20))
        st.write(f"Columns: {df.columns.tolist()}")
        st.write(f"Total rows: {len(df)}")
