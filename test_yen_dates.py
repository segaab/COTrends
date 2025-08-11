# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import requests
from sodapy import Socrata
from yahooquery import Ticker
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple

# -----------------------
# CONFIG
# -----------------------
# Hardcoded Socrata App Token as requested
APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"

TIMEOUT = 60  # network timeout seconds
LOOKBACK_DAYS_SPECTRUM = 30  # ~6 weeks of trading days
RVOL_PERCENTILE = 0.78
UPTREND_PCT = 4.0   # +4% threshold (percent)
DOWNTREND_PCT = -4.0  # -4% threshold (percent)

WEIGHT_PV_RVOL = 0.40
WEIGHT_COT = 0.35
WEIGHT_OI = 0.25
COT_SHORT_TERM_WT = 0.40
COT_LONG_TERM_WT = 0.60

# -----------------------
# ASSET MAPPING
# display_name -> (COT_name, yahoo_ticker)
# -----------------------
ASSET_MAP = {
    # base metals
    "Gold": ("GOLD - COMMODITY EXCHANGE INC.", "GC=F"),
    "Silver": ("SILVER - COMMODITY EXCHANGE INC.", "SI=F"),

    # currency futures (display uses common name)
    "AUD/USD": ("AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "AUDUSD=X"),
    "GBP/USD": ("BRITISH POUND - CHICAGO MERCANTILE EXCHANGE", "GBPUSD=X"),
    "USD/CAD": ("CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE", "USDCAD=X"),
    "EUR/USD": ("EURO FX - CHICAGO MERCANTILE EXCHANGE", "EURUSD=X"),
    "EUR/GBP": ("EURO FX/BRITISH POUND XRATE - CHICAGO MERCANTILE EXCHANGE", "EURGBP=X"),
    "JPY/USD": ("JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE", "JPY=X"),
    "CHF/USD": ("SWISS FRANC - CHICAGO MERCANTILE EXCHANGE", "CHF=X"),

    # crypto proxies
    "Bitcoin": ("BITCOIN - CHICAGO MERCANTILE EXCHANGE", "BTC-USD"),
    "Micro Bitcoin": ("MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE", "BTC-USD"),
    "Micro Ether": ("MICRO ETHER - CHICAGO MERCANTILE EXCHANGE", "ETH-USD"),

    # equity index futures / proxies
    "E-Mini S&P Financial (ES)": ("E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE", "ES=F"),
    "DJ Real Estate (ETF proxy)": ("DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE", "DJR"),

    # energy & other metals
    "WTI Crude": ("WTI CRUDE OIL FINANCIAL - NEW YORK MERCANTILE EXCHANGE", "CL=F"),
    "Platinum": ("PLATINUM - NEW YORK MERCANTILE EXCHANGE", "PL=F"),
    "Palladium": ("PALLADIUM - NEW YORK MERCANTILE EXCHANGE", "PA=F"),
    "Copper": ("COPPER - COMMODITY EXCHANGE INC.", "HG=F"),
}

# -----------------------
# Logging helper (writes to UI and console)
# -----------------------
def app_log(msg: str, level: str = "info"):
    """Show message in Streamlit UI and print to console for logs"""
    if level == "info":
        st.info(msg)
    elif level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.write(msg)
    print(f"[{level.upper()}] {msg}")

# -----------------------
# Socrata client & COT fetch
# -----------------------
@st.cache_resource(show_spinner=False)
def get_socrata_client() -> Optional[Socrata]:
    try:
        client = Socrata("publicreporting.cftc.gov", APP_TOKEN, timeout=TIMEOUT)
        app_log("Socrata client initialized.", "info")
        return client
    except Exception as e:
        app_log(f"Socrata client init failed: {e}", "error")
        return None

@st.cache_data(ttl=60*60*6, show_spinner=True)
def fetch_cot_data_by_market(cot_market_name: str) -> pd.DataFrame:
    client = get_socrata_client()
    if client is None:
        return pd.DataFrame()
    # correct column name is 'market_and_exchange_names'
    where_clause = f"market_and_exchange_names='{cot_market_name}'"
    try:
        results = client.get("6dca-aqww", where=where_clause, order="report_date DESC", limit=1000)
    except requests.exceptions.Timeout:
        app_log(f"COT fetch timeout for {cot_market_name}", "error")
        return pd.DataFrame()
    except requests.exceptions.HTTPError as e:
        app_log(f"COT HTTP error for {cot_market_name}: {e}", "error")
        return pd.DataFrame()
    except Exception as e:
        app_log(f"COT fetch unexpected error for {cot_market_name}: {e}", "error")
        return pd.DataFrame()

    if not results:
        app_log(f"No COT rows returned for {cot_market_name}", "warning")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(results)
    df["report_date"] = pd.to_datetime(df.get("report_date", pd.Series()), errors="coerce")
    # convert expected numeric columns if present
    expected_cols = [
        "noncomm_positions_long_all", "noncomm_positions_short_all", "noncomm_positions_spread_all",
        "comm_positions_long_all", "comm_positions_short_all",
        "tot_rept_positions_long_all", "tot_rept_positions_short",
        "nonrept_positions_long_all", "nonrept_positions_short_all",
        "open_interest_all"
    ]
    for col in expected_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    app_log(f"Fetched COT for {cot_market_name}: {len(df)} rows", "success")
    return df.sort_values("report_date")

# -----------------------
# Yahoo history fetch
# -----------------------
@st.cache_data(ttl=60*60*2, show_spinner=True)
def fetch_yahoo_history(ticker: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    try:
        t = Ticker(ticker, asynchronous=False, timeout=TIMEOUT)
        hist = t.history(start=start_iso, end=end_iso)
    except Exception as e:
        app_log(f"Yahoo fetch error for {ticker}: {e}", "error")
        return pd.DataFrame()

    if hist is None or hist.empty:
        app_log(f"No yahoo data for {ticker}", "warning")
        return pd.DataFrame()

    hist = hist.reset_index()
    hist.columns = [c.lower() for c in hist.columns]

    # ensure date column exists and is tz-naive
    if "date" not in hist.columns:
        app_log(f"Yahoo data for {ticker} missing 'date' column", "error")
        return pd.DataFrame()

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    # remove tz info if present
    try:
        if pd.api.types.is_datetime64tz_dtype(hist["date"]):
            hist["date"] = hist["date"].dt.tz_convert(None).dt.tz_localize(None)
    except Exception:
        # fallback: force naive by astype
        try:
            hist["date"] = hist["date"].dt.tz_localize(None)
        except Exception:
            pass

    hist["date"] = hist["date"].dt.date
    # ensure required columns
    if "close" not in hist.columns:
        hist["close"] = np.nan
    if "volume" not in hist.columns:
        hist["volume"] = np.nan
    # drop rows without close
    hist = hist.dropna(subset=["close"]).reset_index(drop=True)
    app_log(f"Yahoo {ticker} rows: {len(hist)}", "success")
    return hist

# -----------------------
# Spectrum detection & scoring
# -----------------------
def calculate_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["avg_vol_20"] = df["volume"].rolling(window=lookback, min_periods=1).mean()
    df["rvol"] = df["volume"] / df["avg_vol_20"]
    return df

def detect_spectrum_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df_tail = df.sort_values("date").tail(LOOKBACK_DAYS_SPECTRUM).reset_index(drop=True)
    if df_tail.empty or len(df_tail) < 3:
        return pd.DataFrame()
    df_tail = calculate_rvol(df_tail)
    rvol_thresh = np.percentile(df_tail["rvol"].dropna(), RVOL_PERCENTILE * 100)
    pts = []
    for i in range(len(df_tail)):
        if pd.isna(df_tail.loc[i, "rvol"]):
            continue
        if df_tail.loc[i, "rvol"] >= rvol_thresh:
            start = max(i - 4, 0)
            end = min(i + 4, len(df_tail) - 1)
            ps = df_tail.loc[start, "close"]
            pe = df_tail.loc[end, "close"]
            vs = df_tail.loc[start, "volume"] if "volume" in df_tail.columns else 0
            ve = df_tail.loc[end, "volume"] if "volume" in df_tail.columns else 0
            if ps == 0:
                continue
            price_change_pct = (pe - ps) / ps * 100
            vol_change_pct = ((ve - vs) / vs * 100) if vs != 0 else 0
            if price_change_pct >= 0 and vol_change_pct > 0:
                kind = "Accumulation"
            elif price_change_pct < 0 and vol_change_pct > 0:
                kind = "Distribution"
            else:
                continue
            next_idx = min(i + 1, len(df_tail) - 1)
            next_close = df_tail.loc[next_idx, "close"]
            event_close = df_tail.loc[i, "close"]
            next_day_return = (next_close - event_close) / event_close * 100 if event_close != 0 else 0
            pts.append({
                "index": i,
                "date": df_tail.loc[i, "date"],
                "type": kind,
                "price_change_pct": price_change_pct,
                "vol_change_pct": vol_change_pct,
                "rvol": df_tail.loc[i, "rvol"],
                "next_day_return_pct": next_day_return
            })
    return pd.DataFrame(pts)

def assign_weights_and_score(points: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    if points.empty:
        return 3.0, points
    weights = []
    for _, r in points.iterrows():
        pchg = r["price_change_pct"]
        nd = r["next_day_return_pct"]
        if r["type"] == "Accumulation":
            if 0 <= pchg < 1:
                pw = 1
            elif 1 <= pchg < 2:
                pw = 2
            else:
                pw = 3
            if 0 <= nd < 1:
                nw = 1
            elif 1 <= nd < 2:
                nw = 2
            else:
                nw = 3
            total = pw + nw
        else:
            if -1 <= pchg < 0:
                pw = -1
            elif -2 <= pchg < -1:
                pw = -2
            else:
                pw = -3
            if -1 <= nd < 0:
                nw = -1
            elif -2 <= nd < -1:
                nw = -2
            else:
                nw = -3
            total = pw + nw
        weights.append(total)
    points = points.copy()
    points["weight_raw"] = weights
    clipped = np.clip(points["weight_raw"], -6, 6)
    points["weight_norm_0_5"] = (clipped + 6) / 12 * 5.0
    pv_score = float(points["weight_norm_0_5"].mean()) if not points.empty else 3.0
    pv_score = max(0.0, min(5.0, pv_score))
    return pv_score, points

def trending_score_from_prices(df: pd.DataFrame) -> float:
    tail = df.sort_values("date").tail(LOOKBACK_DAYS_SPECTRUM)
    if tail.empty or len(tail) < 2:
        return 3.0
    pct_change = (tail["close"].iloc[-1] - tail["close"].iloc[0]) / tail["close"].iloc[0] * 100
    if pct_change > UPTREND_PCT:
        return 5.0
    if pct_change < DOWNTREND_PCT:
        return 1.0
    return 3.0

# -----------------------
# COT scoring
# -----------------------
def compute_cot_scores(cot_df: pd.DataFrame) -> Tuple[float, float]:
    if cot_df.empty:
        return 3.0, 3.0
    if not all(c in cot_df.columns for c in ["noncomm_positions_long_all", "noncomm_positions_short_all",
                                             "comm_positions_long_all", "comm_positions_short_all"]):
        return 3.0, 3.0
    cot = cot_df.sort_values("report_date").copy()
    cot["noncomm_net"] = cot["noncomm_positions_long_all"] - cot["noncomm_positions_short_all"]
    cot["comm_net"] = cot["comm_positions_long_all"] - cot["comm_positions_short_all"]
    # short-term
    if len(cot) < 2:
        short_score = 3.0
    else:
        latest = cot.iloc[-1]
        prev = cot.iloc[-2]
        delta_noncomm = latest["noncomm_net"] - prev["noncomm_net"]
        delta_comm = latest["comm_net"] - prev["comm_net"]
        noncomm_score = 5.0 if delta_noncomm > 0 else (1.0 if delta_noncomm < 0 else 3.0)
        comm_score = 5.0 if delta_comm < 0 else (1.0 if delta_comm > 0 else 3.0)
        short_score = (noncomm_score + comm_score) / 2.0
    # long-term 12-week MA trend
    if len(cot) < 12:
        long_score = 3.0
    else:
        cot["noncomm_net_ma12"] = cot["noncomm_net"].rolling(window=12, min_periods=1).mean()
        cot["comm_net_ma12"] = cot["comm_net"].rolling(window=12, min_periods=1).mean()
        ma_noncomm = cot["noncomm_net_ma12"].dropna().values
        ma_comm = cot["comm_net_ma12"].dropna().values
        if len(ma_noncomm) < 2 or len(ma_comm) < 2:
            long_score = 3.0
        else:
            noncomm_trend = ma_noncomm[-1] - ma_noncomm[0]
            comm_trend = ma_comm[-1] - ma_comm[0]
            noncomm_score_lt = 5.0 if noncomm_trend > 0 else (1.0 if noncomm_trend < 0 else 3.0)
            comm_score_lt = 5.0 if comm_trend < 0 else (1.0 if comm_trend > 0 else 3.0)
            long_score = (noncomm_score_lt + comm_score_lt) / 2.0
    return short_score, long_score

# -----------------------
# Orchestrator per asset
# -----------------------
def process_asset(display: str, cot_name: str, ticker: str, start_iso: str, end_iso: str) -> Dict[str, Any]:
    result = {"display": display, "ok": False, "error": None}
    try:
        cot_df = fetch_cot_data_by_market(cot_name)
        price_df = fetch_yahoo_history(ticker, start_iso, end_iso)
        if cot_df.empty:
            app_log(f"[{display}] COT data empty.", "warning")
        if price_df.empty:
            app_log(f"[{display}] Price data empty.", "warning")
        if cot_df.empty or price_df.empty:
            result["error"] = "missing_data"
            return result
        points = detect_spectrum_points(price_df)
        pv_score, points = assign_weights_and_score(points)
        trend_score = trending_score_from_prices(price_df)
        cot_short, cot_long = compute_cot_scores(cot_df)
        cot_combined = cot_short * COT_SHORT_TERM_WT + cot_long * COT_LONG_TERM_WT
        oi_score = 3.0
        if "openinterest" in price_df.columns:
            oi_vals = price_df["openinterest"].dropna()
            if len(oi_vals) >= 20:
                slope = np.polyfit(range(len(oi_vals)), oi_vals, 1)[0]
                oi_score = 5.0 if slope > 0 else (1.0 if slope < 0 else 3.0)
        health_raw = pv_score * WEIGHT_PV_RVOL + cot_combined * WEIGHT_COT + oi_score * WEIGHT_OI
        health = int(round(max(0, min(5, health_raw))))
        result.update({
            "ok": True,
            "cot_df": cot_df,
            "price_df": price_df,
            "points_df": points.reset_index(drop=True),
            "pv_score": pv_score,
            "trend_score": trend_score,
            "cot_short": cot_short,
            "cot_long": cot_long,
            "cot_combined": cot_combined,
            "oi_score": oi_score,
            "health": health
        })
        app_log(f"[{display}] done: health={health}, pv={pv_score:.2f}, cot={cot_combined:.2f}, oi={oi_score:.2f}", "success")
        return result
    except Exception as e:
        app_log(f"[{display}] processing failed: {e}", "error")
        result["error"] = str(e)
        return result

# -----------------------
# Plot helpers
# -----------------------
def plot_event_with_next_day(price_df: pd.DataFrame, event_date, window=1):
    price_df = price_df.sort_values("date").reset_index(drop=True)
    if isinstance(event_date, str):
        event_date = datetime.datetime.fromisoformat(event_date).date()
    idxs = price_df.index[price_df["date"] == event_date].tolist()
    if not idxs:
        st.warning("Event date not in price data.")
        return
    idx = idxs[0]
    start = max(idx - window, 0)
    end = min(idx + window + 1, len(price_df))
    sub = price_df.iloc[start:end]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"], mode="lines+markers", name="Close"))
    fig.add_vrect(x0=event_date, x1=event_date, fillcolor="LightGreen", opacity=0.25, layer="below", line_width=0)
    if idx + 1 < len(price_df):
        next_day = price_df.loc[idx + 1, "date"]
        fig.add_vrect(x0=next_day, x1=next_day, fillcolor="LightBlue", opacity=0.25, layer="below", line_width=0)
    fig.update_layout(title="Event Day and Next Day Price", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def plot_overview_price_with_points(price_df: pd.DataFrame, points_df: pd.DataFrame):
    df = price_df.sort_values("date").reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close"))
    if not points_df.empty:
        accum = points_df[points_df["type"] == "Accumulation"]
        distrib = points_df[points_df["type"] == "Distribution"]
        if not accum.empty:
            yvals = [df.loc[df["date"] == d, "close"].values[0] if not df.loc[df["date"] == d, "close"].empty else None for d in accum["date"]]
            fig.add_trace(go.Scatter(x=accum["date"], y=yvals, mode="markers", name="Accumulation", marker=dict(color="green", symbol="triangle-up", size=10)))
        if not distrib.empty:
            yvals = [df.loc[df["date"] == d, "close"].values[0] if not df.loc[df["date"] == d, "close"].empty else None for d in distrib["date"]]
            fig.add_trace(go.Scatter(x=distrib["date"], y=yvals, mode="markers", name="Distribution", marker=dict(color="red", symbol="triangle-down", size=10)))
    fig.update_layout(title="Price with Detected Points", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Markets Health Gauge", layout="wide")
st.title("Markets Health Gauge — multi-asset (synchronized fetch)")

st.markdown("Assets: " + ", ".join(ASSET_MAP.keys()))

col1, col2 = st.columns(2)
with col1:
    default_end = datetime.date.today()
    end_date = st.date_input("End Date", value=default_end)
with col2:
    default_start = default_end - datetime.timedelta(days=365)
    start_date = st.date_input("Start Date", value=default_start)

if start_date >= end_date:
    st.error("Start date must be before end date")
    st.stop()

# parallelism setting + refresh
workers = st.sidebar.slider("Parallel workers", 2, 16, 8)
if st.sidebar.button("Refresh (clear caches)"):
    st.cache_data.clear()
    st.experimental_rerun()

start_iso = start_date.isoformat()
end_iso = end_date.isoformat()

status_ph = st.empty()
logs_ph = st.empty()

assets = [(disp, *ASSET_MAP[disp]) for disp in ASSET_MAP.keys()]

results = {}
start_time = time.time()
status_ph.info(f"Starting processing {len(assets)} assets with {workers} workers...")

with ThreadPoolExecutor(max_workers=workers) as ex:
    futures = {ex.submit(process_asset, disp, cotn, tkr, start_iso, end_iso): disp for disp, cotn, tkr in assets}
    completed = 0
    logs = []
    for f in as_completed(futures):
        disp = futures[f]
        try:
            res = f.result()
        except Exception as e:
            res = {"display": disp, "ok": False, "error": str(e)}
            app_log(f"{disp} exception: {e}", "error")
        results[disp] = res
        completed += 1
        status_ph.info(f"Completed {completed}/{len(assets)}")
        logs.append(f"{disp}: {'OK' if res.get('ok') else 'ERR'} - {res.get('error')}")
        logs_ph.text("\n".join(logs))

elapsed = time.time() - start_time
status_ph.success(f"All done in {elapsed:.1f}s")

# asset inspector
selected = st.sidebar.selectbox("Inspect asset", list(ASSET_MAP.keys()), index=0)
res = results.get(selected)
if not res:
    st.warning("No result for selected asset")
else:
    st.header(selected)
    if not res.get("ok"):
        st.error(f"Failed: {res.get('error')}")
    else:
        st.write(f"Ticker: {res['ticker']}")
        st.metric("Health (0-5)", res["health"])
        st.write("- PV score:", round(res["pv_score"], 2))
        st.write("- Trend score:", round(res["trend_score"], 2))
        st.write(f"- COT short: {res['cot_short']:.2f}, COT long: {res['cot_long']:.2f}, combine
