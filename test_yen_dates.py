# fixed_cotrends_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import os
import logging
import threading
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sodapy import Socrata
from yahooquery import Ticker
from huggingface_hub import InferenceClient
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Setup ---
SODAPY_APP_TOKEN = "PP3ezxaUTiGforvvbBUGzwRx7"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=60)

# Initialize HF client once, expects HF_TOKEN in environment
hf_client = None
try:
    hf_client = InferenceClient(provider="cerebras", api_key=os.getenv("HF_TOKEN"))
except Exception as e:
    logger.warning(f"HF client not initialized: {e}")

# --- Asset mapping (COT market names -> Yahoo futures tickers) ---
assets = {
    "GOLD - COMMODITY EXCHANGE INC.": "GC=F",
    "SILVER - COMMODITY EXCHANGE INC.": "SI=F",
    "PLATINUM - NEW YORK MERCANTILE EXCHANGE": "PL=F",
    "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": "PA=F",
    "WTI CRUDE OIL FINANCIAL - NEW YORK MERCANTILE EXCHANGE": "CL=F",
    "COPPER - COMMODITY EXCHANGE INC.": "HG=F",
    "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6A=F",
    "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": "6B=F",
    "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": "6C=F",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "6E=F",
    "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": "6J=F",
    "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": "6S=F",
    "BITCOIN - CHICAGO MERCANTILE EXCHANGE": "BTC=F",
    "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE": "MBT=F",
    "MICRO ETHER - CHICAGO MERCANTILE EXCHANGE": "ETH=F",
    "E-MINI S&P FINANCIAL INDEX - CHICAGO MERCANTILE EXCHANGE": "ES=F",
    "DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE": "DJR",
    "NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": "NG=F",
    "CORN - CHICAGO BOARD OF TRADE": "ZC=F",
    "SOYBEANS - CHICAGO BOARD OF TRADE": "ZS=F",
}

# Make alias for legacy code
ASSET_MAPPING = assets

# --- Fetch COT Data ---
def fetch_cot_data(market_name: str, max_attempts: int = 3) -> pd.DataFrame:
    logger.info(f"Fetching COT data for {market_name}")
    where_clause = f'market_and_exchange_names="{market_name}"'
    attempt = 0
    while attempt < max_attempts:
        try:
            results = client.get(
                "6dca-aqww",
                where=where_clause,
                order="report_date_as_yyyy_mm_dd DESC",
                limit=1500,
            )
            if results:
                df = pd.DataFrame.from_records(results)
                df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
                # Numeric conversions with coercion
                def to_num(col):
                    if col in df.columns:
                        return pd.to_numeric(df[col], errors="coerce")
                    else:
                        return pd.Series(np.nan, index=df.index)

                df["open_interest_all"] = to_num("open_interest_all")
                df["commercial_long"] = to_num("commercial_long_all")
                df["commercial_short"] = to_num("commercial_short_all")
                df["non_commercial_long"] = to_num("non_commercial_long_all")
                df["non_commercial_short"] = to_num("non_commercial_short_all")

                df["commercial_net"] = df["commercial_long"].fillna(0) - df["commercial_short"].fillna(0)
                df["non_commercial_net"] = df["non_commercial_long"].fillna(0) - df["non_commercial_short"].fillna(0)

                # Position % (safe denom)
                df["commercial_position_pct"] = np.where(
                    (df["commercial_long"] + df["commercial_short"]) > 0,
                    df["commercial_long"] / (df["commercial_long"] + df["commercial_short"]) * 100,
                    50.0,
                )
                df["non_commercial_position_pct"] = np.where(
                    (df["non_commercial_long"] + df["non_commercial_short"]) > 0,
                    df["non_commercial_long"] / (df["non_commercial_long"] + df["non_commercial_short"]) * 100,
                    50.0,
                )

                # Rolling z-scores (52-week ~ 52 reports)
                df = df.sort_values("report_date")
                df["commercial_net_zscore"] = (df["commercial_net"] - df["commercial_net"].rolling(52, min_periods=1).mean()) / df["commercial_net"].rolling(52, min_periods=1).std().replace(0, np.nan)
                df["non_commercial_net_zscore"] = (df["non_commercial_net"] - df["non_commercial_net"].rolling(52, min_periods=1).mean()) / df["non_commercial_net"].rolling(52, min_periods=1).std().replace(0, np.nan)

                return df.sort_values("report_date").reset_index(drop=True)
            else:
                logger.warning(f"No COT data for {market_name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching COT data for {market_name}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching COT data for {market_name} after {max_attempts} attempts.")
    return pd.DataFrame()

# --- Fetch Price Data (yahooquery) ---
def fetch_yahooquery_data(ticker: str, start_date: str = None, end_date: str = None, max_attempts: int = 3) -> pd.DataFrame:
    """
    If start_date/end_date not provided, default to last 2 years.
    Returns DataFrame with a 'date' column (datetime) and typical OHLCV columns.
    """
    logger.info(f"Fetching Yahoo data for {ticker} from {start_date} to {end_date}")
    if end_date is None:
        end_date = datetime.date.today().isoformat()
    if start_date is None:
        start_date = (datetime.date.today() - datetime.timedelta(days=365 * 2)).isoformat()

    attempt = 0
    while attempt < max_attempts:
        try:
            t = Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")
            if isinstance(hist, pd.DataFrame) and hist.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()
            # yahooquery sometimes returns MultiIndex if multiple tickers specified
            if isinstance(hist.index, pd.MultiIndex):
                # If ticker present as level 0
                if ticker in hist.index.levels[0]:
                    hist = hist.loc[ticker]
                else:
                    hist = hist.reset_index(level=0, drop=True)

            hist = hist.reset_index()
            # Normalize column names to lowercase
            hist.columns = [c.lower() for c in hist.columns]
            if "date" not in hist.columns and "index" in hist.columns:
                hist = hist.rename(columns={"index": "date"})
            hist["date"] = pd.to_datetime(hist["date"])
            hist = hist.sort_values("date").reset_index(drop=True)

            # Calculate technical indicators
            hist = calculate_technical_indicators(hist)

            return hist
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {e}")
        attempt += 1
        time.sleep(2)
    logger.error(f"Failed fetching Yahoo data for {ticker} after {max_attempts} attempts.")
    return pd.DataFrame()

# --- Calculate Technical Indicators ---
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    close_col = "close" if "close" in df.columns else None
    high_col = "high" if "high" in df.columns else None
    low_col = "low" if "low" in df.columns else None

    if not all([close_col, high_col, low_col]):
        return df

    # RVOL
    vol_col = "volume" if "volume" in df.columns else None
    if vol_col is not None:
        df["rvol"] = df[vol_col] / df[vol_col].rolling(20, min_periods=1).mean()

    # RSI
    delta = df[close_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MAs
    df["sma20"] = df[close_col].rolling(20, min_periods=1).mean()
    df["sma50"] = df[close_col].rolling(50, min_periods=1).mean()
    df["sma200"] = df[close_col].rolling(200, min_periods=1).mean()

    # Bollinger Bands -> use bb_upper / bb_lower and store also bb_width
    df["bb_middle"] = df[close_col].rolling(20, min_periods=1).mean()
    df["bb_std"] = df[close_col].rolling(20, min_periods=1).std()
    df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)

    # ATR (named 'atr' for consistency)
    tr1 = df[high_col] - df[low_col]
    tr2 = (df[high_col] - df[close_col].shift()).abs()
    tr3 = (df[low_col] - df[close_col].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=1).mean()

    # Volatility annualized %
    df["volatility"] = df[close_col].pct_change().rolling(20, min_periods=1).std() * np.sqrt(252) * 100

    # 52-week high/low
    df["52w_high"] = df[close_col].rolling(252, min_periods=1).max()
    df["52w_low"] = df[close_col].rolling(252, min_periods=1).min()
    df["pct_from_52w_high"] = (df[close_col] / df["52w_high"] - 1) * 100
    df["pct_from_52w_low"] = (df[close_col] / df["52w_low"] - 1) * 100

    return df







# --- Merge COT and Price Data ---
def merge_cot_price(cot_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    if cot_df is None or cot_df.empty or price_df is None or price_df.empty:
        return pd.DataFrame()

    cot_columns = ["report_date", "open_interest_all", "commercial_net", "non_commercial_net",
                   "commercial_position_pct", "non_commercial_position_pct",
                   "commercial_net_zscore", "non_commercial_net_zscore"]

    for col in cot_columns:
        if col not in cot_df.columns:
            cot_df[col] = np.nan

    cot_df_small = cot_df[cot_columns].copy()
    cot_df_small = cot_df_small.rename(columns={"report_date": "date"})
    cot_df_small["date"] = pd.to_datetime(cot_df_small["date"])

    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values("date").reset_index(drop=True)
    cot_df_small = cot_df_small.sort_values("date").reset_index(drop=True)

    full_dates = pd.DataFrame({"date": pd.date_range(price_df["date"].min(), price_df["date"].max())})
    cot_df_filled = pd.merge_asof(full_dates, cot_df_small, on="date", direction="backward")

    merged = pd.merge(price_df, cot_df_filled, on="date", how="left")
    # forward fill COT numeric fields
    merged = merged.sort_values("date").reset_index(drop=True)
    for col in cot_columns[1:]:
        merged[col] = merged[col].ffill()
    # keep consistent column names for downstream logic
    merged = merged.rename(columns={
        "open_interest_all": "open_interest_all",
        "commercial_net": "commercial_net",
        "non_commercial_net": "noncommercial_net",  # unify naming used elsewhere
        "commercial_net_zscore": "commercial_net_zscore",
        "non_commercial_net_zscore": "noncommercial_net_zscore"
    })
    return merged

# --- Calculate Health Gauge (expects merged_df) ---
def calculate_health_gauge(merged_df: pd.DataFrame) -> float:
    if merged_df is None or merged_df.empty:
        return np.nan

    latest = merged_df.tail(1).iloc[0]
    recent = merged_df.tail(90).copy()

    close_col = "close" if "close" in recent.columns else None
    if close_col is None:
        return np.nan

    scores = []

    # 1. Commercial net position extreme score (25%)
    if "commercial_net_zscore" in latest and not pd.isna(latest["commercial_net_zscore"]):
        comm_score = max(0, min(1, 0.5 - latest["commercial_net_zscore"] / 4))
        scores.append((comm_score, 0.25))

    # 2. Trend alignment score (20%)
    if all(x in latest.index for x in ["sma20", "sma50", "sma200"]):
        last_close = latest[close_col]
        trend_signals = [
            last_close > latest["sma20"],
            latest["sma20"] > latest["sma50"],
            latest["sma50"] > latest["sma200"],
        ]
        trend_score = sum(trend_signals) / len(trend_signals)
        scores.append((trend_score, 0.20))

    # 3. Momentum score (15%)
    if "rsi" in latest.index:
        rsi = latest["rsi"]
        if pd.isna(rsi):
            rsi_score = 0.5
        elif rsi < 30:
            rsi_score = 0.3
        elif rsi > 70:
            rsi_score = 0.7
        else:
            rsi_score = 0.5 + (rsi - 50) / 100
        scores.append((rsi_score, 0.15))

    # 4. Volatility and volume score (15%)
    if "bb_width" in recent.columns and "rvol" in recent.columns:
        bb_width_percentile = stats.percentileofscore(recent["bb_width"].dropna(), latest.get("bb_width", np.nan)) / 100 if not recent["bb_width"].dropna().empty else 0.5
        bb_score = 1 - bb_width_percentile
        rvol_score = min(1.0, latest.get("rvol", 0) / 2.0) if not pd.isna(latest.get("rvol", np.nan)) else 0.5
        vol_score = 0.7 * bb_score + 0.3 * rvol_score
        scores.append((vol_score, 0.15))

    # 5. Distance from 52-week high/low score (15%)
    if "pct_from_52w_high" in latest.index and "pct_from_52w_low" in latest.index:
        high_score = 1 - (abs(latest["pct_from_52w_high"]) / 100)
        high_score = max(0, min(1, high_score))
        low_score = min(1, latest["pct_from_52w_low"] / 100)
        low_score = max(0, min(1, low_score))
        dist_score = 0.7 * high_score + 0.3 * low_score
        scores.append((dist_score, 0.15))

    # 6. Open interest (10%)
    if "open_interest_all" in recent.columns:
        oi = recent["open_interest_all"].dropna()
        if not oi.empty:
            oi_pctile = stats.percentileofscore(oi, latest.get("open_interest_all", np.nan)) / 100
            scores.append((oi_pctile, 0.10))

    if not scores:
        return 5.0

    weighted_sum = sum(score * weight for score, weight in scores)
    total_weight = sum(weight for _, weight in scores)
    health_score = (weighted_sum / total_weight) * 10
    return float(health_score)

# --- Single signal generator (merged_df + cot_df) ---
def generate_signal_from_merged(merged_df: pd.DataFrame, cot_df: pd.DataFrame) -> dict:
    if merged_df is None or merged_df.empty:
        return {"signal": "NEUTRAL", "strength": 0, "reasoning": "Insufficient data"}

    close_col = "close" if "close" in merged_df.columns else None
    if close_col is None:
        return {"signal": "NEUTRAL", "strength": 0, "reasoning": "Price data missing"}

    recent = merged_df.tail(30).copy()
    latest = recent.iloc[-1]

    signal_reasons = []
    bullish_points = 0
    bearish_points = 0

    # COT signals (from merged_df if available)
    if "commercial_net_zscore" in latest and not pd.isna(latest["commercial_net_zscore"]):
        z = latest["commercial_net_zscore"]
        if z < -1.5:
            bullish_points += 2
            signal_reasons.append("Commercials heavily net long (contrarian bullish)")
        elif z < -0.5:
            bullish_points += 1
            signal_reasons.append("Commercials moderately net long")
        elif z > 1.5:
            bearish_points += 2
            signal_reasons.append("Commercials heavily net short (contrarian bearish)")
        elif z > 0.5:
            bearish_points += 1
            signal_reasons.append("Commercials moderately net short")

    # Trend
    if all(x in latest.index for x in ["sma20", "sma50", "sma200"]):
        if latest[close_col] > latest["sma20"] > latest["sma50"] > latest["sma200"]:
            bullish_points += 2
            signal_reasons.append("Strong uptrend")
        elif latest[close_col] > latest["sma50"] and latest["sma50"] > latest["sma200"]:
            bullish_points += 1
            signal_reasons.append("Uptrend")
        elif latest[close_col] < latest["sma20"] < latest["sma50"] < latest["sma200"]:
            bearish_points += 2
            signal_reasons.append("Strong downtrend")
        elif latest[close_col] < latest["sma50"] and latest["sma50"] < latest["sma200"]:
            bearish_points += 1
            signal_reasons.append("Downtrend")

    # RSI
    if "rsi" in latest.index and not pd.isna(latest["rsi"]):
        if latest["rsi"] < 30:
            bullish_points += 1
            signal_reasons.append("RSI oversold")
        elif latest["rsi"] > 70:
            bearish_points += 1
            signal_reasons.append("RSI overbought")

    # Bollinger
    if all(x in latest.index for x in ["bb_upper", "bb_lower", close_col]):
        if latest[close_col] > latest["bb_upper"]:
            bearish_points += 1
            signal_reasons.append("Price above upper Bollinger Band")
        elif latest[close_col] < latest["bb_lower"]:
            bullish_points += 1
            signal_reasons.append("Price below lower Bollinger Band")

    # Volume
    if "rvol" in latest.index and not pd.isna(latest["rvol"]):
        if latest["rvol"] > 1.5:
            if len(recent) > 1:
                price_change = latest[close_col] - recent.iloc[-2][close_col]
                if price_change > 0:
                    bullish_points += 1
                    signal_reasons.append("High volume on price advance")
                elif price_change < 0:
                    bearish_points += 1
                    signal_reasons.append("High volume on price decline")

    # 52-week proximity
    if "pct_from_52w_high" in latest.index and "pct_from_52w_low" in latest.index:
        if latest["pct_from_52w_high"] > -5:
            bullish_points += 1
            signal_reasons.append("Price near 52-week high")
        elif latest["pct_from_52w_low"] < 10:
            bearish_points += 1
            signal_reasons.append("Price near 52-week low")

    net_score = bullish_points - bearish_points
    if net_score >= 3:
        signal = "STRONG BUY"
        strength = min(5, int(net_score))
    elif net_score > 0:
        signal = "BUY"
        strength = min(3, int(net_score))
    elif net_score <= -3:
        signal = "STRONG SELL"
        strength = min(5, int(abs(net_score)))
    elif net_score < 0:
        signal = "SELL"
        strength = min(3, int(abs(net_score)))
    else:
        signal = "NEUTRAL"
        strength = 0

    # Health gauge if present in merged_df
    try:
        health_score = calculate_health_gauge(merged_df)
        signal_reasons.append(f"Health gauge: {health_score:.2f}/10")
    except Exception:
        pass

    return {"signal": signal, "strength": strength, "reasoning": "; ".join(signal_reasons)}

# --- Trade Setup Generation ---
def generate_trade_setup(signal: str, df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    latest = df.iloc[-1]
    atr = latest.get("atr", np.nan)

    setup = {"signal": signal}
    if signal == "BUY":
        setup["stop_loss"] = latest["close"] - (2 * atr) if not pd.isna(atr) else None
        setup["target"] = latest["close"] + (3 * atr) if not pd.isna(atr) else None
    elif signal == "SELL":
        setup["stop_loss"] = latest["close"] + (2 * atr) if not pd.isna(atr) else None
        setup["target"] = latest["close"] - (3 * atr) if not pd.isna(atr) else None
    else:
        setup["stop_loss"] = None
        setup["target"] = None

    return setup

# --- AI Market Newsletter Generation ---
def build_newsletter_prompt(asset: str, signals: list, df: pd.DataFrame, cot_df: pd.DataFrame) -> str:
    if df is None or df.empty or cot_df is None or cot_df.empty:
        return f"No sufficient data available for {asset}."

    latest = df.iloc[-1]
    cot_latest = cot_df.iloc[-1]
    signal_texts = [f"{s.get('signal','')}: {s.get('reason','')}" for s in signals] if isinstance(signals, list) else [str(signals)]

    prompt = f"""
    Generate a professional market newsletter for {asset}.

    Key Data:
    - Latest close price: {latest['close']:.2f}
    - RSI: {latest.get('rsi', np.nan):.2f}
    - SMA50: {latest.get('sma50', np.nan):.2f}
    - SMA200: {latest.get('sma200', np.nan):.2f}
    - Bollinger Bands: {latest.get('bb_lower', np.nan):.2f} – {latest.get('bb_upper', np.nan):.2f}
    - ATR: {latest.get('atr', np.nan):.2f}
    - Relative Volume: {latest.get('rvol', np.nan):.2f}
    - 52w High/Low: {latest.get('52w_low', np.nan):.2f} – {latest.get('52w_high', np.nan):.2f}

    Commitment of Traders:
    - Non-commercial net positions: {cot_latest.get('non_commercial_net', cot_latest.get('non_commercial_net', np.nan))}
    - Commercial net positions: {cot_latest.get('commercial_net', np.nan)}

    Signals: {', '.join(signal_texts)}

    Write in a structured, concise, professional style.
    """
    return prompt.strip()

def generate_market_newsletter(prompt: str) -> str:
    if hf_client is None:
        return "HF client not configured; can't generate newsletter."
    try:
        response = hf_client.text_generation(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            inputs=prompt,
            parameters={"max_new_tokens": 500},
        )
        # response shape may vary; try common keys
        if isinstance(response, dict):
            return response.get("generated_text", "") or response.get("text", "") or str(response)
        elif isinstance(response, list) and len(response) > 0:
            return response[0].get("generated_text", "") or str(response[0])
        return str(response)
    except Exception as e:
        logger.error(f"Newsletter generation failed: {e}")
        return "Error generating newsletter."

# --- Visualization ---
def create_asset_chart(df: pd.DataFrame, cot_df: pd.DataFrame, asset_name: str):
    # Ensure date-based index
    if "date" in df.columns:
        df_plot = df.set_index("date").copy()
    else:
        df_plot = df.copy()
        df_plot.index = pd.to_datetime(df_plot.index)

    if "date" in cot_df.columns:
        cot_plot = cot_df.set_index("date").copy()
    else:
        cot_plot = cot_df.copy()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f"{asset_name} Price & Indicators", "COT Non-Commercial Net Positions")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot["open"], high=df_plot["high"],
        low=df_plot["low"], close=df_plot["close"], name="Price"
    ), row=1, col=1)

    # Moving averages
    if "sma50" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["sma50"], mode="lines", name="SMA50"), row=1, col=1)
    if "sma200" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["sma200"], mode="lines", name="SMA200"), row=1, col=1)

    # Bollinger Bands
    if "bb_upper" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["bb_upper"], mode="lines", name="BB Upper", line=dict(width=1)), row=1, col=1)
    if "bb_lower" in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["bb_lower"], mode="lines", name="BB Lower", line=dict(width=1)), row=1, col=1)

    # COT Net Positions - align by date where possible
    if "non_commercial_net" in cot_plot.columns:
        fig.add_trace(go.Bar(x=cot_plot.index, y=cot_plot["non_commercial_net"], name="Non-Comm Net"), row=2, col=1)
    elif "commercial_net" in cot_plot.columns:
        fig.add_trace(go.Bar(x=cot_plot.index, y=cot_plot["commercial_net"], name="Commercial Net"), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=800)
    return fig

# --- Opportunity Dashboard ---
def create_opportunity_dashboard(opportunities: dict):
    if not opportunities:
        return pd.DataFrame(columns=["ticker", "health_gauge", "signal"])
    rows = []
    for k, v in opportunities.items():
        rows.append({
            "asset": k,
            "ticker": v.get("ticker"),
            "health_gauge": float(v.get("health_gauge", np.nan)),
            "signal": v.get("signal", "NEUTRAL")
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by="health_gauge", ascending=False)
    return df.set_index("asset")

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Market Dashboard", layout="wide")
    st.title("📊 Market Intelligence Dashboard")

    # Sidebar
    st.sidebar.header("Controls")
    selected_date = st.sidebar.date_input("Select Date", datetime.date.today())
    refresh_data = st.sidebar.button("🔄 Refresh Data")
    report_type = st.sidebar.selectbox("Report Type", ["Dashboard", "Detailed Analysis", "Newsletter"])

    if "generate_newsletter" not in st.session_state:
        st.session_state["generate_newsletter"] = False

    if st.sidebar.button("📰 Generate Newsletter"):
        st.session_state["generate_newsletter"] = True

    opportunities = {}
    # Loop through assets
    for cot_name, ticker in ASSET_MAPPING.items():
        cot_df = fetch_cot_data(cot_name)
        price_df = fetch_yahooquery_data(ticker)
        if cot_df is None or cot_df.empty or price_df is None or price_df.empty:
            logger.info(f"Skipping {cot_name} due to missing data.")
            continue

        merged_df = merge_cot_price(cot_df, price_df)
        if merged_df is None or merged_df.empty:
            logger.info(f"Skipping {cot_name} because merged DF empty.")
            continue

        health = calculate_health_gauge(merged_df)
        signal_obj = generate_signal_from_merged(merged_df, cot_df)
        trade_setup = generate_trade_setup(signal_obj["signal"], merged_df) if signal_obj else {}

        opportunities[cot_name] = {
            "ticker": ticker,
            "health_gauge": health,
            "signals": [signal_obj],
            "trade_setup": trade_setup,
            "cot_df": cot_df,
            "price_df": merged_df,
            "signal": signal_obj["signal"],
        }

    # Tabs
    tabs = st.tabs(["📈 Market Dashboard", "🔍 Detailed Analysis", "📰 Market Newsletter"])

    # --- Dashboard Tab ---
    with tabs[0]:
        st.subheader("Market Opportunities")
        dashboard_df = create_opportunity_dashboard(opportunities)
        st.dataframe(dashboard_df[["ticker", "health_gauge", "signal"]])

    # --- Detailed Analysis Tab ---
    with tabs[1]:
        if not opportunities:
            st.info("No opportunities available (data missing).")
        else:
            asset_choice = st.selectbox("Select Asset", list(opportunities.keys()))
            asset_data = opportunities[asset_choice]
            fig = create_asset_chart(asset_data["price_df"], asset_data["cot_df"], asset_choice)
            st.plotly_chart(fig, use_container_width=True)

            st.write("**Signals:**", asset_data["signals"])
            st.write("**Trade Setup:**", asset_data["trade_setup"])
            st.write("**COT Latest:**")
            st.dataframe(asset_data["cot_df"].tail(1).T)
            st.write("**Technical Indicators (latest):**")
            st.dataframe(asset_data["price_df"].tail(1).T)

    # --- Newsletter Tab ---
    with tabs[2]:
        if st.session_state.get("generate_newsletter", False):
            st.subheader("Generated Market Newsletter")
            all_newsletters = []
            for asset, data in opportunities.items():
                prompt = build_newsletter_prompt(asset, data["signals"], data["price_df"], data["cot_df"])
                newsletter = generate_market_newsletter(prompt)
                all_newsletters.append(f"### {asset}\n{newsletter}\n")
            st.markdown("\n\n".join(all_newsletters))
        else:
            st.info("Click '📰 Generate Newsletter' in the sidebar to create a report.")

if __name__ == "__main__":
    main()