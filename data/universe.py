"""
Stock universe definitions — fetched live from official exchange sources.

India  : Nifty 500 constituent list from NSE archives CSV (official, no auth needed)
         https://archives.nseindia.com/content/indices/ind_nifty500list.csv
US     : S&P 500 constituent list from Wikipedia (sourced from official S&P index)
         https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

Lists are cached locally so the model can run offline after first fetch.
Call refresh_universe() to force a re-download.
"""

import os
import time
import pandas as pd
import requests
from pathlib import Path

CACHE_DIR = Path('ml_output')
CACHE_DIR.mkdir(exist_ok=True)

_NSE_NIFTY500_URL = (
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nseindia.com/",
}


# ---------------------------------------------------------------------------
# India — Nifty 500 (live from NSE)
# ---------------------------------------------------------------------------

def fetch_nifty500(force_refresh: bool = False) -> list[str]:
    """
    Fetch Nifty 500 constituents from NSE and return as yfinance tickers (.NS).
    Caches to ml_output/nifty500_tickers.csv for offline reuse.
    """
    cache_file = CACHE_DIR / "nifty500_tickers.csv"

    if not force_refresh and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 168:  # refresh weekly
            df = pd.read_csv(cache_file)
            return df["ticker"].tolist()

    print("[universe] Fetching Nifty 500 from NSE archives …")
    try:
        resp = requests.get(_NSE_NIFTY500_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        # NSE CSV columns: Company Name, Industry, Symbol, Series, ISIN Code
        symbols = df["Symbol"].dropna().str.strip().tolist()
        tickers = [f"{s}.NS" for s in symbols]
        pd.DataFrame({"ticker": tickers}).to_csv(cache_file, index=False)
        print(f"[universe] Nifty 500: {len(tickers)} tickers fetched and cached.")
        return tickers
    except Exception as e:
        print(f"[universe] NSE fetch failed ({e}). Falling back to cache or static list.")
        if cache_file.exists():
            return pd.read_csv(cache_file)["ticker"].tolist()
        return _INDIA_STATIC_FALLBACK


def fetch_sp500(force_refresh: bool = False) -> list[str]:
    """
    Fetch S&P 500 constituents from Wikipedia (sourced from official S&P index data).
    Caches to ml_output/sp500_tickers.csv for offline reuse.
    """
    cache_file = CACHE_DIR / "sp500_tickers.csv"

    if not force_refresh and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 168:
            df = pd.read_csv(cache_file)
            return df["ticker"].tolist()

    print("[universe] Fetching S&P 500 from Wikipedia …")
    try:
        import io
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=_HEADERS, timeout=30,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
        df = tables[0]
        # Wikipedia column: "Symbol" — dots in some tickers (e.g. BRK.B → BRK-B for yfinance)
        tickers = (
            df["Symbol"]
            .dropna()
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        pd.DataFrame({"ticker": tickers}).to_csv(cache_file, index=False)
        print(f"[universe] S&P 500: {len(tickers)} tickers fetched and cached.")
        return tickers
    except Exception as e:
        print(f"[universe] S&P 500 fetch failed ({e}). Falling back to cache or static list.")
        if cache_file.exists():
            return pd.read_csv(cache_file)["ticker"].tolist()
        return _US_STATIC_FALLBACK


def get_universe(market: str, force_refresh: bool = False) -> list[str]:
    if market == "india":
        return fetch_nifty500(force_refresh)
    elif market == "us":
        return fetch_sp500(force_refresh)
    elif market == "all":
        return fetch_nifty500(force_refresh) + fetch_sp500(force_refresh)
    raise ValueError(f"Unknown market: {market!r}. Use 'india', 'us', or 'all'.")


def refresh_universe():
    fetch_nifty500(force_refresh=True)
    fetch_sp500(force_refresh=True)


# ---------------------------------------------------------------------------
# Static fallbacks (used only if network and cache both fail)
# Nifty 100 for India, representative subset for US
# ---------------------------------------------------------------------------

_INDIA_STATIC_FALLBACK = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFOSYS.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "WIPRO.NS", "HCLTECH.NS",
    "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "COALINDIA.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "TECHM.NS", "NESTLEIND.NS", "BAJAJFINSV.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS", "M&M.NS", "TATACONSUM.NS", "BRITANNIA.NS", "PIDILITIND.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "GRASIM.NS", "INDUSINDBK.NS", "SBILIFE.NS",
    "HDFCLIFE.NS", "ICICIGI.NS", "HAVELLS.NS", "DABUR.NS", "MARICO.NS",
    "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "GODREJCP.NS", "COLPAL.NS",
    "BERGEPAINT.NS", "TVSMOTOR.NS", "BOSCHLTD.NS", "SIEMENS.NS", "ABB.NS",
    "VEDL.NS", "HINDALCO.NS", "SAIL.NS", "NMDC.NS", "GAIL.NS",
    "BPCL.NS", "IOC.NS", "HINDPETRO.NS", "PETRONET.NS", "PFC.NS",
    "RECLTD.NS", "BANKBARODA.NS", "CANBK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "MUTHOOTFIN.NS", "CHOLAFIN.NS", "SBICARD.NS", "MOTHERSON.NS", "ASHOKLEY.NS",
    "BALKRISIND.NS", "MRF.NS", "DMART.NS", "ZOMATO.NS", "NAUKRI.NS",
    "MPHASIS.NS", "LTIM.NS", "PERSISTENT.NS", "COFORGE.NS", "OFSS.NS",
]

_US_STATIC_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "UNH", "V",
    "XOM", "MA", "LLY", "HD", "PG", "AVGO", "COST", "MRK", "ABBV", "CVX",
    "CRM", "NFLX", "AMD", "TMO", "BAC", "PEP", "ADBE", "WMT", "ACN", "MCD",
    "LIN", "CSCO", "ABT", "TXN", "DHR", "PM", "AMGN", "NEE", "VZ", "INTC",
    "ORCL", "IBM", "QCOM", "HON", "RTX", "UPS", "CAT", "BA", "GE", "GS",
    "MS", "BLK", "C", "WFC", "USB", "AXP", "SCHW", "LOW", "TGT", "SBUX",
    "NKE", "TJX", "UNP", "CSX", "FDX", "DE", "ETN", "LMT", "GD", "REGN",
    "GILD", "VRTX", "BMY", "PFE", "JNJ", "MDT", "SYK", "ELV", "CI", "CVS",
]


# Backward-compat aliases used by older modules
INDIA_TICKERS = _INDIA_STATIC_FALLBACK
US_TICKERS    = _US_STATIC_FALLBACK
