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

_NSE_NIFTY500_URL    = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
# Full NSE equity list (all EQ-series listed stocks, ~2200 tickers)
_NSE_ALL_EQUITY_URL  = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
# BSE equity list — all active BSE equities with ISIN, scraped via BSE public API
_BSE_ALL_EQUITY_URL  = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active&PageNo=1&start=0&length=10000"

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


def _fetch_nse_index(url: str, cache_file: Path, label: str,
                     force_refresh: bool = False) -> list[str]:
    """Generic NSE index CSV fetcher (Symbol column)."""
    if not force_refresh and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 168:
            return pd.read_csv(cache_file)["ticker"].tolist()
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        symbols = df["Symbol"].dropna().str.strip().tolist()
        tickers = [f"{s}.NS" for s in symbols]
        pd.DataFrame({"ticker": tickers}).to_csv(cache_file, index=False)
        print(f"[universe] {label}: {len(tickers)} tickers fetched and cached.")
        return tickers
    except Exception as e:
        print(f"[universe] {label} fetch failed ({e}).")
        if cache_file.exists():
            return pd.read_csv(cache_file)["ticker"].tolist()
        return []


def fetch_nse_all(force_refresh: bool = False, min_paid_up: float = 10.0) -> list[str]:
    """
    Fetch the full NSE equity list (all EQ-series stocks, ~2200 tickers).
    Filters to stocks with paid-up capital ≥ min_paid_up crore as a liquidity proxy.
    Nifty 500 tickers are returned first (most liquid first).
    """
    cache_file = CACHE_DIR / "nse_all_equity.csv"
    if not force_refresh and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 168:
            return pd.read_csv(cache_file)["ticker"].tolist()

    print("[universe] Fetching full NSE equity list …")
    try:
        resp = requests.get(_NSE_ALL_EQUITY_URL, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text))
        df.columns = df.columns.str.strip()   # NSE CSV has leading spaces in headers
        eq = df[df['SERIES'].str.strip() == 'EQ'].copy()
        eq['SYMBOL'] = eq['SYMBOL'].str.strip()
        # Use paid-up value as rough liquidity proxy; sort descending
        if 'PAID UP VALUE' in eq.columns:
            eq['paid'] = pd.to_numeric(eq['PAID UP VALUE'], errors='coerce').fillna(0)
            eq = eq[eq['paid'] >= min_paid_up].sort_values('paid', ascending=False)
        tickers = [f"{s}.NS" for s in eq['SYMBOL'].tolist()]
        # Nifty 500 first for priority
        n500 = _fetch_nse_index(_NSE_NIFTY500_URL, CACHE_DIR / "nifty500_tickers.csv",
                                "Nifty 500", force_refresh=False)
        n500_set = set(n500)
        ordered = n500 + [t for t in tickers if t not in n500_set]
        pd.DataFrame({"ticker": ordered}).to_csv(cache_file, index=False)
        print(f"[universe] NSE All Equity: {len(ordered)} tickers (Nifty500 first).")
        return ordered
    except Exception as e:
        print(f"[universe] NSE All Equity fetch failed ({e}). Falling back to Nifty 500.")
        return _fetch_nse_index(_NSE_NIFTY500_URL, CACHE_DIR / "nifty500_tickers.csv",
                                "Nifty 500 (fallback)", force_refresh=False)


def fetch_bse_stocks(force_refresh: bool = False,
                     min_mktcap: float = 100.0) -> tuple[list[str], dict[str, str]]:
    """
    Fetch active BSE equity stocks via BSE public API.
    Returns (bse_tickers, isin_to_bse) where bse_tickers use '.BO' suffix.
    isin_to_bse maps ISIN → BSE ticker for deduplication against NSE.
    min_mktcap filters by market cap (crore) to remove illiquid micro-caps.
    """
    cache_file = CACHE_DIR / "bse_all_equity.csv"
    if not force_refresh and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 168:
            df = pd.read_csv(cache_file)
            isin_map = dict(zip(df["isin"], df["ticker"])) if "isin" in df.columns else {}
            return df["ticker"].tolist(), isin_map

    print("[universe] Fetching BSE equity list …")
    bse_headers = {**_HEADERS, "Referer": "https://www.bseindia.com/"}
    try:
        resp = requests.get(_BSE_ALL_EQUITY_URL, headers=bse_headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # API returns a list directly or {"Table": [...]}
        records = data.get("Table", data) if isinstance(data, dict) else data
        df = pd.DataFrame(records)

        # BSE API columns: SCRIP_CD, ISIN_NUMBER, Mktcap, Scrip_Name, GROUP
        scrip_col = "SCRIP_CD"
        isin_col  = "ISIN_NUMBER"
        mktcap_col = "Mktcap"

        df[scrip_col] = df[scrip_col].astype(str).str.strip()

        # Filter by market cap to remove illiquid shells
        if mktcap_col in df.columns:
            df["_mktcap"] = pd.to_numeric(df[mktcap_col], errors="coerce").fillna(0)
            df = df[df["_mktcap"] >= min_mktcap].sort_values("_mktcap", ascending=False)

        isin_map: dict[str, str] = {}
        tickers: list[str] = []
        for _, row in df.iterrows():
            sc   = str(row[scrip_col]).strip()
            isin = str(row.get(isin_col, "")).strip()
            if not sc or sc == "nan":
                continue
            tkr = f"{sc}.BO"
            tickers.append(tkr)
            if isin and isin != "nan":
                isin_map[isin] = tkr

        out = pd.DataFrame({
            "ticker": tickers,
            "isin": [next((i for i, t in isin_map.items() if t == tkr), "") for tkr in tickers],
        })
        out.to_csv(cache_file, index=False)
        print(f"[universe] BSE All Equity: {len(tickers)} tickers (mktcap ≥ {min_mktcap} cr) fetched and cached.")
        return tickers, isin_map
    except Exception as e:
        print(f"[universe] BSE fetch failed ({e}).")
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            isin_map = dict(zip(df["isin"], df["ticker"])) if "isin" in df.columns else {}
            return df["ticker"].tolist(), isin_map
        return [], {}


def fetch_nifty_india(force_refresh: bool = False) -> list[str]:
    """
    Fetch full India universe: NSE all-equity (Nifty500 first) + BSE-only stocks.
    BSE stocks whose ISIN already appears on NSE are deduplicated (prefer .NS).
    Returns ordered list: Nifty500 (.NS) → remaining NSE (.NS) → BSE-only (.BO).
    """
    cache_file = CACHE_DIR / "india_full_universe.csv"
    if not force_refresh and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 168:
            return pd.read_csv(cache_file)["ticker"].tolist()

    nse_tickers = fetch_nse_all(force_refresh=force_refresh)

    # Build NSE ISIN set for deduplication
    nse_isin_set: set[str] = set()
    try:
        eq_raw = requests.get(_NSE_ALL_EQUITY_URL, headers=_HEADERS, timeout=30)
        eq_raw.raise_for_status()
        eq_df = pd.read_csv(pd.io.common.StringIO(eq_raw.text))
        eq_df.columns = eq_df.columns.str.strip()
        isin_col = next((c for c in eq_df.columns if "ISIN" in c.upper()), None)
        if isin_col:
            nse_isin_set = set(eq_df[isin_col].dropna().str.strip().tolist())
    except Exception:
        pass  # proceed without ISIN dedup if NSE CSV fails

    bse_tickers, bse_isin_map = fetch_bse_stocks(force_refresh=force_refresh)

    # Keep only BSE tickers whose ISIN is NOT already in NSE
    bse_only: list[str] = []
    seen_bse_isins = {isin for isin, t in bse_isin_map.items() if isin in nse_isin_set}
    nse_ticker_set = set(nse_tickers)
    for tkr in bse_tickers:
        if tkr in nse_ticker_set:
            continue
        # Find ISIN for this ticker
        isin = next((i for i, t in bse_isin_map.items() if t == tkr), None)
        if isin and isin in nse_isin_set:
            continue  # same company on NSE, skip
        bse_only.append(tkr)

    combined = nse_tickers + bse_only
    pd.DataFrame({"ticker": combined}).to_csv(cache_file, index=False)
    print(f"[universe] India full universe: {len(nse_tickers)} NSE + {len(bse_only)} BSE-only = {len(combined)} total.")
    return combined


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
        return fetch_nifty_india(force_refresh)
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
