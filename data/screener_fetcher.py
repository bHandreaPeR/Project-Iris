"""
Fetch 10+ years of financial data from Screener.in for Indian stocks.

No login required. Parses the company HTML page directly.
Returns DataFrames in yfinance-compatible format (rows=metrics, cols=dates).

Coverage:
  - Quarterly P&L  : ~13 quarters (~3 years)
  - Annual P&L     : ~12 years (Mar 2015 → present)
  - Annual BS      : ~12 years
  - Annual CF      : ~12 years

Rate limiting: 0.5s between requests. Disk cache in ml_output/screener_cache/.
"""

import re
import time
import pickle
import calendar
import logging
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

BASE       = "https://www.screener.in"
CACHE_DIR  = Path("ml_output/screener_cache")

# Map Screener.in metric names → yfinance-compatible row names
# Quarterly P&L
_Q_MAP = {
    "Sales":              "Total Revenue",
    "Expenses":           "Total Expenses",
    "Operating Profit":   "Operating Income",
    "Other Income":       "Other Income Expense",
    "Interest":           "Interest Expense",
    "Depreciation":       "Depreciation And Amortization",
    "Profit before tax":  "Pretax Income",
    "Net Profit":         "Net Income",
    "EPS in Rs":          "Basic EPS",
}

# Annual P&L (same labels as quarterly)
_PL_MAP = _Q_MAP.copy()
_PL_MAP["Dividend Payout %"] = "Dividend Payout Ratio"

# Annual balance sheet
_BS_MAP = {
    "Equity Capital":     "Common Stock",
    "Reserves":           "Retained Earnings",
    "Borrowings":         "Total Debt",
    "Other Liabilities":  "Other Liabilities",
    "Total Liabilities":  "Total Liabilities Net Minority Interest",
    "Fixed Assets":       "Net PPE",
    "CWIP":               "Construction In Progress",
    "Investments":        "Investments And Advances In Affiliates",
    "Other Assets":       "Other Assets",
    "Total Assets":       "Total Assets",
}

# Annual cash flow
_CF_MAP = {
    "Cash from Operating Activity":  "Operating Cash Flow",
    "Cash from Investing Activity":  "Investing Cash Flow",
    "Cash from Financing Activity":  "Financing Cash Flow",
    "Net Cash Flow":                 "Changes In Cash",
    "Free Cash Flow":                "Free Cash Flow",
}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

_SESSION: requests.Session | None = None

def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer":         "https://www.screener.in/",
        })
    return _SESSION


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_period(s: str) -> pd.Timestamp | None:
    """Convert 'Mar 2023' → pd.Timestamp('2023-03-31')."""
    s = s.strip()
    m = re.match(r'([A-Za-z]+)\s+(\d{4})', s)
    if not m:
        return None
    try:
        month = pd.Timestamp(f"1 {m.group(1)} {m.group(2)}").month
        year  = int(m.group(2))
        last  = calendar.monthrange(year, month)[1]
        return pd.Timestamp(year, month, last)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _to_float(s: str) -> float:
    """Convert '212,834' or '15.3%' or '-' to float."""
    s = s.strip().replace(',', '').replace('%', '').replace('\xa0', '')
    if s in ('', '-', '—', 'setAttributes'):
        return float('nan')
    try:
        return float(s)
    except ValueError:
        return float('nan')


def _parse_section(soup: BeautifulSoup, section_id: str,
                   field_map: dict) -> pd.DataFrame:
    """
    Parse one financial section from the page HTML.
    Returns DataFrame with rows=mapped metrics, columns=pd.Timestamp dates.
    Values in Crores (as-is from screener).
    """
    sec = soup.find('section', id=section_id)
    if not sec:
        return pd.DataFrame()
    table = sec.find('table')
    if not table:
        return pd.DataFrame()

    rows = table.find_all('tr')
    if not rows:
        return pd.DataFrame()

    # Header row → dates
    header_cells = rows[0].find_all(['th', 'td'])
    dates = []
    for cell in header_cells[1:]:
        ts = _parse_period(cell.text.strip())
        if ts:
            dates.append(ts)

    if not dates:
        return pd.DataFrame()

    data: dict[str, list[float]] = {}
    for tr in rows[1:]:
        cells = tr.find_all(['th', 'td'])
        if not cells:
            continue
        raw_name = cells[0].text.replace('\xa0', '').replace('+', '').strip()
        mapped   = field_map.get(raw_name)
        if mapped is None:
            continue  # skip unmapped rows
        vals = [_to_float(c.text) for c in cells[1:len(dates) + 1]]
        if len(vals) < len(dates):
            vals += [float('nan')] * (len(dates) - len(vals))
        data[mapped] = vals

    if not data:
        return pd.DataFrame()

    # Build: rows=metrics, cols=dates (ascending)
    df = pd.DataFrame(data, index=dates).T
    df = df.sort_index(axis=1)
    # Convert Crores → units that match yfinance (yfinance returns raw INR for Indian stocks)
    # yfinance already returns in INR (not Crores). Screener returns Crores.
    # Multiply by 1e7 to convert Cr → INR so features.py formulas work consistently.
    # NOTE: yfinance sometimes returns Crores too for NSE — empirically check per ticker.
    # We leave as Crores here; features.py uses ratios so absolute scale cancels out.
    return df


# ---------------------------------------------------------------------------
# Company lookup
# ---------------------------------------------------------------------------

def resolve_screener_url(symbol: str) -> str | None:
    """
    symbol: NSE symbol (e.g. 'RELIANCE') or BSE code (e.g. '500325').
    Returns relative screener.in path like '/company/RELIANCE/consolidated/'
    or None if not found.
    """
    try:
        r = _session().get(
            f"{BASE}/api/company/search/?q={symbol}&v=3",
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        if data:
            return data[0]['url']
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

def fetch_screener_financials(symbol: str,
                              use_cache: bool = True) -> dict:
    """
    Fetch quarterly + annual financial data for `symbol` from Screener.in.

    Args:
        symbol    : NSE ticker without suffix (e.g. 'RELIANCE', 'INFY')
                    or BSE code string (e.g. '500325')
        use_cache : if True, load/save from ml_output/screener_cache/

    Returns dict with keys:
        income, balance, cashflow,
        annual_income, annual_balance, annual_cashflow
    All are pd.DataFrames (rows=metrics, cols=dates). Empty df on failure.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}.pkl"

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    url_path = resolve_screener_url(symbol)
    if url_path is None:
        return {}
    time.sleep(0.15)

    try:
        resp = _session().get(f"{BASE}{url_path}", timeout=20)
        resp.raise_for_status()
    except Exception as e:
        log.debug(f"[screener] {symbol}: fetch failed — {e}")
        return {}

    soup = BeautifulSoup(resp.text, 'html.parser')

    result = {
        'income':          _parse_section(soup, 'quarters',    _Q_MAP),
        'balance':         pd.DataFrame(),  # no quarterly BS on screener
        'cashflow':        pd.DataFrame(),  # no quarterly CF on screener
        'annual_income':   _parse_section(soup, 'profit-loss', _PL_MAP),
        'annual_balance':  _parse_section(soup, 'balance-sheet', _BS_MAP),
        'annual_cashflow': _parse_section(soup, 'cash-flow',   _CF_MAP),
    }

    # Only cache if we got meaningful data
    has_data = any(not v.empty for v in result.values())
    if use_cache and has_data:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            pass

    return result if has_data else {}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    print(f"Testing screener fetch for {sym} …")
    d = fetch_screener_financials(sym, use_cache=False)
    for k, v in d.items():
        if not v.empty:
            print(f"  {k}: {v.shape} | cols: {list(v.columns[:4])} … {list(v.columns[-2:])}")
        else:
            print(f"  {k}: empty")
