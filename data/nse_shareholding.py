"""
NSE India shareholding pattern and FII/DII flow data.

Data sources (all official, free):
  1. NSE shareholding pattern API — quarterly disclosure (SEBI-mandated)
     https://www.nseindia.com/api/corporate-shareholding-pattern?symbol=SYMBOL
  2. NSE FII/DII daily activity
     https://www.nseindia.com/api/fiidiiTradeReact

NSE requires a session cookie obtained from the main page.
We refresh it automatically on each call.

Features extracted:
  sh_promoter_pct          : promoter + promoter group holding %
  sh_promoter_pledge_pct   : % of promoter shares that are pledged
  sh_fii_pct               : foreign institutional holding %
  sh_dii_pct               : domestic institutional holding %
  sh_mf_pct                : mutual fund holding % (subset of DII)
  sh_retail_pct            : public / retail holding %
  sh_promoter_delta_qoq    : QoQ change in promoter holding (percentage points)
  sh_fii_delta_qoq         : QoQ change in FII holding
  sh_dii_delta_qoq         : QoQ change in DII holding
  sh_pledge_delta_qoq      : QoQ change in pledge %
"""

import time
import math
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd

_CACHE_DIR = Path("ml_output/nse_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}

_session = requests.Session()
_session_initialized = False


def _init_session():
    """Prime the NSE session by visiting the main page to get cookies."""
    global _session_initialized
    if _session_initialized:
        return
    try:
        _session.get(
            "https://www.nseindia.com",
            headers=_SESSION_HEADERS,
            timeout=15,
        )
        time.sleep(1.0)
        _session_initialized = True
    except Exception:
        pass


def _nse_get(url: str, params: dict | None = None) -> dict | list | None:
    """Make an authenticated NSE API request, handling session refresh."""
    global _session_initialized
    _init_session()
    try:
        resp = _session.get(
            url, params=params, headers=_SESSION_HEADERS, timeout=20
        )
        if resp.status_code == 401 or resp.status_code == 403:
            # Session expired — reinit
            _session_initialized = False
            _init_session()
            resp = _session.get(
                url, params=params, headers=_SESSION_HEADERS, timeout=20
            )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[nse_shareholding] API call failed: {e}")
        return None


def _symbol(ticker: str) -> str:
    """Strip .NS / .BO suffix for NSE API queries."""
    return ticker.upper().replace(".NS", "").replace(".BO", "")


def fetch_shareholding_raw(ticker: str) -> list[dict] | None:
    """
    Fetch the raw shareholding pattern JSON from NSE for the past 4 quarters.
    Returns list of quarterly records or None if unavailable.
    """
    symbol = _symbol(ticker)
    cache_file = _CACHE_DIR / f"sh_{symbol}.json"

    # Cache for 24 hours (shareholding is quarterly, so daily refresh is enough)
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
        import json
        with open(cache_file) as f:
            return json.load(f)

    data = _nse_get(
        "https://www.nseindia.com/api/corporate-shareholding-pattern",
        params={"symbol": symbol},
    )
    if not data:
        return None

    # NSE returns: {"data": [...]} or just [...]
    records = data.get("data", data) if isinstance(data, dict) else data
    if not isinstance(records, list) or len(records) == 0:
        return None

    import json
    with open(cache_file, "w") as f:
        json.dump(records, f)
    time.sleep(0.5)
    return records


def _parse_pct(val) -> float:
    """Parse a percentage string or float into a 0–100 float."""
    try:
        return float(str(val).replace("%", "").strip())
    except (ValueError, TypeError):
        return float('nan')


def shareholding_features(ticker: str) -> dict:
    """
    Compute shareholding features for the most recent quarter + QoQ deltas.
    Returns dict of float features. All NaN if data unavailable.
    """
    nan = float('nan')
    default = {
        'sh_promoter_pct':        nan,
        'sh_promoter_pledge_pct': nan,
        'sh_fii_pct':             nan,
        'sh_dii_pct':             nan,
        'sh_mf_pct':              nan,
        'sh_retail_pct':          nan,
        'sh_promoter_delta_qoq':  nan,
        'sh_fii_delta_qoq':       nan,
        'sh_dii_delta_qoq':       nan,
        'sh_pledge_delta_qoq':    nan,
    }

    if not ticker.endswith(('.NS', '.BO')):
        return default  # Only applicable to Indian stocks

    records = fetch_shareholding_raw(ticker)
    if not records or len(records) < 1:
        return default

    def _extract(rec: dict) -> dict:
        """Extract key fields from one NSE shareholding record."""
        # NSE field names vary by API version — try multiple keys
        def _pct(*keys):
            for k in keys:
                if k in rec and rec[k] not in (None, '', '-'):
                    return _parse_pct(rec[k])
            return float('nan')

        promoter = _pct('promoterAndPromoterGroupShareholding',
                        'promoter_pct', 'promoterHolding')
        pledge   = _pct('promoterPledge', 'promoter_pledge_pct',
                        'pledgedSharesPromoter')
        fii      = _pct('fiiHolding', 'fii_pct', 'foreignHolding')
        dii      = _pct('diiHolding', 'dii_pct', 'domesticHolding')
        mf       = _pct('mutualFundHolding', 'mf_pct')
        retail   = _pct('publicHolding', 'retail_pct', 'public_pct')
        return {
            'promoter': promoter,
            'pledge':   pledge,
            'fii':      fii,
            'dii':      dii,
            'mf':       mf,
            'retail':   retail,
        }

    latest = _extract(records[0])
    prior  = _extract(records[1]) if len(records) >= 2 else {}

    def delta(key):
        curr = latest.get(key, nan)
        prev = prior.get(key, nan)
        return curr - prev if not (math.isnan(curr) or math.isnan(prev)) else nan

    return {
        'sh_promoter_pct':        latest['promoter'],
        'sh_promoter_pledge_pct': latest['pledge'],
        'sh_fii_pct':             latest['fii'],
        'sh_dii_pct':             latest['dii'],
        'sh_mf_pct':              latest['mf'],
        'sh_retail_pct':          latest['retail'],
        'sh_promoter_delta_qoq':  delta('promoter'),
        'sh_fii_delta_qoq':       delta('fii'),
        'sh_dii_delta_qoq':       delta('dii'),
        'sh_pledge_delta_qoq':    delta('pledge'),
    }


# ---------------------------------------------------------------------------
# FII / DII daily flow (market-wide, used as macro feature)
# ---------------------------------------------------------------------------

def fetch_fii_dii_flow() -> dict:
    """
    Fetch today's FII and DII net buying/selling from NSE.
    Returns dict: fii_net_cr (₹ crore), dii_net_cr.
    """
    nan = float('nan')
    data = _nse_get("https://www.nseindia.com/api/fiidiiTradeReact")
    if not data:
        return {'macro_fii_net_cr': nan, 'macro_dii_net_cr': nan}

    try:
        records = data if isinstance(data, list) else data.get("data", [])
        # Most recent row
        row = records[0] if records else {}
        fii_net = _parse_pct(row.get("netVal_fii", nan))
        dii_net = _parse_pct(row.get("netVal_dii", nan))
        return {'macro_fii_net_cr': fii_net, 'macro_dii_net_cr': dii_net}
    except Exception:
        return {'macro_fii_net_cr': nan, 'macro_dii_net_cr': nan}
