"""
NSE F&O (Futures & Options) data fetcher.

Fetches live option chain from NSE for F&O-eligible stocks and computes
derivatives-based sentiment and positioning features.

Features returned (all floats, NaN for non-F&O or on failure):
  fo_pcr              : Put-Call Ratio (Σ PE OI / Σ CE OI, nearest expiry)
  fo_pcr_trend        : PCR change vs 5-session rolling cache
  fo_iv_atm           : ATM implied volatility (avg CE + PE at nearest strike)
  fo_iv_pct_52w       : IV percentile within 52-week IV range (0–1)
  fo_oi_change_5d_pct : 5-day % change in total OI
  fo_long_buildup     : 1.0 if price↑ + OI↑ (new longs), else 0.0
  fo_short_buildup    : 1.0 if price↓ + OI↑ (new shorts), else 0.0
  fo_max_pain         : (current_price − max_pain_strike) / current_price

Rolling features (fo_pcr_trend, fo_iv_pct_52w, fo_oi_change_5d_pct) return NaN
until enough history is cached in ml_output/fno_cache/{SYMBOL}_history.json.

NSE API: https://www.nseindia.com/api/option-chain-equities?symbol={SYMBOL}
F&O eligibility: https://archives.nseindia.com/content/fo/fo_mktlots.csv (weekly cache)
"""

import json
import time
import math
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np
import requests

from data.nse_session import nse_get

_CACHE_DIR  = Path("ml_output/fno_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FO_LOTS_URL  = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
_FO_LOTS_CACHE = Path("ml_output/fo_eligible_symbols.csv")
_OC_URL       = "https://www.nseindia.com/api/option-chain-equities"

_NAN_DICT = {
    'fo_pcr':               float('nan'),
    'fo_pcr_trend':         float('nan'),
    'fo_iv_atm':            float('nan'),
    'fo_iv_pct_52w':        float('nan'),
    'fo_oi_change_5d_pct':  float('nan'),
    'fo_long_buildup':      float('nan'),
    'fo_short_buildup':     float('nan'),
    'fo_max_pain':          float('nan'),
}


# ---------------------------------------------------------------------------
# F&O eligibility
# ---------------------------------------------------------------------------

_fo_eligible: set[str] | None = None


def _get_fo_eligible() -> set[str]:
    global _fo_eligible
    if _fo_eligible is not None:
        return _fo_eligible

    # Use weekly cache
    if _FO_LOTS_CACHE.exists():
        age_hours = (time.time() - _FO_LOTS_CACHE.stat().st_mtime) / 3600
        if age_hours < 168:
            try:
                df = pd.read_csv(_FO_LOTS_CACHE)
                _fo_eligible = set(df['symbol'].dropna().str.upper().tolist())
                return _fo_eligible
            except Exception:
                pass

    try:
        resp = requests.get(_FO_LOTS_URL, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp.text), header=0)
        # First column is symbol
        sym_col = df.columns[0]
        symbols = df[sym_col].dropna().str.strip().str.upper().tolist()
        pd.DataFrame({'symbol': symbols}).to_csv(_FO_LOTS_CACHE, index=False)
        _fo_eligible = set(symbols)
        return _fo_eligible
    except Exception:
        _fo_eligible = set()
        return _fo_eligible


def _is_fo_eligible(symbol: str) -> bool:
    return symbol.upper() in _get_fo_eligible()


# ---------------------------------------------------------------------------
# Rolling history cache
# ---------------------------------------------------------------------------

def _history_path(symbol: str) -> Path:
    return _CACHE_DIR / f"{symbol}_history.json"


def _load_history(symbol: str) -> list[dict]:
    p = _history_path(symbol)
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return []


def _append_history(symbol: str, entry: dict) -> list[dict]:
    hist = _load_history(symbol)
    # Deduplicate by date
    today_str = str(date.today())
    hist = [h for h in hist if h.get('date') != today_str]
    hist.append(entry)
    # Keep last 260 sessions (~1 year)
    hist = hist[-260:]
    try:
        with open(_history_path(symbol), 'w') as f:
            json.dump(hist, f)
    except Exception:
        pass
    return hist


# ---------------------------------------------------------------------------
# Option chain parsing
# ---------------------------------------------------------------------------

def _parse_option_chain(data: dict) -> dict | None:
    """
    Parse NSE option chain JSON.
    Returns dict with: underlying_value, expiry_dates, records (per strike/expiry).
    """
    try:
        rec = data.get('records', {})
        filtered = data.get('filtered', {})
        underlying = float(rec.get('underlyingValue', 0))
        expiry_dates = rec.get('expiryDates', [])
        rows = rec.get('data', [])
        return {
            'underlying': underlying,
            'expiry_dates': expiry_dates,
            'rows': rows,
        }
    except Exception:
        return None


def _compute_pcr(rows: list, nearest_expiry: str) -> float:
    """Sum PE OI / CE OI for nearest expiry."""
    pe_oi = 0.0
    ce_oi = 0.0
    for row in rows:
        if row.get('expiryDate') != nearest_expiry:
            continue
        ce = row.get('CE', {}) or {}
        pe = row.get('PE', {}) or {}
        ce_oi += float(ce.get('openInterest', 0) or 0)
        pe_oi += float(pe.get('openInterest', 0) or 0)
    return (pe_oi / ce_oi) if ce_oi > 0 else float('nan')


def _compute_total_oi(rows: list, nearest_expiry: str) -> float:
    total = 0.0
    for row in rows:
        if row.get('expiryDate') != nearest_expiry:
            continue
        ce = row.get('CE', {}) or {}
        pe = row.get('PE', {}) or {}
        total += float(ce.get('openInterest', 0) or 0)
        total += float(pe.get('openInterest', 0) or 0)
    return total


def _compute_atm_iv(rows: list, nearest_expiry: str, underlying: float) -> float:
    """Find ATM strike (nearest to underlying), return avg CE+PE IV."""
    candidates = []
    for row in rows:
        if row.get('expiryDate') != nearest_expiry:
            continue
        strike = float(row.get('strikePrice', 0))
        ce = row.get('CE', {}) or {}
        pe = row.get('PE', {}) or {}
        ce_iv = float(ce.get('impliedVolatility', 0) or 0)
        pe_iv = float(pe.get('impliedVolatility', 0) or 0)
        if ce_iv > 0 and pe_iv > 0:
            candidates.append((abs(strike - underlying), (ce_iv + pe_iv) / 2))
    if not candidates:
        return float('nan')
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _compute_max_pain(rows: list, nearest_expiry: str) -> float:
    """
    Max pain: strike where total option writer profit is maximised.
    Returns the max-pain strike price.
    """
    strikes: dict[float, dict] = {}
    for row in rows:
        if row.get('expiryDate') != nearest_expiry:
            continue
        strike = float(row.get('strikePrice', 0))
        ce = row.get('CE', {}) or {}
        pe = row.get('PE', {}) or {}
        ce_oi = float(ce.get('openInterest', 0) or 0)
        pe_oi = float(pe.get('openInterest', 0) or 0)
        if strike not in strikes:
            strikes[strike] = {'ce': 0.0, 'pe': 0.0}
        strikes[strike]['ce'] += ce_oi
        strikes[strike]['pe'] += pe_oi

    if not strikes:
        return float('nan')

    all_strikes = sorted(strikes.keys())
    min_pain = float('inf')
    max_pain_strike = all_strikes[0]

    for K in all_strikes:
        pain = 0.0
        for S in all_strikes:
            ce_oi = strikes[S]['ce']
            pe_oi = strikes[S]['pe']
            # CE buyer pain: max(S - K, 0) * CE_OI[S]
            pain += max(S - K, 0) * ce_oi
            # PE buyer pain: max(K - S, 0) * PE_OI[S]
            pain += max(K - S, 0) * pe_oi
        if pain < min_pain:
            min_pain = pain
            max_pain_strike = K

    return float(max_pain_strike)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fetch_fno_snapshot(ticker: str, stmts: dict) -> dict:
    """
    Fetch current F&O snapshot for a ticker and return feature dict.
    Returns all-NaN dict for non-F&O tickers or on API failure.

    Args:
        ticker : yfinance ticker (e.g. 'RELIANCE.NS')
        stmts  : statements dict (used for current price from price_hist)
    """
    symbol = ticker.replace('.NS', '').replace('.BO', '').upper()

    if not _is_fo_eligible(symbol):
        return dict(_NAN_DICT)

    data = nse_get(_OC_URL, params={'symbol': symbol})
    if not data:
        return dict(_NAN_DICT)

    parsed = _parse_option_chain(data)
    if not parsed or not parsed['expiry_dates']:
        return dict(_NAN_DICT)

    nearest_expiry = parsed['expiry_dates'][0]
    underlying     = parsed['underlying']
    rows           = parsed['rows']

    if underlying <= 0:
        # Fallback to price_hist
        ph = stmts.get('price_hist', pd.DataFrame())
        if not ph.empty:
            try:
                underlying = float(ph['Close'].dropna().iloc[-1])
            except Exception:
                pass

    result = dict(_NAN_DICT)

    # ── Instantaneous features ────────────────────────────────────────────
    pcr      = _compute_pcr(rows, nearest_expiry)
    iv_atm   = _compute_atm_iv(rows, nearest_expiry, underlying)
    total_oi = _compute_total_oi(rows, nearest_expiry)
    mp_strike = _compute_max_pain(rows, nearest_expiry)

    result['fo_pcr']    = pcr
    result['fo_iv_atm'] = iv_atm
    if underlying > 0 and not math.isnan(mp_strike):
        result['fo_max_pain'] = (underlying - mp_strike) / underlying

    # ── Update rolling history ────────────────────────────────────────────
    today_entry = {
        'date':       str(date.today()),
        'pcr':        pcr if not math.isnan(pcr) else None,
        'iv_atm':     iv_atm if not math.isnan(iv_atm) else None,
        'total_oi':   total_oi,
        'underlying': underlying,
    }
    history = _append_history(symbol, today_entry)

    # ── PCR trend (5-session) ─────────────────────────────────────────────
    valid_pcr = [(h['date'], h['pcr']) for h in history if h.get('pcr') is not None]
    if len(valid_pcr) >= 6:
        recent_pcr = [v[1] for v in valid_pcr[-6:]]
        result['fo_pcr_trend'] = float(recent_pcr[-1] - recent_pcr[-6])

    # ── IV percentile (52-week) ───────────────────────────────────────────
    valid_iv = [h['iv_atm'] for h in history if h.get('iv_atm') is not None]
    if len(valid_iv) >= 20 and not math.isnan(iv_atm):
        iv_lo = min(valid_iv)
        iv_hi = max(valid_iv)
        if iv_hi > iv_lo:
            result['fo_iv_pct_52w'] = (iv_atm - iv_lo) / (iv_hi - iv_lo)

    # ── OI change 5-day ───────────────────────────────────────────────────
    valid_oi = [(h['date'], h['total_oi']) for h in history if h.get('total_oi') is not None]
    if len(valid_oi) >= 6 and total_oi > 0:
        old_oi = valid_oi[-6][1]
        if old_oi > 0:
            result['fo_oi_change_5d_pct'] = (total_oi - old_oi) / old_oi

    # ── Long/short buildup ────────────────────────────────────────────────
    oi_chg_5d = result['fo_oi_change_5d_pct']
    if len(valid_oi) >= 6 and not math.isnan(oi_chg_5d):
        old_price = None
        for h in reversed(history[:-1]):
            if h.get('underlying') is not None and h['underlying'] > 0:
                # Find the 5-session-ago price
                if len([x for x in history if x.get('underlying')]) >= 6:
                    price_hist = [h2['underlying'] for h2 in history
                                  if h2.get('underlying') is not None][-6:]
                    old_price = price_hist[0]
                break

        if old_price is not None and old_price > 0:
            price_chg = (underlying - old_price) / old_price
            oi_up     = oi_chg_5d > 0
            price_up  = price_chg > 0
            result['fo_long_buildup']  = 1.0 if (price_up and oi_up) else 0.0
            result['fo_short_buildup'] = 1.0 if (not price_up and oi_up) else 0.0

    return result
