"""
Corporate actions data fetcher.

Fetches and computes features from:
  1. yfinance t.actions  — dividends + splits history (no auth, always available)
  2. NSE corporate actions API — buyback, rights, bonus announcements (India)

Features returned (all floats, NaN on failure):
  corp_div_yield_ttm     : trailing 12-month dividend yield
  corp_div_growth_3y     : 3-year CAGR of annual dividends
  corp_div_consistency   : years with ≥1 dividend payment in past 5y (0–5)
  corp_buyback_flag      : 1.0 if buyback announced in last 90d, else 0.0
  corp_rights_flag       : 1.0 if rights issue announced in last 180d
  corp_bonus_flag        : 1.0 if bonus issue/split announced in last 90d
  corp_promoter_buy_flag : 1.0 if promoter holding increased ≥0.5pp QoQ

All features are computed with as_of_date guard to prevent look-ahead bias.
Disk cache: ml_output/corp_cache/{SYMBOL}.pkl, 24h TTL.
"""

import time
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf

from data.nse_session import nse_get

_CACHE_DIR = Path("ml_output/corp_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_NSE_CORP_ACTIONS_URL = (
    "https://www.nseindia.com/api/corporates-corporateActions"
    "?index=equities&symbol={symbol}"
)

_NAN_DICT = {
    'corp_div_yield_ttm':     float('nan'),
    'corp_div_growth_3y':     float('nan'),
    'corp_div_consistency':   float('nan'),
    'corp_buyback_flag':      float('nan'),
    'corp_rights_flag':       float('nan'),
    'corp_bonus_flag':        float('nan'),
    'corp_promoter_buy_flag': float('nan'),
}


def _ticker_to_symbol(ticker: str) -> str:
    """RELIANCE.NS → RELIANCE, 500325.BO → 500325"""
    return ticker.replace('.NS', '').replace('.BO', '')


def _load_cache(symbol: str) -> dict | None:
    path = _CACHE_DIR / f"{symbol}.pkl"
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > 24:
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(symbol: str, data: dict) -> None:
    try:
        with open(_CACHE_DIR / f"{symbol}.pkl", 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass


def _div_features(actions: pd.DataFrame, current_price: float,
                  as_of: pd.Timestamp) -> dict:
    """Compute dividend features from yfinance actions DataFrame."""
    out = {
        'corp_div_yield_ttm':   float('nan'),
        'corp_div_growth_3y':   float('nan'),
        'corp_div_consistency': float('nan'),
    }
    if actions is None or actions.empty:
        return out
    if current_price <= 0:
        return out

    # Normalise timezone
    idx = actions.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    as_of_n = as_of.tz_localize(None) if as_of.tzinfo else as_of

    divs = actions.get('Dividends', pd.Series(dtype=float))
    if isinstance(divs, pd.DataFrame):
        divs = divs.squeeze()
    divs = divs[divs > 0].copy()
    if divs.empty:
        return out
    divs.index = pd.to_datetime(divs.index).tz_localize(None) if divs.index.tz is not None \
        else pd.to_datetime(divs.index)

    # Clip to as_of (look-ahead guard)
    divs = divs[divs.index <= as_of_n]
    if divs.empty:
        return out

    # TTM yield
    ttm_start = as_of_n - pd.DateOffset(years=1)
    ttm_divs  = divs[divs.index >= ttm_start].sum()
    out['corp_div_yield_ttm'] = float(ttm_divs / current_price)

    # Annual dividends for each of last 5 years
    annual = {}
    for yr_offset in range(5):
        yr_start = as_of_n - pd.DateOffset(years=yr_offset + 1)
        yr_end   = as_of_n - pd.DateOffset(years=yr_offset)
        annual[yr_offset] = float(divs[(divs.index >= yr_start) & (divs.index < yr_end)].sum())

    # Consistency: years with ≥1 payment in past 5y
    out['corp_div_consistency'] = float(sum(1 for v in annual.values() if v > 0))

    # 3-year CAGR: (yr0_sum / yr3_sum)^(1/3) - 1
    yr0 = annual.get(0, 0)   # most recent 12m
    yr3 = annual.get(3, 0)   # 3-4 years ago
    if yr3 > 0 and yr0 > 0:
        out['corp_div_growth_3y'] = float((yr0 / yr3) ** (1 / 3) - 1)

    return out


def _nse_event_flags(symbol: str, as_of: pd.Timestamp) -> dict:
    """
    Query NSE corporate actions API for buyback, rights, bonus within lookback.
    Returns flags as 0.0 / 1.0 / NaN.
    """
    out = {
        'corp_buyback_flag': float('nan'),
        'corp_rights_flag':  float('nan'),
        'corp_bonus_flag':   float('nan'),
    }

    data = nse_get(_NSE_CORP_ACTIONS_URL.format(symbol=symbol))
    if not data:
        return out

    as_of_n = as_of.tz_localize(None) if as_of.tzinfo else as_of
    buyback = 0.0
    rights  = 0.0
    bonus   = 0.0

    for item in (data if isinstance(data, list) else []):
        purpose = str(item.get('purpose', '')).lower()
        ex_date_str = item.get('exDate') or item.get('exdate') or ''
        try:
            ex_date = pd.Timestamp(ex_date_str).tz_localize(None)
        except Exception:
            continue

        # Clip to as_of (look-ahead guard)
        if ex_date > as_of_n:
            continue

        days_ago = (as_of_n - ex_date).days
        if 'buyback' in purpose and days_ago <= 90:
            buyback = 1.0
        if 'rights' in purpose and days_ago <= 180:
            rights = 1.0
        if ('bonus' in purpose or 'split' in purpose) and days_ago <= 90:
            bonus = 1.0

    out['corp_buyback_flag'] = buyback
    out['corp_rights_flag']  = rights
    out['corp_bonus_flag']   = bonus
    return out


def fetch_corporate_actions(ticker: str,
                             stmts: dict,
                             as_of_date: pd.Timestamp | None = None) -> dict:
    """
    Fetch and compute all corporate action features for a ticker.

    Args:
        ticker     : yfinance ticker (e.g. 'RELIANCE.NS')
        stmts      : statements dict from fetch_statements_india() — used for
                     current price (info['currentPrice']) and promoter delta
        as_of_date : clip all data to this date (defaults to today)

    Returns dict with all corp_* keys. All NaN on failure.
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.now().normalize()

    symbol = _ticker_to_symbol(ticker)
    cache  = _load_cache(symbol)
    if cache is not None:
        return cache

    result = dict(_NAN_DICT)

    # ── 1. yfinance actions ──────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t       = yf.Ticker(ticker)
            actions = t.actions
    except Exception:
        actions = None

    info          = stmts.get('info', {}) or {}
    current_price = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)

    # Fallback price from price_hist
    if current_price <= 0:
        ph = stmts.get('price_hist', pd.DataFrame())
        if not ph.empty:
            try:
                current_price = float(ph['Close'].dropna().iloc[-1])
            except Exception:
                pass

    div_feats = _div_features(actions, current_price, as_of_date)
    result.update(div_feats)

    # ── 2. NSE corporate event flags ─────────────────────────────────────
    if ticker.endswith(('.NS', '.BO')):
        nse_feats = _nse_event_flags(symbol, as_of_date)
        result.update(nse_feats)

    # ── 3. Promoter buy proxy from shareholding delta ────────────────────
    sh_promoter_delta = info.get('_sh_promoter_delta_qoq', float('nan'))
    # Try reading from stmts if injected by shareholding pipeline
    if isinstance(sh_promoter_delta, float) and not pd.isna(sh_promoter_delta):
        result['corp_promoter_buy_flag'] = 1.0 if sh_promoter_delta >= 0.5 else 0.0

    _save_cache(symbol, result)
    return result
