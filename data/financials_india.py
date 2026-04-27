"""
Unified financial statement fetcher for Indian stocks (.NS / .BO).

Priority:
  1. Screener.in  — 13 quarters + 12 years annual (primary)
  2. yfinance     — ~8 quarters (fallback / supplement)

Merge logic:
  - Financial statements (income/BS/CF): screener wins (deeper history)
  - price_hist, info, earnings_hist: always from yfinance (live data)
  - If screener fails completely, falls back to pure yfinance
"""

import re
import pandas as pd

from data.financials import fetch_statements as _yf_fetch
from data.screener_fetcher import fetch_screener_financials


def _nse_symbol(ticker: str) -> str:
    """RELIANCE.NS → 'RELIANCE'; 500325.BO → '500325'"""
    return re.sub(r'\.(NS|BO)$', '', ticker)


def fetch_statements_india(ticker: str, use_screener_cache: bool = True) -> dict:
    """
    Fetch financial statements for an Indian stock ticker.

    Args:
        ticker: yfinance ticker, e.g. 'RELIANCE.NS' or '500325.BO'
        use_screener_cache: use screener.in disk cache

    Returns the same dict structure as data.financials.fetch_statements():
        income, balance, cashflow,
        annual_income, annual_balance, annual_cashflow,
        price_hist, info, earnings_hist
    """
    # Always need yfinance for price history, info, earnings_history
    yf = _yf_fetch(ticker)

    symbol  = _nse_symbol(ticker)
    screener = fetch_screener_financials(symbol, use_cache=use_screener_cache)

    if not screener:
        return yf  # full fallback

    # Merge: prefer screener for statement depth, yfinance for live/price data
    return {
        'income':          _prefer_deeper(screener.get('income'),          yf['income']),
        'balance':         _prefer_deeper(screener.get('balance'),         yf['balance']),
        'cashflow':        _prefer_deeper(screener.get('cashflow'),        yf['cashflow']),
        'annual_income':   _prefer_deeper(screener.get('annual_income'),   yf['annual_income']),
        'annual_balance':  _prefer_deeper(screener.get('annual_balance'),  yf['annual_balance']),
        'annual_cashflow': _prefer_deeper(screener.get('annual_cashflow'), yf['annual_cashflow']),
        'price_hist':      yf['price_hist'],
        'info':            yf['info'],
        'earnings_hist':   yf['earnings_hist'],
    }


def _prefer_deeper(screener_df: pd.DataFrame | None,
                   yf_df: pd.DataFrame) -> pd.DataFrame:
    """Return whichever DataFrame has more columns (time periods)."""
    if screener_df is None or screener_df.empty:
        return yf_df
    if yf_df.empty:
        return screener_df
    # Prefer the one with more time periods
    return screener_df if screener_df.shape[1] >= yf_df.shape[1] else yf_df
