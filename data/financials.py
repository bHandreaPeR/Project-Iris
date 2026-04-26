"""
Fetch quarterly financial statements for a single ticker via yfinance.

Returns point-in-time quarterly DataFrames so the feature engineer can
build a look-ahead-bias-free panel:
  - income      : quarterly income statement  (columns = quarter-end dates)
  - balance     : quarterly balance sheet
  - cashflow    : quarterly cash flow statement
  - price_hist  : daily OHLCV for return computation
  - info        : latest fundamental info dict (market cap, shares, etc.)

Note on Indian stocks (.NS):
  yfinance carries quarterly financials for many NSE large-caps but coverage
  is thinner than for US stocks.  Missing values are NaN — the feature
  engineer handles them gracefully.
"""

import time
import pandas as pd
import yfinance as yf


def fetch_statements(ticker: str) -> dict:
    """
    Fetch all four financial tables + price history for one ticker.
    Returns a dict with keys: income, balance, cashflow, price_hist, info.
    """
    t = yf.Ticker(ticker)

    def _get(attr):
        try:
            df = getattr(t, attr)
            if df is None or df.empty:
                return pd.DataFrame()
            # columns are period-end Timestamps → sort ascending
            return df.sort_index(axis=1)
        except Exception:
            return pd.DataFrame()

    income   = _get('quarterly_income_stmt')
    balance  = _get('quarterly_balance_sheet')
    cashflow = _get('quarterly_cashflow')

    try:
        price_hist = t.history(period='5y', interval='1d', auto_adjust=True)
    except Exception:
        price_hist = pd.DataFrame()

    try:
        info = t.info or {}
    except Exception:
        info = {}

    return {
        'income':     income,
        'balance':    balance,
        'cashflow':   cashflow,
        'price_hist': price_hist,
        'info':       info,
    }


def safe_get(df: pd.DataFrame, row: str, col) -> float:
    """Return df.loc[row, col] as float, or NaN if missing."""
    try:
        val = df.loc[row, col]
        return float(val) if val is not None else float('nan')
    except (KeyError, TypeError, ValueError):
        return float('nan')


def trailing_sum(df: pd.DataFrame, row: str, n_quarters: int = 4) -> float:
    """TTM (trailing twelve months) sum of a row across last n quarters."""
    if df.empty or row not in df.index:
        return float('nan')
    vals = df.loc[row].dropna()
    if len(vals) < n_quarters:
        return float('nan')
    return float(vals.iloc[-n_quarters:].sum())


def yoy_growth(df: pd.DataFrame, row: str, col_idx: int = -1) -> float:
    """Year-over-year growth of df.loc[row] at column col_idx."""
    if df.empty or row not in df.index:
        return float('nan')
    vals = df.loc[row].dropna()
    if len(vals) < 5:
        return float('nan')
    curr = float(vals.iloc[col_idx])
    prior = float(vals.iloc[col_idx - 4])
    if prior == 0 or prior != prior:
        return float('nan')
    return (curr - prior) / abs(prior)


def qoq_growth(df: pd.DataFrame, row: str) -> float:
    """Quarter-over-quarter growth of df.loc[row] at most recent vs prior quarter."""
    if df.empty or row not in df.index:
        return float('nan')
    vals = df.loc[row].dropna()
    if len(vals) < 2:
        return float('nan')
    curr  = float(vals.iloc[-1])
    prior = float(vals.iloc[-2])
    if prior == 0 or prior != prior:
        return float('nan')
    return (curr - prior) / abs(prior)
