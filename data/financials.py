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

import math
import time
import pandas as pd
import yfinance as yf


def fetch_statements(ticker: str) -> dict:
    """
    Fetch quarterly + annual financial tables, price history, and info.

    Keys returned:
      income, balance, cashflow          — quarterly statements
      annual_income, annual_balance,
      annual_cashflow                    — annual statements (for Piotroski / Beneish)
      price_hist                         — 5-year daily OHLCV
      info                               — yfinance .info dict
    """
    t = yf.Ticker(ticker)

    def _get(attr):
        try:
            df = getattr(t, attr)
            if df is None or df.empty:
                return pd.DataFrame()
            return df.sort_index(axis=1)
        except Exception:
            return pd.DataFrame()

    income          = _get('quarterly_income_stmt')
    balance         = _get('quarterly_balance_sheet')
    cashflow        = _get('quarterly_cashflow')
    annual_income   = _get('income_stmt')
    annual_balance  = _get('balance_sheet')
    annual_cashflow = _get('cashflow')

    try:
        price_hist = t.history(period='5y', interval='1d', auto_adjust=True)
    except Exception:
        price_hist = pd.DataFrame()

    try:
        info = t.info or {}
    except Exception:
        info = {}

    return {
        'income':          income,
        'balance':         balance,
        'cashflow':        cashflow,
        'annual_income':   annual_income,
        'annual_balance':  annual_balance,
        'annual_cashflow': annual_cashflow,
        'price_hist':      price_hist,
        'info':            info,
    }


# ---------------------------------------------------------------------------
# Cash conversion cycle helpers
# ---------------------------------------------------------------------------

def _ttm_val(df: pd.DataFrame, *keys) -> float:
    """TTM sum of the first matched row key."""
    for k in keys:
        if k in df.index:
            vals = df.loc[k].dropna()
            if len(vals) >= 4:
                return float(vals.iloc[-4:].sum())
            elif len(vals) > 0:
                return float(vals.iloc[-1] * (4 / len(vals.iloc[-4:])))
    return float('nan')


def _latest(df: pd.DataFrame, *keys) -> float:
    """Most recent non-null value for first matched row key."""
    for k in keys:
        if k in df.index:
            vals = df.loc[k].dropna()
            if len(vals):
                return float(vals.iloc[-1])
    return float('nan')


def cash_cycle_features(quarterly_inc: pd.DataFrame,
                        quarterly_bs: pd.DataFrame) -> dict:
    """
    Compute DSO, DIO, DPO and cash conversion cycle.
    Uses trailing 12-month revenue/COGS and most-recent balance sheet values.
    All values are in days — NaN if required data unavailable.
    """
    rev_ttm  = _ttm_val(quarterly_inc, 'Total Revenue')
    cogs_ttm = _ttm_val(quarterly_inc, 'Cost Of Revenue', 'Cost Of Goods And Services Sold')
    rec      = _latest(quarterly_bs, 'Accounts Receivable')
    inv      = _latest(quarterly_bs, 'Inventory')
    pay      = _latest(quarterly_bs, 'Accounts Payable')

    def _nan(): return float('nan')
    def _div(a, b): return float('nan') if (math.isnan(a) or math.isnan(b) or b == 0) else a / b

    dso = _div(rec, rev_ttm) * 365  if not (math.isnan(rec) or math.isnan(rev_ttm))  else _nan()
    dio = _div(inv, cogs_ttm) * 365 if not (math.isnan(inv) or math.isnan(cogs_ttm)) else _nan()
    dpo = _div(pay, cogs_ttm) * 365 if not (math.isnan(pay) or math.isnan(cogs_ttm)) else _nan()
    ccc = dso + dio - dpo           if not any(math.isnan(x) for x in [dso, dio, dpo]) else _nan()

    return {
        'wc_dso': dso,
        'wc_dio': dio,
        'wc_dpo': dpo,
        'wc_ccc': ccc,
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
