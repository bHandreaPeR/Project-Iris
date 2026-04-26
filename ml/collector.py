"""
Build the historical panel dataset: (ticker × quarter) → features + forward return.

Design choices that prevent look-ahead bias:
  - We snapshot financials available at each quarter-end date.
  - The forward return is computed from (quarter_end + FILING_LAG_DAYS) to
    (quarter_end + FILING_LAG_DAYS + FORWARD_DAYS).
    Filing lag ≈ 45 days for US (10-Q), ≈ 60 days for India (quarterly results).
    This ensures we only use data a real investor could have acted on.
  - Fundamentals are clipped to columns <= as_of_date before feature engineering.
"""

import math
import time
import traceback
import pandas as pd
import numpy as np

from data.financials import fetch_statements
from ml.features import compute_all

# How many calendar days after quarter-end a real investor has the filing
FILING_LAG_US    = 45
FILING_LAG_INDIA = 60

# Forward return horizon in calendar days
FORWARD_DAYS = 91   # ~3 months; change to 182 for 6-month model


def _filing_lag(ticker: str) -> int:
    return FILING_LAG_INDIA if ticker.endswith('.NS') or ticker.endswith('.BO') else FILING_LAG_US


def _forward_return(price_hist: pd.DataFrame,
                    entry_date: pd.Timestamp,
                    exit_date: pd.Timestamp) -> float:
    """Compute price return between entry_date and exit_date."""
    if price_hist.empty:
        return float('nan')
    ph = price_hist['Close'].sort_index()
    try:
        entry_price = ph.loc[:entry_date].iloc[-1]
        exit_price  = ph.loc[:exit_date].iloc[-1]
        if entry_price == 0:
            return float('nan')
        return float((exit_price - entry_price) / abs(entry_price))
    except (IndexError, KeyError):
        return float('nan')


def _clip_stmts(stmts: dict, as_of: pd.Timestamp) -> dict:
    """Return copies of financial tables with only columns <= as_of."""
    clipped = {}
    for key in ('income', 'balance', 'cashflow'):
        df = stmts[key]
        if df.empty:
            clipped[key] = df
        else:
            cols = [c for c in df.columns if pd.Timestamp(c) <= as_of]
            clipped[key] = df[cols] if cols else pd.DataFrame()
    clipped['price_hist'] = stmts['price_hist']
    clipped['info']       = stmts['info']
    return clipped


def build_ticker_panel(ticker: str, stmts: dict) -> pd.DataFrame:
    """
    For a single ticker, build one row per quarter-end date.
    Each row: all features (computed from data available at that date) +
              fwd_return (the target, from entry to exit defined by filing lag).
    """
    rows = []
    lag  = _filing_lag(ticker)

    # Determine quarterly snapshots from the income statement dates
    inc = stmts['income']
    if inc.empty:
        return pd.DataFrame()

    quarter_ends = sorted([pd.Timestamp(c) for c in inc.columns])

    for qe in quarter_ends:
        entry_date = qe + pd.DateOffset(days=lag)
        exit_date  = entry_date + pd.DateOffset(days=FORWARD_DAYS)

        # Clip data to only what's available at entry_date
        snp = _clip_stmts(stmts, qe)

        # Compute features
        try:
            feats = compute_all(snp, entry_date)
        except Exception:
            continue

        # Skip rows where every feature is NaN (no data at all)
        feat_vals = [v for v in feats.values() if not math.isnan(v)]
        if len(feat_vals) < 5:
            continue

        fwd = _forward_return(stmts['price_hist'], entry_date, exit_date)

        row = {'ticker': ticker, 'quarter_end': qe, 'entry_date': entry_date}
        row.update(feats)
        row['fwd_return'] = fwd
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.set_index(['ticker', 'quarter_end'])
    return df


def build_panel(tickers: list[str],
                cache_path: str | None = None,
                verbose: bool = True) -> pd.DataFrame:
    """
    Build the full cross-sectional panel for a list of tickers.

    Args:
        tickers    : list of yfinance ticker strings
        cache_path : if given, save/load from this parquet path to avoid re-fetching
        verbose    : print progress

    Returns:
        DataFrame indexed by (ticker, quarter_end) with features + fwd_return.
    """
    import os

    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f"[collector] Loading cached panel from {cache_path}")
        return pd.read_parquet(cache_path)

    all_panels = []
    n = len(tickers)

    for i, tkr in enumerate(tickers, 1):
        if verbose:
            print(f"[collector] {i}/{n}  {tkr} …", end='  ', flush=True)
        try:
            stmts = fetch_statements(tkr)
            ticker_df = build_ticker_panel(tkr, stmts)
            if not ticker_df.empty:
                all_panels.append(ticker_df)
                if verbose:
                    print(f"{len(ticker_df)} rows")
            else:
                if verbose:
                    print("no data")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            traceback.print_exc()
        time.sleep(0.3)

    if not all_panels:
        return pd.DataFrame()

    panel = pd.concat(all_panels).sort_index()

    if cache_path:
        panel.to_parquet(cache_path)
        if verbose:
            print(f"[collector] Panel saved to {cache_path}  ({len(panel)} rows)")

    return panel


def summary(panel: pd.DataFrame) -> None:
    """Print a quick data quality summary."""
    print(f"\nPanel shape     : {panel.shape}")
    print(f"Tickers         : {panel.index.get_level_values('ticker').nunique()}")
    print(f"Quarter range   : {panel.index.get_level_values('quarter_end').min().date()} "
          f"→ {panel.index.get_level_values('quarter_end').max().date()}")
    coverage = panel.notna().mean().sort_values()
    print(f"\nFeature coverage (bottom 10):")
    print(coverage.head(10).to_string())
    fwd = panel['fwd_return'].dropna()
    print(f"\nForward return  : mean={fwd.mean():.2%}  std={fwd.std():.2%}  "
          f"skew={fwd.skew():.2f}  n={len(fwd)}")
