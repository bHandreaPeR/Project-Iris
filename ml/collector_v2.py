"""
Panel builder v2 — assembles the full (ticker × quarter) feature matrix.

For each ticker:
  1. Fetch quarterly + annual financial statements (yfinance)
  2. Fetch shareholding pattern (NSE API for India; skipped for US)
  3. Fetch insider transactions (SEC EDGAR Form 4 for US; skipped for India)
  4. Fetch 3-tier news sentiment (GDELT + FinBERT) [optional, slow]
  5. Compute all v2 features for each quarter snapshot
  6. Compute 3 forward return targets: 1m, 3m, 12m (in calendar days)

Look-ahead bias prevention:
  — Financial data is clipped to columns ≤ quarter-end date before feature
    computation. A filing-lag offset is applied so the entry date reflects
    when the data was actually available to an investor.
  — Forward returns are measured from (entry_date) to (entry_date + horizon).
  — News/shareholding use data as of the entry date, not look-ahead.

NaN policy: no imputation. XGBoost handles NaN natively.
"""

import math
import time
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

from data.financials import fetch_statements
from data.financials_india import fetch_statements_india
from data.nse_shareholding import shareholding_features
from data.sec_insiders import insider_summary
from data.news_pipeline import fetch_news_sentiment
from ml.features_v2 import compute_all_v2, FEATURE_NAMES

_CACHE_DIR = Path("ml_output")

FILING_LAG_US    = 45   # days after quarter-end when 10-Q is typically filed
FILING_LAG_INDIA = 60   # days after quarter-end for NSE quarterly results

HORIZONS = {
    'fwd_1m':  30,
    'fwd_3m':  91,
    'fwd_12m': 365,
}


def _lag(ticker: str) -> int:
    return FILING_LAG_INDIA if ticker.endswith(('.NS', '.BO')) else FILING_LAG_US


def _clip_stmts(stmts: dict, as_of: pd.Timestamp) -> dict:
    clipped = dict(stmts)
    for key in ('income', 'balance', 'cashflow',
                'annual_income', 'annual_balance', 'annual_cashflow'):
        df = stmts.get(key, pd.DataFrame())
        if df.empty:
            clipped[key] = df
        else:
            cols = [c for c in df.columns if pd.Timestamp(c) <= as_of]
            clipped[key] = df[cols] if cols else pd.DataFrame()
    return clipped


def _fwd_return(price_hist: pd.DataFrame,
                entry: pd.Timestamp, exit_: pd.Timestamp) -> float:
    if price_hist.empty:
        return float('nan')
    ph = price_hist['Close'].sort_index()
    # Strip timezone so tz-aware price index can compare with tz-naive timestamps
    if ph.index.tz is not None:
        ph.index = ph.index.tz_localize(None)
    entry_n = entry.tz_localize(None) if entry.tzinfo is not None else entry
    exit_n  = exit_.tz_localize(None) if exit_.tzinfo is not None else exit_
    try:
        p0 = float(ph.loc[:entry_n].iloc[-1])
        p1 = float(ph.loc[:exit_n].iloc[-1])
        return (p1 - p0) / abs(p0) if p0 != 0 else float('nan')
    except (IndexError, KeyError):
        return float('nan')


def build_ticker_panel(ticker: str,
                       stmts: dict,
                       use_shareholding: bool = True,
                       use_insiders: bool = True,
                       use_news: bool = False) -> pd.DataFrame:
    """
    Build one row per quarter-end for a single ticker.
    Returns DataFrame indexed by (ticker, quarter_end).
    """
    inc = stmts.get('income', pd.DataFrame())
    if inc.empty:
        return pd.DataFrame()

    lag = _lag(ticker)
    quarter_ends = sorted([pd.Timestamp(c) for c in inc.columns])

    # Pre-fetch optional data sources (fetched once per ticker, not per quarter)
    sh_data = None
    if use_shareholding:
        try:
            sh_data = shareholding_features(ticker)
        except Exception:
            pass

    insider_data = None
    if use_insiders and not ticker.endswith(('.NS', '.BO')):
        try:
            insider_data = insider_summary(ticker, lookback_days=180)
        except Exception:
            pass

    company_name = stmts.get('info', {}).get('longName', '')
    sector       = stmts.get('info', {}).get('sector', '')

    rows = []
    quarter_end_set: set[pd.Timestamp] = set()

    def _build_row(qe: pd.Timestamp) -> dict | None:
        entry_date = qe + pd.DateOffset(days=lag)
        snap = _clip_stmts(stmts, qe)

        news_data = None
        if use_news:
            try:
                news_data = fetch_news_sentiment(
                    ticker, company_name, sector, timespan='30d'
                )
            except Exception:
                pass

        try:
            feats = compute_all_v2(snap, entry_date,
                                   shareholding=sh_data,
                                   insider=insider_data,
                                   news=news_data)
        except Exception:
            return None

        n_valid = sum(1 for v in feats.values()
                      if not math.isnan(float(v) if isinstance(v, (int, float)) else float('nan')))
        if n_valid < 10:
            return None

        row = {'ticker': ticker, 'quarter_end': qe, 'entry_date': entry_date}
        row.update(feats)
        for col, days in HORIZONS.items():
            exit_date = entry_date + pd.DateOffset(days=days)
            row[col] = _fwd_return(stmts.get('price_hist', pd.DataFrame()),
                                   entry_date, exit_date)
        return row

    # Quarterly rows (from income stmt — typically 13 quarters from screener.in)
    for qe in quarter_ends:
        r = _build_row(qe)
        if r is not None:
            rows.append(r)
            quarter_end_set.add(qe)

    # Annual rows — extend history back to 2015 using annual statements
    ann_inc = stmts.get('annual_income', pd.DataFrame())
    if not ann_inc.empty:
        ann_dates = sorted([pd.Timestamp(c) for c in ann_inc.columns])
        for ann_qe in ann_dates:
            if ann_qe in quarter_end_set:
                continue  # already have a quarterly row for this period
            r = _build_row(ann_qe)
            if r is not None:
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index(['ticker', 'quarter_end'])
    return df


def build_panel(tickers: list[str],
                cache_path: str | None = None,
                use_shareholding: bool = True,
                use_insiders: bool = True,
                use_news: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    """
    Build the full cross-sectional panel for a list of tickers.

    Args:
        tickers          : yfinance ticker strings
        cache_path       : parquet path to cache/reload panel
        use_shareholding : fetch NSE shareholding (India only)
        use_insiders     : fetch SEC Form 4 (US only)
        use_news         : fetch GDELT + FinBERT sentiment (slow, optional)
        verbose          : print progress
    """
    import os
    if cache_path and os.path.exists(cache_path):
        if verbose:
            print(f"[collector_v2] Loading cached panel from {cache_path}")
        return pd.read_parquet(cache_path)

    all_panels = []
    n = len(tickers)

    for i, tkr in enumerate(tickers, 1):
        if verbose:
            print(f"[collector_v2] {i:3d}/{n}  {tkr:<20}", end="  ", flush=True)
        try:
            stmts = (fetch_statements_india(tkr)
                     if tkr.endswith(('.NS', '.BO'))
                     else fetch_statements(tkr))
            ticker_df = build_ticker_panel(
                tkr, stmts,
                use_shareholding=use_shareholding,
                use_insiders=use_insiders,
                use_news=use_news,
            )
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
        time.sleep(0.2)

    if not all_panels:
        print("[collector_v2] No data collected.")
        return pd.DataFrame()

    panel = pd.concat(all_panels).sort_index()

    if cache_path:
        panel.to_parquet(cache_path)
        if verbose:
            print(f"\n[collector_v2] Saved {len(panel)} rows → {cache_path}")

    return panel


def panel_summary(panel: pd.DataFrame) -> None:
    tickers  = panel.index.get_level_values('ticker').nunique()
    quarters = panel.index.get_level_values('quarter_end')
    feat_cols = [c for c in panel.columns if c not in list(HORIZONS.keys()) + ['entry_date']]
    coverage  = panel[feat_cols].notna().mean().sort_values()

    print(f"\n{'═'*55}")
    print(f"  PANEL SUMMARY")
    print(f"{'═'*55}")
    print(f"  Rows           : {len(panel):,}")
    print(f"  Tickers        : {tickers}")
    print(f"  Quarter range  : {quarters.min().date()} → {quarters.max().date()}")
    print(f"  Features       : {len(feat_cols)}")
    print(f"  Avg coverage   : {coverage.mean():.1%}")
    print(f"\n  Bottom-10 feature coverage:")
    for f, c in coverage.head(10).items():
        print(f"    {f:<45} {c:.1%}")
    print()
    for col in HORIZONS:
        s = panel[col].dropna()
        if len(s):
            print(f"  {col}: mean={s.mean():+.2%}  std={s.std():.2%}  n={len(s):,}")
    print()
