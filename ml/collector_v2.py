"""
Panel builder v2 — assembles the full (ticker × signal_date) feature matrix.

For each ticker:
  1. Fetch quarterly + annual financial statements (Screener.in + yfinance)
  2. Fetch shareholding pattern (NSE API for India; skipped for US)
  3. Fetch insider transactions (SEC EDGAR Form 4 for US; skipped for India)
  4. Fetch corporate actions (dividends, splits, buyback, rights, bonus)
  5. Fetch NSE F&O data (PCR, OI, IV) — India F&O-eligible stocks only
  6. Fetch 3-tier news sentiment (GDELT + FinBERT) [optional, slow]
  7. Compute all v2 features for each quarter snapshot
  8. Compute 3 forward return targets: 1m, 3m, 12m (in calendar days)

Panel index: (ticker, signal_date)
  signal_date = actual date the filing was detected / became available.
  For historical rows: quarter_end + filing_lag (best estimate).
  For the latest row when lag is still in the future: today (live detection).
  quarter_end is kept as a regular metadata column.

Look-ahead bias prevention:
  — Financial data clipped to columns ≤ quarter_end before feature computation.
  — Forward returns measured from signal_date to signal_date + horizon.
  — Price/momentum features clipped to as_of = signal_date.
  — News/shareholding use data as of signal_date, not look-ahead.

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

# Metadata columns — excluded from feature set
_META_COLS = ['quarter_end', 'data_lag_days']


def _lag(ticker: str) -> int:
    return FILING_LAG_INDIA if ticker.endswith(('.NS', '.BO')) else FILING_LAG_US


def validate_panel_schema(panel: pd.DataFrame) -> None:
    """Raise a clear error if an old-format panel (quarter_end in index) is passed."""
    index_names = list(panel.index.names)
    if 'quarter_end' in index_names:
        raise ValueError(
            "Old-format panel detected: 'quarter_end' is in the index. "
            "v5 uses (ticker, signal_date) as the index with quarter_end as a column. "
            "Delete cached parquets and rebuild with run_full_india_v5.py."
        )
    if 'signal_date' not in index_names:
        raise ValueError(
            f"Panel index must contain 'signal_date'. Got: {index_names}. "
            "Rebuild with run_full_india_v5.py."
        )


def _clip_stmts(stmts: dict, as_of: pd.Timestamp) -> dict:
    """Clip all financial statement DataFrames to columns ≤ as_of (look-ahead guard)."""
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
                       use_news: bool = False,
                       use_corp_actions: bool = True,
                       use_fno: bool = True) -> pd.DataFrame:
    """
    Build one row per filing event for a single ticker.
    Returns DataFrame indexed by (ticker, signal_date).
    signal_date = date the filing was (or will be) publicly available.
    """
    inc = stmts.get('income', pd.DataFrame())
    if inc.empty:
        return pd.DataFrame()

    lag = _lag(ticker)
    quarter_ends = sorted([pd.Timestamp(c) for c in inc.columns])

    # Pre-fetch optional data sources (once per ticker, reused across all rows)
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

    corp_data = None
    if use_corp_actions:
        try:
            from data.corporate_actions import fetch_corporate_actions
            corp_data = fetch_corporate_actions(ticker, stmts)
        except Exception:
            pass

    fno_data = None
    if use_fno and ticker.endswith(('.NS', '.BO')):
        try:
            from data.nse_fno import fetch_fno_snapshot
            fno_data = fetch_fno_snapshot(ticker, stmts)
        except Exception:
            pass

    company_name = stmts.get('info', {}).get('longName', '')
    sector       = stmts.get('info', {}).get('sector', '')

    rows = []
    quarter_end_set: set[pd.Timestamp] = set()

    def _build_row(qe: pd.Timestamp, is_latest: bool = False) -> dict | None:
        lag_date = qe + pd.DateOffset(days=lag)
        today = pd.Timestamp.now().normalize()

        # signal_date = actual date data became available.
        # For the latest row: if lag hasn't elapsed yet (filing just happened),
        # use today so the signal age reflects live detection, not a future estimate.
        # For historical rows: always use the lag-based date for accurate
        # forward-return computation during training.
        if is_latest and lag_date > today:
            signal_date = today
        else:
            signal_date = lag_date

        data_lag_days = (signal_date - qe).days
        snap = _clip_stmts(stmts, qe)

        news_data = None
        if use_news:
            try:
                from data.news_pipeline import fetch_news_sentiment_windowed
                # Only compute live-only pulse features for the latest row
                news_data = fetch_news_sentiment_windowed(
                    ticker, company_name, sector,
                    live_only=is_latest
                )
            except Exception:
                try:
                    news_data = fetch_news_sentiment(
                        ticker, company_name, sector, timespan='30d'
                    )
                except Exception:
                    pass

        # F&O and corp actions: pass as-of signal_date for look-ahead safety
        # (only meaningful for latest rows; historical rows get static data)
        ca_data  = corp_data  if is_latest else None
        fno_snap = fno_data   if is_latest else None

        try:
            feats = compute_all_v2(snap, signal_date,
                                   shareholding=sh_data,
                                   insider=insider_data,
                                   news=news_data,
                                   corp_actions=ca_data,
                                   fno=fno_snap)
        except Exception:
            return None

        n_valid = sum(1 for v in feats.values()
                      if not math.isnan(float(v) if isinstance(v, (int, float)) else float('nan')))
        if n_valid < 10:
            return None

        row = {
            'ticker':        ticker,
            'signal_date':   signal_date,   # → becomes index key
            'quarter_end':   qe,            # metadata column
            'data_lag_days': data_lag_days, # audit trail
        }
        row.update(feats)
        for col, days in HORIZONS.items():
            exit_date = signal_date + pd.DateOffset(days=days)
            row[col] = _fwd_return(stmts.get('price_hist', pd.DataFrame()),
                                   signal_date, exit_date)
        return row

    # Quarterly rows (from income stmt)
    latest_qe = quarter_ends[-1] if quarter_ends else None
    for qe in quarter_ends:
        r = _build_row(qe, is_latest=(qe == latest_qe))
        if r is not None:
            rows.append(r)
            quarter_end_set.add(qe)

    # Annual rows — extend history using annual statements (Screener.in: 11 years)
    ann_inc = stmts.get('annual_income', pd.DataFrame())
    if not ann_inc.empty:
        ann_dates = sorted([pd.Timestamp(c) for c in ann_inc.columns])
        latest_ann = ann_dates[-1] if ann_dates else None
        for ann_qe in ann_dates:
            if ann_qe in quarter_end_set:
                continue  # already have a quarterly row for this period
            r = _build_row(ann_qe, is_latest=(ann_qe == latest_ann))
            if r is not None:
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index(['ticker', 'signal_date'])
    return df


def build_panel(tickers: list[str],
                cache_path: str | None = None,
                use_shareholding: bool = True,
                use_insiders: bool = True,
                use_news: bool = False,
                use_corp_actions: bool = True,
                use_fno: bool = True,
                verbose: bool = True) -> pd.DataFrame:
    """
    Build the full cross-sectional panel for a list of tickers.

    Args:
        tickers          : yfinance ticker strings
        cache_path       : parquet path to cache/reload panel
        use_shareholding : fetch NSE shareholding (India only)
        use_insiders     : fetch SEC Form 4 (US only)
        use_news         : fetch GDELT + FinBERT sentiment (slow, optional)
        use_corp_actions : fetch corporate actions (dividends, buyback, etc.)
        use_fno          : fetch NSE F&O derivatives data (India only)
        verbose          : print progress
    """
    import os
    if cache_path and os.path.exists(cache_path):
        panel = pd.read_parquet(cache_path)
        try:
            validate_panel_schema(panel)
        except ValueError as e:
            print(f"[collector_v2] WARNING: {e}")
            print("[collector_v2] Ignoring cache and rebuilding...")
            os.remove(cache_path)
            panel = None
        else:
            if verbose:
                print(f"[collector_v2] Loading cached panel from {cache_path}")
            return panel

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
                use_corp_actions=use_corp_actions,
                use_fno=use_fno,
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
    validate_panel_schema(panel)
    tickers = panel.index.get_level_values('ticker').nunique()
    sdates  = panel.index.get_level_values('signal_date')
    qends   = panel['quarter_end'] if 'quarter_end' in panel.columns else pd.Series()

    meta_cols = _META_COLS + list(HORIZONS.keys())
    feat_cols = [c for c in panel.columns if c not in meta_cols]
    coverage  = panel[feat_cols].notna().mean().sort_values()

    print(f"\n{'═'*55}")
    print(f"  PANEL SUMMARY (v5 — event-driven)")
    print(f"{'═'*55}")
    print(f"  Rows            : {len(panel):,}")
    print(f"  Tickers         : {tickers}")
    print(f"  Signal date range: {sdates.min().date()} → {sdates.max().date()}")
    if not qends.empty:
        print(f"  Quarter range   : {qends.min().date()} → {qends.max().date()}")
    print(f"  Features        : {len(feat_cols)}")
    print(f"  Avg coverage    : {coverage.mean():.1%}")
    print(f"\n  Bottom-10 feature coverage:")
    for f, c in coverage.head(10).items():
        print(f"    {f:<45} {c:.1%}")
    print()
    for col in HORIZONS:
        s = panel[col].dropna()
        if len(s):
            print(f"  {col}: mean={s.mean():+.2%}  std={s.std():.2%}  n={len(s):,}")
    print()
