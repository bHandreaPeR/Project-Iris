"""
Full India + BSE pipeline v4 — Screener.in deep history (2015→2026).

Key improvements over v3:
  - Screener.in as primary data source: 13 quarters + 12 years annual per ticker
  - Annual snapshot rows extend panel from 7 quarters → 20+ time periods
  - Walk-forward folds: 7 → 40+ (quarterly + annual year-ends)
  - Model IC/ICIR expected to improve with more validation folds
"""

import sys, pickle, warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from data.universe import fetch_nifty_india
from ml.collector_v2 import build_panel, panel_summary
from ml.multi_horizon import MultiHorizonModel, TARGET_COLS
from ml.predict_report import ranked_watchlist, backtest_examples
from ml.reasoning import generate_reasoning

_OUT     = Path("ml_output")
PANEL_V4 = _OUT / "v2_panel_india_v4.parquet"
MODEL_V4 = _OUT / "v2_model_india_v4.pkl"
WATCH_V4 = _OUT / "watchlist_india_v4.csv"


# ---------------------------------------------------------------------------
# Step 1 — collect
# ---------------------------------------------------------------------------

def step_collect(tickers: list[str]) -> pd.DataFrame:
    print(f"\n[v4] Collecting {len(tickers)} tickers — Screener.in primary source …")
    if PANEL_V4.exists():
        print(f"[v4] Found cached panel {PANEL_V4} — loading …")
        return pd.read_parquet(PANEL_V4)

    panel = build_panel(
        tickers,
        cache_path=str(PANEL_V4),
        use_shareholding=False,   # NSE shareholding API returns 404; skip to save time
        use_insiders=False,
        use_news=False,
        verbose=True,
    )
    return panel


# ---------------------------------------------------------------------------
# Step 2 — train
# ---------------------------------------------------------------------------

def step_train(panel: pd.DataFrame) -> tuple[MultiHorizonModel, dict]:
    qs = panel.index.get_level_values('quarter_end').unique().sort_values()
    print(f"\n[v4] Training on {len(panel)} rows, "
          f"{panel.index.get_level_values('ticker').nunique()} tickers, "
          f"{len(qs)} time periods ({qs.min().date()} → {qs.max().date()}) …")
    panel_summary(panel)

    model   = MultiHorizonModel()
    metrics = model.train(panel)
    print(model.walk_forward_summary(metrics))
    return model, metrics


# ---------------------------------------------------------------------------
# Step 3 — feature importance audit
# ---------------------------------------------------------------------------

def step_importance_audit(model: MultiHorizonModel):
    print("\n" + "═" * 65)
    print("  FEATURE IMPORTANCE AUDIT (top 20 per horizon)")
    print("═" * 65)
    fi = model.top_feature_importance(n=20)

    for horizon in TARGET_COLS:
        sub = fi[fi['horizon'] == horizon].head(20)
        print(f"\n  ── {horizon} ──")
        for _, row in sub.iterrows():
            bar = "█" * int(row['importance'] * 200)
            print(f"    {row['feature']:<45} {row['importance']:.4f}  {bar}")

    consensus = (fi.groupby('feature')['importance']
                   .sum()
                   .sort_values(ascending=False)
                   .head(15))
    print("\n  ── Cross-horizon consensus top 15 ──")
    for feat, imp in consensus.items():
        print(f"    {feat:<45} {imp:.4f}")


# ---------------------------------------------------------------------------
# Step 4 — predict
# ---------------------------------------------------------------------------

def step_predict(model: MultiHorizonModel, panel: pd.DataFrame) -> pd.DataFrame:
    import yfinance as yf

    feat_cols = model.feat_cols
    latest_q  = panel.index.get_level_values('quarter_end').max()
    current   = panel.loc[panel.index.get_level_values('quarter_end') == latest_q]
    tickers   = current.index.get_level_values('ticker').tolist()
    print(f"\n[v4] Predicting {len(tickers)} tickers from period {latest_q.date()} …")

    price_map: dict[str, float] = {}
    for chunk in [tickers[i:i+200] for i in range(0, len(tickers), 200)]:
        try:
            pdata = yf.download(chunk, period='2d', interval='1d',
                                auto_adjust=True, progress=False)['Close']
            last  = pdata.iloc[-1] if len(pdata) else pdata
            for tkr in chunk:
                try:
                    v = float(last[tkr])
                    if v > 0:
                        price_map[tkr] = v
                except Exception:
                    pass
        except Exception:
            pass

    ticker_preds: dict[str, dict] = {}
    feats_map:   dict[str, pd.Series] = {}
    for tkr, grp in current.groupby(level='ticker'):
        try:
            feats = grp.iloc[-1][feat_cols]
            ticker_preds[tkr] = model.predict_ticker(feats)
            feats_map[tkr]    = feats
        except Exception:
            pass

    watchlist = ranked_watchlist(ticker_preds, price_map, horizon='fwd_3m')
    watchlist.to_csv(WATCH_V4)

    reasons = []
    for _, row in watchlist.iterrows():
        tkr = row['ticker']
        sig = row.get('signal', '')
        if tkr in feats_map:
            r = generate_reasoning(feats_map[tkr], sig, n_lines=3)
            reasons.append(" | ".join(r))
        else:
            reasons.append("")
    watchlist['reasoning'] = reasons

    print(f"\n[v4] Watchlist saved → {WATCH_V4}")
    return watchlist


# ---------------------------------------------------------------------------
# Data depth summary
# ---------------------------------------------------------------------------

def data_depth_summary(panel: pd.DataFrame):
    qs = panel.index.get_level_values('quarter_end')
    rows_per_ticker = panel.groupby(level='ticker').size()
    print("\n" + "═" * 55)
    print("  DATA DEPTH SUMMARY")
    print("═" * 55)
    print(f"  Total rows       : {len(panel):,}")
    print(f"  Time periods     : {qs.unique().nunique()} "
          f"({qs.min().date()} → {qs.max().date()})")
    print(f"  Rows per ticker  : "
          f"min={rows_per_ticker.min()} "
          f"median={rows_per_ticker.median():.0f} "
          f"max={rows_per_ticker.max()}")
    print(f"  Tickers ≥10 rows : "
          f"{(rows_per_ticker >= 10).sum()} / {len(rows_per_ticker)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Universe
    all_tickers = fetch_nifty_india()
    ns = sum(1 for t in all_tickers if t.endswith('.NS'))
    bo = sum(1 for t in all_tickers if t.endswith('.BO'))
    print(f"[v4] India universe: {len(all_tickers)} tickers ({ns} NSE + {bo} BSE-only)")

    # Collect
    panel = step_collect(all_tickers)
    if panel.empty:
        sys.exit("[v4] No data collected.")

    tickers_got = panel.index.get_level_values('ticker').nunique()
    ns_got = sum(1 for t in panel.index.get_level_values('ticker').unique() if t.endswith('.NS'))
    bo_got = sum(1 for t in panel.index.get_level_values('ticker').unique() if t.endswith('.BO'))
    print(f"[v4] Panel: {len(panel)} rows, {tickers_got} tickers ({ns_got} NSE, {bo_got} BSE)")

    data_depth_summary(panel)

    # Train
    model, metrics = step_train(panel)

    with open(MODEL_V4, 'wb') as f:
        pickle.dump(model, f)
    print(f"[v4] Model saved → {MODEL_V4}")

    # Feature audit
    step_importance_audit(model)

    # Backtest examples
    print(backtest_examples(model.oof_df, panel, n_examples=8))

    # Predict
    watchlist = step_predict(model, panel)

    # Top 50
    print("\n" + "═" * 90)
    print("  TOP 50 INDIA PICKS — 3-month horizon")
    print("═" * 90)
    cols = ['ticker', 'pred_return', 'signal', 'confidence', 'price_target', 'reasoning']
    show_cols = [c for c in cols if c in watchlist.columns]
    pd.set_option('display.max_colwidth', 80)
    pd.set_option('display.width', 200)
    print(watchlist[show_cols].head(50).to_string())

    print("\n[v4] Done.")
