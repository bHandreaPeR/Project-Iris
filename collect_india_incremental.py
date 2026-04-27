"""
Incremental India panel builder.
Collects only tickers not already in v2_panel_india.parquet,
then merges and saves v2_panel_india_full.parquet.
"""
import sys
import pickle
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.universe import fetch_nifty_india
from ml.collector_v2 import build_panel, panel_summary
from ml.multi_horizon import MultiHorizonModel, TARGET_COLS
from ml.predict_report import ranked_watchlist

_OUT = Path("ml_output")
EXISTING_PANEL = _OUT / "v2_panel_india.parquet"
NEW_PANEL      = _OUT / "v2_panel_india_new.parquet"
FULL_PANEL     = _OUT / "v2_panel_india_full.parquet"
MODEL_PATH     = _OUT / "v2_model_india_full.pkl"


def main():
    # 1. Load full universe
    print("[incremental] Resolving India universe …")
    all_tickers = fetch_nifty_india()
    ns = [t for t in all_tickers if t.endswith('.NS')]
    bo = [t for t in all_tickers if t.endswith('.BO')]
    print(f"[incremental] Universe: {len(all_tickers)} ({len(ns)} NSE + {len(bo)} BSE-only)")

    # 2. Find which tickers still need collection
    existing_tickers: set[str] = set()
    if EXISTING_PANEL.exists():
        ex = pd.read_parquet(EXISTING_PANEL)
        existing_tickers = set(ex.index.get_level_values('ticker').unique())
        print(f"[incremental] Existing panel: {len(existing_tickers)} tickers, {len(ex)} rows")

    new_tickers = [t for t in all_tickers if t not in existing_tickers]
    print(f"[incremental] New tickers to collect: {len(new_tickers)}")

    # 3. Collect new tickers
    if new_tickers:
        if NEW_PANEL.exists():
            NEW_PANEL.unlink()  # force fresh collect of new tickers
        new_panel = build_panel(
            new_tickers,
            cache_path=str(NEW_PANEL),
            use_shareholding=True,
            use_insiders=False,
            use_news=False,
            verbose=True,
        )
        print(f"[incremental] New panel: {len(new_panel)} rows")
    else:
        new_panel = pd.DataFrame()
        print("[incremental] No new tickers to collect.")

    # 4. Merge
    parts = []
    if EXISTING_PANEL.exists():
        parts.append(pd.read_parquet(EXISTING_PANEL))
    if not new_panel.empty:
        parts.append(new_panel)

    if not parts:
        print("[incremental] ERROR: no data at all.")
        sys.exit(1)

    full_panel = pd.concat(parts).sort_index()
    # Drop duplicate (ticker, quarter_end) keeping last
    full_panel = full_panel[~full_panel.index.duplicated(keep='last')]
    full_panel.to_parquet(FULL_PANEL)
    tickers_in = full_panel.index.get_level_values('ticker').nunique()
    ns_in = sum(1 for t in full_panel.index.get_level_values('ticker').unique() if t.endswith('.NS'))
    bo_in = sum(1 for t in full_panel.index.get_level_values('ticker').unique() if t.endswith('.BO'))
    print(f"\n[incremental] Full panel: {len(full_panel)} rows, {tickers_in} tickers ({ns_in} NSE, {bo_in} BSE)")
    print(f"[incremental] Saved → {FULL_PANEL}")

    # 5. Train
    print("\n[incremental] Training multi-horizon model on full India panel …")
    panel_summary(full_panel)
    model = MultiHorizonModel()
    metrics = model.train(full_panel)
    print(model.walk_forward_summary(metrics))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"[incremental] Model saved → {MODEL_PATH}")

    # 6. Top picks
    feat_cols = model.feat_cols
    latest_q = full_panel.index.get_level_values('quarter_end').max()
    current  = full_panel.loc[full_panel.index.get_level_values('quarter_end') == latest_q]
    print(f"\n[incremental] Generating predictions for {len(current)} tickers in Q {latest_q.date()} …")

    import yfinance as yf
    all_tkrs = current.index.get_level_values('ticker').tolist()
    price_map: dict[str, float] = {}
    for chunk in [all_tkrs[i:i+200] for i in range(0, len(all_tkrs), 200)]:
        try:
            pdata = yf.download(chunk, period='2d', interval='1d',
                                auto_adjust=True, progress=False)['Close']
            if hasattr(pdata, 'iloc'):
                last = pdata.iloc[-1] if len(pdata) > 0 else pdata
                for tkr in chunk:
                    try:
                        v = float(last[tkr]) if tkr in last.index else float('nan')
                        if v > 0:
                            price_map[tkr] = v
                    except Exception:
                        pass
        except Exception:
            pass

    ticker_preds: dict[str, dict] = {}
    for tkr, grp in current.groupby(level='ticker'):
        try:
            feats = grp.iloc[-1][feat_cols]
            ticker_preds[tkr] = model.predict_ticker(feats)
        except Exception:
            pass

    watchlist = ranked_watchlist(ticker_preds, price_map, horizon='fwd_3m')
    wl_path = _OUT / "watchlist_india_full.csv"
    watchlist.to_csv(wl_path)
    print(f"\n[incremental] Watchlist saved → {wl_path}")
    print("\nTop 30 India picks (3-month horizon):")
    print(watchlist.head(30).to_string())


if __name__ == "__main__":
    main()
