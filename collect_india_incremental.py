"""
Incremental India panel updater.

Two modes:
  python collect_india_incremental.py           → refresh stale tickers + add new ones
  python collect_india_incremental.py --new-only → only collect tickers missing from panel

Refresh logic:
  For every ticker already in the panel, fetch its latest income statement
  (via yfinance).  If yfinance now reports a newer quarter_end than the one
  stored in the panel, the ticker is marked "stale" and re-collected.
  This means: when Bandhan files today, tomorrow's run detects the new quarter
  and produces a fresh signal with entry_date = today (age = 0 days).
"""
import sys
import argparse
import pickle
import warnings
import pandas as pd
import yfinance as yf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

from ml.collector_v2 import build_panel, build_ticker_panel, panel_summary
from ml.multi_horizon import MultiHorizonModel

_OUT         = Path("ml_output")
PANEL_V3     = _OUT / "v2_panel_india_v3.parquet"   # canonical panel
MODEL_V3     = _OUT / "v2_model_india_v3.pkl"
STALE_CACHE  = _OUT / "stale_tickers_last_run.csv"

REFRESH_CHECK_WORKERS = 20   # parallel yfinance calls for staleness check
COLLECT_WORKERS       = 1    # sequential for main collect (rate-limit safe)


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

def _latest_yf_quarter(ticker: str) -> pd.Timestamp | None:
    """Return the latest quarter_end yfinance currently has for this ticker."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t   = yf.Ticker(ticker)
            inc = t.quarterly_income_stmt
            if inc is None or inc.empty:
                return None
            cols = [pd.Timestamp(c) for c in inc.columns]
            return max(cols)
    except Exception:
        return None


def find_stale_tickers(panel: pd.DataFrame, max_workers: int = REFRESH_CHECK_WORKERS,
                       verbose: bool = True) -> list[str]:
    """
    Return tickers whose latest panel quarter_end is older than what yfinance
    currently reports — i.e. the company has filed new results since last run.
    """
    # Latest quarter per ticker — use quarter_end column (v5) or index level (legacy)
    reset = panel.reset_index()
    if 'quarter_end' in reset.columns:
        panel_latest = reset.groupby("ticker")["quarter_end"].max().to_dict()
    else:
        # Legacy v3/v4 format: quarter_end is an index level
        panel_latest = (
            panel.index.to_frame()
                 .reset_index(drop=True)
                 .groupby("ticker")["quarter_end"]
                 .max()
                 .to_dict()
        )
    all_tickers = list(panel_latest.keys())
    if verbose:
        print(f"[staleness] Checking {len(all_tickers)} tickers for new filings …")

    stale: list[str] = []
    checked = 0

    def _check(tkr: str) -> tuple[str, bool]:
        yf_latest = _latest_yf_quarter(tkr)
        if yf_latest is None:
            return tkr, False
        return tkr, yf_latest > panel_latest[tkr]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_check, t): t for t in all_tickers}
        for fut in as_completed(futures):
            tkr, is_stale = fut.result()
            checked += 1
            if is_stale:
                stale.append(tkr)
            if verbose and checked % 200 == 0:
                print(f"  checked {checked}/{len(all_tickers)} … {len(stale)} stale so far")

    if verbose:
        print(f"[staleness] Done. {len(stale)} stale tickers found.")
    return stale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-only",   action="store_true",
                        help="Only collect tickers missing from panel; skip refresh check")
    parser.add_argument("--panel",      default=str(PANEL_V3),
                        help="Path to existing panel parquet")
    parser.add_argument("--model",      default=str(MODEL_V3),
                        help="Path to existing model pkl")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Update panel but skip model re-train (fast, for watchlist refresh)")
    args = parser.parse_args()

    panel_path = Path(args.panel)
    model_path = Path(args.model)

    # ------------------------------------------------------------------
    # 1. Load existing panel
    # ------------------------------------------------------------------
    if not panel_path.exists():
        print(f"[incremental] Panel not found at {panel_path}. Run run_full_india_v3.py first.")
        sys.exit(1)

    panel = pd.read_parquet(panel_path)
    existing_tickers: set[str] = set(panel.index.get_level_values("ticker").unique())
    print(f"[incremental] Loaded panel: {len(panel)} rows, {len(existing_tickers)} tickers")

    # ------------------------------------------------------------------
    # 2. Load universe (from panel itself — no extra fetch needed)
    # ------------------------------------------------------------------
    all_tickers = sorted(existing_tickers)

    # ------------------------------------------------------------------
    # 3. Find tickers to refresh (stale) and tickers to add (new)
    # ------------------------------------------------------------------
    tickers_to_recollect: list[str] = []

    if not args.new_only:
        stale = find_stale_tickers(panel, verbose=True)
        tickers_to_recollect.extend(stale)
        # Cache stale list for debugging
        pd.DataFrame({"ticker": stale}).to_csv(STALE_CACHE, index=False)
        print(f"[incremental] {len(stale)} tickers to refresh: {stale[:10]}{'…' if len(stale)>10 else ''}")
    else:
        print("[incremental] --new-only: skipping staleness check")

    if not tickers_to_recollect:
        print("[incremental] Nothing to update. Panel is current.")
        # Still regenerate watchlist HTML from existing model
        _regenerate_html(panel, model_path)
        return

    # ------------------------------------------------------------------
    # 4. Re-collect stale tickers
    # ------------------------------------------------------------------
    print(f"\n[incremental] Re-collecting {len(tickers_to_recollect)} tickers …")

    # Build fresh rows for stale tickers
    fresh_panel = build_panel(
        tickers_to_recollect,
        cache_path=None,          # no intermediate cache — small batch
        use_shareholding=True,
        use_insiders=False,
        use_news=False,
        verbose=True,
    )
    print(f"[incremental] Fresh rows collected: {len(fresh_panel)}")

    # ------------------------------------------------------------------
    # 5. Merge: drop stale rows, append fresh rows
    # ------------------------------------------------------------------
    stale_set = set(tickers_to_recollect)
    panel_kept = panel[~panel.index.get_level_values("ticker").isin(stale_set)]
    updated_panel = pd.concat([panel_kept, fresh_panel]).sort_index()
    updated_panel = updated_panel[~updated_panel.index.duplicated(keep="last")]

    updated_panel.to_parquet(panel_path)
    n_tickers = updated_panel.index.get_level_values("ticker").nunique()
    print(f"\n[incremental] Updated panel: {len(updated_panel)} rows, {n_tickers} tickers")
    print(f"[incremental] Saved → {panel_path}")

    # ------------------------------------------------------------------
    # 6. Re-train model (unless --no-retrain)
    # ------------------------------------------------------------------
    if not args.no_retrain:
        print("\n[incremental] Re-training model …")
        panel_summary(updated_panel)
        model = MultiHorizonModel()
        metrics = model.train(updated_panel)
        print(model.walk_forward_summary(metrics))
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[incremental] Model saved → {model_path}")
    else:
        print("[incremental] --no-retrain: loading existing model")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # ------------------------------------------------------------------
    # 7. Regenerate watchlist HTML
    # ------------------------------------------------------------------
    _regenerate_html(updated_panel, model_path)


def _regenerate_html(panel: pd.DataFrame, model_path: Path):
    """Call the watchlist generator as a subprocess so it picks up the updated panel/model."""
    import subprocess
    print("\n[incremental] Regenerating watchlist HTML …")
    result = subprocess.run(
        [sys.executable, "generate_watchlist_html.py"],
        capture_output=False
    )
    if result.returncode != 0:
        print("[incremental] Warning: watchlist generation failed")
    else:
        print("[incremental] Watchlist regenerated → ml_output/watchlist_live.html")


if __name__ == "__main__":
    main()
