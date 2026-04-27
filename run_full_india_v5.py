"""
Project Iris v5 — Full India panel rebuild and model training.

Changes from v4:
  - Panel indexed by (ticker, signal_date) — event-driven, not quarter-batched
  - Monthly walk-forward folds (not quarterly)
  - Corporate actions features (dividends, buyback, rights, bonus)
  - NSE F&O derivatives features (PCR, OI, IV, max pain)
  - Enhanced news sentiment (7d pulse, vol spike, event flag)
  - XGBoost + LightGBM ensemble
  - Market cap bucket feature
  - Zero-importance feature pruning

Usage:
  python run_full_india_v5.py                        # full rebuild
  python run_full_india_v5.py --no-fno               # skip F&O (faster)
  python run_full_india_v5.py --no-news              # skip news sentiment
  python run_full_india_v5.py --no-corp-actions      # skip corporate actions
  python run_full_india_v5.py --tune                 # Optuna hyperparameter search
  python run_full_india_v5.py --tickers 50           # quick test on first 50 tickers
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

_OUT = Path("ml_output")
_OUT.mkdir(exist_ok=True)

PANEL_PATH = _OUT / "v2_panel_india_v5.parquet"
MODEL_PATH = _OUT / "v2_model_india_v5.pkl"
CSV_PATH   = _OUT / "watchlist_india_v5.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Project Iris v5 — India full rebuild")
    p.add_argument("--no-fno",          action="store_true", help="Skip NSE F&O data")
    p.add_argument("--no-news",         action="store_true", help="Skip news sentiment")
    p.add_argument("--no-corp-actions", action="store_true", help="Skip corporate actions")
    p.add_argument("--tune",            action="store_true", help="Optuna hyperparameter tuning")
    p.add_argument("--tickers",         type=int, default=0, help="Limit to first N tickers (0 = all)")
    p.add_argument("--force",           action="store_true", help="Delete cached panel and rebuild")
    return p.parse_args()


def step_universe(args) -> list[str]:
    from data.universe import fetch_nse_all, fetch_bse_stocks
    print("[v5] Resolving India universe …")
    nse = fetch_nse_all()
    bse_tickers, _ = fetch_bse_stocks()
    all_tickers = nse + [t for t in bse_tickers if t not in set(nse)]
    if args.tickers > 0:
        all_tickers = all_tickers[:args.tickers]
        print(f"[v5] (--tickers {args.tickers}: using first {len(all_tickers)} tickers)")
    print(f"[v5] Universe: {len(all_tickers)} tickers")
    return all_tickers


def step_collect(tickers: list[str], args) -> pd.DataFrame:
    from ml.collector_v2 import build_panel, panel_summary

    if args.force and PANEL_PATH.exists():
        print(f"[v5] --force: deleting {PANEL_PATH}")
        PANEL_PATH.unlink()

    print("[v5] Collecting data …")
    panel = build_panel(
        tickers,
        cache_path     = str(PANEL_PATH),
        use_shareholding  = True,
        use_insiders      = False,   # India only — no SEC EDGAR
        use_news          = not args.no_news,
        use_corp_actions  = not args.no_corp_actions,
        use_fno           = not args.no_fno,
        verbose           = True,
    )
    panel_summary(panel)
    return panel


def step_train(panel: pd.DataFrame, args) -> tuple:
    from ml.multi_horizon import MultiHorizonModel

    xgb_params = None
    if args.tune:
        xgb_params = step_tune(panel)

    print("\n[v5] Training multi-horizon model …")
    model   = MultiHorizonModel()
    metrics = model.train(panel, xgb_params=xgb_params)
    print(model.walk_forward_summary(metrics))

    # Prune zero-importance features and retrain final models
    model.prune_zero_importance_features()
    print("[v5] Retraining final models on pruned feature set …")
    for target, hm in model.models.items():
        hm.fit_final(panel)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[v5] Model saved → {MODEL_PATH}")

    return model, metrics


def step_tune(panel: pd.DataFrame) -> dict:
    """Optuna hyperparameter search over last 6 monthly folds."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[v5] optuna not installed — skipping tuning (pip install optuna)")
        return {}

    from ml.multi_horizon import HorizonModel, MIN_TRAIN, MIN_TRAIN_MONTHS, MIN_TEST_SIZE
    import numpy as np
    from scipy import stats as scipy_stats

    print("[v5] Running Optuna hyperparameter search (fwd_3m target) …")
    signal_dates = panel.index.get_level_values('signal_date')
    max_sd = signal_dates.max()
    # Use last 6 monthly cutoffs for tuning
    tune_cutoffs = pd.date_range(
        end   = max_sd,
        periods = 7,
        freq  = 'MS',
    )[-6:]

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 800),
            'learning_rate':    trial.suggest_float('lr', 0.01, 0.1, log=True),
            'max_depth':        trial.suggest_int('max_depth', 3, 7),
            'min_child_weight': trial.suggest_int('mcw', 5, 30),
            'subsample':        trial.suggest_float('ss', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.5, 1.0),
        }
        from ml.multi_horizon import _xgb_quantile
        import math
        feat_cols = [c for c in panel.columns
                     if c not in ['fwd_1m','fwd_3m','fwd_12m','quarter_end','data_lag_days']]
        ic_vals = []
        for cutoff in tune_cutoffs:
            cutoff_end = cutoff + pd.DateOffset(days=30)
            train_mask = signal_dates < cutoff
            test_mask  = (signal_dates >= cutoff) & (signal_dates < cutoff_end)
            train = panel[train_mask].dropna(subset=['fwd_3m'])
            test  = panel[test_mask].dropna(subset=['fwd_3m'])
            if len(train) < MIN_TRAIN or test_mask.sum() < MIN_TEST_SIZE:
                continue
            X_tr = np.clip(train[feat_cols].values.astype(float), -1e9, 1e9)
            y_tr = train['fwd_3m'].values.astype(float)
            X_te = np.clip(test[feat_cols].values.astype(float), -1e9, 1e9)
            y_te = test['fwd_3m'].values.astype(float)
            X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
            X_te = np.where(np.isinf(X_te), np.nan, X_te)
            m = _xgb_quantile(0.50, params)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            mask = ~(np.isnan(y_te) | np.isnan(pred))
            if mask.sum() < 5:
                continue
            rho, _ = scipy_stats.spearmanr(y_te[mask], pred[mask])
            if not math.isnan(rho):
                ic_vals.append(rho)
        return float(np.mean(ic_vals)) if ic_vals else 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print(f"[v5] Best params: {study.best_params}  IC={study.best_value:.4f}")
    return study.best_params


def step_watchlist(panel: pd.DataFrame, model) -> pd.DataFrame:
    import subprocess
    print("\n[v5] Generating HTML watchlist …")
    result = subprocess.run(
        [sys.executable, "generate_watchlist_html.py",
         "--panel",  str(PANEL_PATH),
         "--model",  str(MODEL_PATH)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[v5] Warning: watchlist generation failed; see output above")
    return pd.DataFrame()


def main():
    args    = parse_args()
    tickers = step_universe(args)
    panel   = step_collect(tickers, args)
    model, metrics = step_train(panel, args)
    step_watchlist(panel, model)
    print(f"\n[v5] Done.\n  Panel  → {PANEL_PATH}\n  Model  → {MODEL_PATH}")


if __name__ == "__main__":
    main()
