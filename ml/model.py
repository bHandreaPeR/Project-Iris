"""
Walk-forward regression model for Project Iris.

Two models are trained in tandem:
  - XGBoost  : captures non-linear interactions between features (primary)
  - Ridge    : linear baseline, fully interpretable coefficients

Walk-forward protocol (time-series safe):
  For each test quarter Q:
    Train on all rows with quarter_end < Q  (growing window)
    Predict rows in quarter Q
    Record IC (Spearman rank correlation) and hit-rate for that quarter

Evaluation metrics:
  IC  (Information Coefficient) : Spearman correlation of predicted vs actual
                                  fwd_return.  IC > 0.05 is usable; > 0.10
                                  is strong in factor research.
  Hit rate                       : fraction of top-N predictions that beat median
  ICIR (IC / std(IC))           : Sharpe of the IC series across folds
  Long-only Sharpe               : if we went long the top-quintile each quarter
"""

import math
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

import xgboost as xgb

warnings.filterwarnings('ignore')


TARGET   = 'fwd_return'
MIN_TRAIN_ROWS = 60   # minimum observations before we start testing


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def _make_preprocessor():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  RobustScaler()),
    ])


def _make_xgb() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _make_ridge() -> Pipeline:
    return Pipeline([
        ('pre',   _make_preprocessor()),
        ('model', Ridge(alpha=10.0)),
    ])


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def ic_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC between predictions and realised returns."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 5:
        return float('nan')
    rho, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, top_pct: float = 0.2) -> float:
    """
    Fraction of top top_pct% predictions that are in the top top_pct% of actual.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return float('nan')
    yt = y_true[mask]
    yp = y_pred[mask]
    n  = max(1, int(len(yp) * top_pct))
    top_pred_idx = np.argsort(yp)[-n:]
    top_true_threshold = np.percentile(yt, (1 - top_pct) * 100)
    hits = (yt[top_pred_idx] >= top_true_threshold).sum()
    return float(hits / n)


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

def walk_forward(panel: pd.DataFrame,
                 min_quarters: int = 8
                 ) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Run walk-forward cross-validation over the panel.

    Args:
        panel        : DataFrame indexed by (ticker, quarter_end), must contain TARGET
        min_quarters : minimum unique quarters in training before first test

    Returns:
        oof_df    : out-of-fold predictions DataFrame (ticker, quarter_end, pred_xgb, pred_ridge, fwd_return)
        xgb_model : final XGBoost model (trained on all data)
        ridge_model: final Ridge model  (trained on all data)
    """
    feat_cols = [c for c in panel.columns if c != TARGET and c != 'entry_date']
    quarter_ends = sorted(panel.index.get_level_values('quarter_end').unique())

    records = []
    ic_xgb_history   = []
    ic_ridge_history  = []

    for i, test_q in enumerate(quarter_ends):
        train_mask = panel.index.get_level_values('quarter_end') < test_q
        test_mask  = panel.index.get_level_values('quarter_end') == test_q

        train = panel[train_mask].dropna(subset=[TARGET])
        test  = panel[test_mask]

        if len(train) < MIN_TRAIN_ROWS:
            continue
        n_unique_train_q = train.index.get_level_values('quarter_end').nunique()
        if n_unique_train_q < min_quarters:
            continue

        X_train = train[feat_cols].values
        y_train = train[TARGET].values
        X_test  = test[feat_cols].values
        y_test  = test[TARGET].values

        # ── XGBoost ──────────────────────────────────────────────────────
        pre_xgb = _make_preprocessor()
        X_tr_xgb = pre_xgb.fit_transform(X_train)
        X_te_xgb = pre_xgb.transform(X_test)

        xgb_m = _make_xgb()
        xgb_m.fit(X_tr_xgb, y_train)
        pred_xgb = xgb_m.predict(X_te_xgb)

        # ── Ridge ─────────────────────────────────────────────────────────
        ridge_m = _make_ridge()
        ridge_m.fit(X_train, y_train)
        pred_ridge = ridge_m.predict(X_test)

        ic_x = ic_score(y_test, pred_xgb)
        ic_r = ic_score(y_test, pred_ridge)
        hr_x = hit_rate(y_test, pred_xgb)

        if not math.isnan(ic_x):
            ic_xgb_history.append(ic_x)
        if not math.isnan(ic_r):
            ic_ridge_history.append(ic_r)

        print(f"  Q {test_q.date()}  n={len(test):3d}  "
              f"IC_xgb={ic_x:+.3f}  IC_ridge={ic_r:+.3f}  HitRate={hr_x:.2f}")

        for j, (tkr, qe) in enumerate(test.index):
            records.append({
                'ticker':      tkr,
                'quarter_end': qe,
                'pred_xgb':    float(pred_xgb[j]),
                'pred_ridge':  float(pred_ridge[j]),
                'fwd_return':  float(y_test[j]) if not math.isnan(float(y_test[j])) else float('nan'),
            })

    # Summary statistics
    def _icir(ic_list):
        if len(ic_list) < 3:
            return float('nan')
        return float(np.mean(ic_list) / (np.std(ic_list) + 1e-9))

    stats_xgb = {
        'mean_IC':    float(np.nanmean(ic_xgb_history)) if ic_xgb_history else float('nan'),
        'std_IC':     float(np.nanstd(ic_xgb_history))  if ic_xgb_history else float('nan'),
        'ICIR':       _icir(ic_xgb_history),
        'pct_pos_IC': float(np.mean([x > 0 for x in ic_xgb_history])) if ic_xgb_history else float('nan'),
        'n_folds':    len(ic_xgb_history),
    }
    stats_ridge = {
        'mean_IC':    float(np.nanmean(ic_ridge_history)) if ic_ridge_history else float('nan'),
        'std_IC':     float(np.nanstd(ic_ridge_history))  if ic_ridge_history else float('nan'),
        'ICIR':       _icir(ic_ridge_history),
        'pct_pos_IC': float(np.mean([x > 0 for x in ic_ridge_history])) if ic_ridge_history else float('nan'),
        'n_folds':    len(ic_ridge_history),
    }

    print("\n─── Walk-forward summary ───────────────────────────────")
    print(f"XGBoost  : mean IC={stats_xgb['mean_IC']:+.4f}  ICIR={stats_xgb['ICIR']:+.2f}  "
          f"pct_pos={stats_xgb['pct_pos_IC']:.0%}  folds={stats_xgb['n_folds']}")
    print(f"Ridge    : mean IC={stats_ridge['mean_IC']:+.4f}  ICIR={stats_ridge['ICIR']:+.2f}  "
          f"pct_pos={stats_ridge['pct_pos_IC']:.0%}  folds={stats_ridge['n_folds']}")

    oof_df = pd.DataFrame(records) if records else pd.DataFrame()

    # Retrain final models on ALL data
    final_xgb_pre = _make_preprocessor()
    all_X = panel[feat_cols].values
    all_y = panel[TARGET].values
    valid = ~np.isnan(all_y)
    X_all = final_xgb_pre.fit_transform(all_X[valid])
    y_all = all_y[valid]

    final_xgb = _make_xgb()
    final_xgb.fit(X_all, y_all)

    final_ridge = _make_ridge()
    final_ridge.fit(all_X[valid], y_all)

    return oof_df, final_xgb, final_ridge, final_xgb_pre, feat_cols


# ---------------------------------------------------------------------------
# Predict current period
# ---------------------------------------------------------------------------

def predict_now(panel: pd.DataFrame,
                model_xgb,
                preprocessor,
                feat_cols: list[str],
                top_n: int = 20) -> pd.DataFrame:
    """
    Generate predictions for the most recent quarter in the panel.
    Returns the top_n stocks ranked by predicted forward return.
    """
    latest_q = panel.index.get_level_values('quarter_end').max()
    current  = panel.loc[panel.index.get_level_values('quarter_end') == latest_q]

    X = preprocessor.transform(current[feat_cols].values)
    preds = model_xgb.predict(X)

    result = current[feat_cols].copy()
    result['pred_fwd_return'] = preds
    result = result[['pred_fwd_return']].sort_values('pred_fwd_return', ascending=False)
    result['rank'] = range(1, len(result) + 1)
    return result.head(top_n)
