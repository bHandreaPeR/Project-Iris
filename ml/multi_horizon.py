"""
Multi-horizon XGBoost regression with quantile confidence intervals.

For each of three horizons (1m, 3m, 12m) we train THREE XGBoost models:
  q10 — 10th-percentile quantile regressor  (lower bound)
  q50 — 50th-percentile quantile regressor  (point estimate / median)
  q90 — 90th-percentile quantile regressor  (upper bound)

This gives calibrated confidence intervals without any distributional
assumptions (unlike Gaussian CI which assumes normal residuals).

NaN handling:
  XGBoost receives raw feature matrices with np.nan for missing values.
  It learns the optimal split direction for NaN values at each node.
  NO imputation, NO preprocessing.

Walk-forward protocol:
  For each test quarter Q:
    Train on all rows where quarter_end < Q  (growing window)
    Predict rows where quarter_end == Q
    Compute IC (Spearman), hit-rate@20%, and MAPE for that fold
  Minimum 8 unique quarters in training before first test.

Final models are retrained on ALL available data with a realized target,
then used for forward predictions on the current period.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

warnings.filterwarnings('ignore')

TARGET_COLS = ['fwd_1m', 'fwd_3m', 'fwd_12m']
HORIZONS    = {'fwd_1m': '1 month', 'fwd_3m': '3 months', 'fwd_12m': '12 months'}
MIN_TRAIN    = 50   # minimum training rows before first test quarter
MIN_QUARTERS = 3   # minimum unique training quarters before first test


# ---------------------------------------------------------------------------
# XGBoost quantile model factory
# ---------------------------------------------------------------------------

def _xgb_quantile(alpha: float) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=alpha,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=10,      # prevents overfitting on small samples
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        missing=np.nan,           # XGBoost native NaN — NO imputation
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method='hist',       # fastest for large datasets
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 5:
        return float('nan')
    rho, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return float(rho)


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, top_pct: float = 0.20) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return float('nan')
    yt, yp = y_true[mask], y_pred[mask]
    n = max(1, int(len(yp) * top_pct))
    top_idx = np.argsort(yp)[-n:]
    thresh  = np.percentile(yt, (1 - top_pct) * 100)
    return float((yt[top_idx] >= thresh).sum() / n)


def long_short_ret(y_true: np.ndarray, y_pred: np.ndarray,
                   top_pct: float = 0.20, bot_pct: float = 0.20) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return float('nan')
    yt, yp = y_true[mask], y_pred[mask]
    n = max(1, int(len(yp) * top_pct))
    top_ret = float(yt[np.argsort(yp)[-n:]].mean())
    bot_ret = float(yt[np.argsort(yp)[:n]].mean())
    return top_ret - bot_ret


# ---------------------------------------------------------------------------
# Single-horizon model
# ---------------------------------------------------------------------------

@dataclass
class HorizonModel:
    target:    str   # 'fwd_1m' | 'fwd_3m' | 'fwd_12m'
    feat_cols: list[str] = field(default_factory=list)
    q10: Optional[xgb.XGBRegressor] = None
    q50: Optional[xgb.XGBRegressor] = None
    q90: Optional[xgb.XGBRegressor] = None
    oof_records: list = field(default_factory=list)

    def _fit_one(self, alpha: float, X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
        X = np.where(np.isinf(X), np.nan, X)
        model = _xgb_quantile(alpha)
        model.fit(X, y)
        return model

    def walk_forward(self, panel: pd.DataFrame) -> dict:
        """
        Run walk-forward CV for this horizon.
        Populates self.oof_records and returns per-fold metrics.
        """
        quarter_ends = sorted(
            panel.index.get_level_values('quarter_end').unique()
        )
        ic_history = []
        hr_history = []
        ls_history = []

        for test_q in quarter_ends:
            train_mask = panel.index.get_level_values('quarter_end') < test_q
            test_mask  = panel.index.get_level_values('quarter_end') == test_q

            train = panel[train_mask].dropna(subset=[self.target])
            test  = panel[test_mask]

            if len(train) < MIN_TRAIN:
                continue
            if train.index.get_level_values('quarter_end').nunique() < MIN_QUARTERS:
                continue

            X_tr = train[self.feat_cols].values.astype(float)
            y_tr = train[self.target].values.astype(float)
            X_te = test[self.feat_cols].values.astype(float)
            y_te = test[self.target].values.astype(float)

            # Replace inf with nan (division-by-zero artifacts in ratio features)
            X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
            X_te = np.where(np.isinf(X_te), np.nan, X_te)

            # Per-fold winsorization at 1st/99th percentile (computed on train only)
            # Prevents base-effect distortions (e.g. NI YoY = +∞ on loss-to-profit)
            lo = np.nanpercentile(X_tr, 1, axis=0)
            hi = np.nanpercentile(X_tr, 99, axis=0)
            X_tr = np.clip(X_tr, lo, hi)
            X_te = np.clip(X_te, lo, hi)
            self._winsor_lo = lo   # store for final model
            self._winsor_hi = hi

            # Fit q50 (point estimate) for IC evaluation
            m50 = self._fit_one(0.50, X_tr, y_tr)
            pred_mid = m50.predict(X_te)

            ic_val = ic(y_te, pred_mid)
            hr_val = hit_rate(y_te, pred_mid)
            ls_val = long_short_ret(y_te, pred_mid)

            if not math.isnan(ic_val):
                ic_history.append(ic_val)
                hr_history.append(hr_val)
                ls_history.append(ls_val)

            print(
                f"    [{self.target}] Q {test_q.date()}"
                f"  n={len(test):3d}"
                f"  IC={ic_val:+.3f}"
                f"  HitRate@20%={hr_val:.2f}"
                f"  L/S={ls_val:+.3f}"
            )

            for j, (tkr, qe) in enumerate(test.index):
                self.oof_records.append({
                    'ticker':      tkr,
                    'quarter_end': qe,
                    'horizon':     self.target,
                    'pred_q50':    float(pred_mid[j]),
                    'actual':      float(y_te[j]) if not math.isnan(float(y_te[j])) else float('nan'),
                })

        def _icir(arr):
            return float(np.mean(arr) / (np.std(arr) + 1e-9)) if len(arr) >= 3 else float('nan')

        return {
            'horizon':   self.target,
            'mean_IC':   float(np.nanmean(ic_history)) if ic_history else float('nan'),
            'std_IC':    float(np.nanstd(ic_history))  if ic_history else float('nan'),
            'ICIR':      _icir(ic_history),
            'pct_pos_IC': float(np.mean([x > 0 for x in ic_history])) if ic_history else float('nan'),
            'mean_HR':   float(np.nanmean(hr_history)) if hr_history else float('nan'),
            'mean_LS':   float(np.nanmean(ls_history)) if ls_history else float('nan'),
            'n_folds':   len(ic_history),
        }

    def fit_final(self, panel: pd.DataFrame) -> None:
        """Retrain all 3 quantile models on ALL available data."""
        df = panel.dropna(subset=[self.target])
        X  = df[self.feat_cols].values.astype(float)
        y  = df[self.target].values.astype(float)
        X  = np.where(np.isinf(X), np.nan, X)
        # Compute and store winsorization bounds from full dataset
        self._winsor_lo = np.nanpercentile(X, 1, axis=0)
        self._winsor_hi = np.nanpercentile(X, 99, axis=0)
        X = np.clip(X, self._winsor_lo, self._winsor_hi)
        self.q10 = self._fit_one(0.10, X, y)
        self.q50 = self._fit_one(0.50, X, y)
        self.q90 = self._fit_one(0.90, X, y)

    def predict(self, X: np.ndarray) -> dict:
        """Returns dict with lower, point, upper predictions."""
        X = np.where(np.isinf(X.astype(float)), np.nan, X.astype(float))
        if hasattr(self, '_winsor_lo') and self._winsor_lo is not None:
            X = np.clip(X, self._winsor_lo, self._winsor_hi)
        return {
            'lower': float(self.q10.predict(X)[0]),
            'point': float(self.q50.predict(X)[0]),
            'upper': float(self.q90.predict(X)[0]),
        }

    def feature_importance(self) -> pd.Series:
        if self.q50 is None:
            return pd.Series(dtype=float)
        imp = self.q50.feature_importances_
        return pd.Series(imp, index=self.feat_cols).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Multi-horizon ensemble
# ---------------------------------------------------------------------------

class MultiHorizonModel:
    """Wraps three HorizonModels (1m, 3m, 12m)."""

    def __init__(self):
        self.models: dict[str, HorizonModel] = {}
        self.feat_cols: list[str] = []
        self.oof_df: pd.DataFrame = pd.DataFrame()

    def train(self, panel: pd.DataFrame) -> dict[str, dict]:
        """
        Walk-forward train all three horizon models.
        Returns summary metrics dict keyed by target column.
        """
        self.feat_cols = [c for c in panel.columns
                          if c not in TARGET_COLS + ['entry_date']]

        all_metrics = {}
        all_oof = []

        for target in TARGET_COLS:
            print(f"\n{'─'*55}")
            print(f"  Horizon: {HORIZONS[target]}  ({target})")
            print(f"{'─'*55}")

            model = HorizonModel(target=target, feat_cols=self.feat_cols)
            metrics = model.walk_forward(panel)
            model.fit_final(panel)

            self.models[target] = model
            all_metrics[target] = metrics
            all_oof.extend(model.oof_records)

        self.oof_df = pd.DataFrame(all_oof) if all_oof else pd.DataFrame()
        return all_metrics

    def predict_ticker(self, ticker_feats: pd.Series) -> dict[str, dict]:
        """
        Generate predictions for a single ticker (as a feature Series).
        Returns dict keyed by target with lower/point/upper.
        """
        X = ticker_feats.reindex(self.feat_cols).values.astype(float).reshape(1, -1)
        X = np.where(np.isinf(X), np.nan, X)
        return {target: m.predict(X) for target, m in self.models.items()}

    def walk_forward_summary(self, metrics: dict[str, dict]) -> str:
        lines = ["\n" + "═" * 65,
                 "  WALK-FORWARD EVALUATION SUMMARY", "═" * 65,
                 f"  {'Horizon':<12} {'Mean IC':>8} {'ICIR':>7} {'% Pos IC':>9} {'Hit@20%':>9} {'L/S Ret':>8}"]
        lines.append("  " + "─" * 55)
        for target, m in metrics.items():
            lines.append(
                f"  {HORIZONS[target]:<12}"
                f"  {m['mean_IC']:>+.4f}"
                f"  {m['ICIR']:>+.3f}"
                f"  {m['pct_pos_IC']:>8.0%}"
                f"  {m['mean_HR']:>8.2f}"
                f"  {m['mean_LS']:>+.3f}"
            )
        lines.append("═" * 65)
        return "\n".join(lines)

    def top_feature_importance(self, n: int = 20) -> pd.DataFrame:
        rows = []
        for target, model in self.models.items():
            fi = model.feature_importance()
            for feat, imp in fi.head(n).items():
                rows.append({'horizon': target, 'feature': feat, 'importance': imp})
        return pd.DataFrame(rows)
