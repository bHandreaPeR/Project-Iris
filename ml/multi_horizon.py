"""
Multi-horizon XGBoost + LightGBM regression with quantile confidence intervals.

For each of three horizons (1m, 3m, 12m) we train THREE quantile models:
  q10 — 10th-percentile  (lower bound)
  q50 — 50th-percentile  (point estimate / median)
  q90 — 90th-percentile  (upper bound)

Walk-forward protocol (v5 — event-driven monthly folds):
  Folds keyed by calendar month, not by quarter_end.
  For each monthly cutoff C (starting 18 months after earliest signal_date):
    Train on all rows where signal_date < C
    Test  on all rows where C ≤ signal_date < C + 30d
  This naturally handles companies filing at different dates:
    Bandhan filing April 28 → test fold for May 2026
    Infosys filing October 15 → test fold for November 2025
  Minimum test fold size: 10 rows (skip sparse months).

Model ensemble:
  XGBoost quantile regressor (primary, used in walk-forward CV)
  LightGBM quantile regressor (secondary, trained in fit_final only)
  Blend: 0.70 × XGB + 0.30 × LGB for point estimate prediction.

NaN handling:
  XGBoost and LightGBM both handle np.nan natively.
  NO imputation, NO preprocessing.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

warnings.filterwarnings('ignore')

TARGET_COLS = ['fwd_1m', 'fwd_3m', 'fwd_12m']
HORIZONS    = {'fwd_1m': '1 month', 'fwd_3m': '3 months', 'fwd_12m': '12 months'}

MIN_TRAIN         = 100   # minimum training rows before first test fold
MIN_TRAIN_MONTHS  = 18    # minimum calendar months of training data
MIN_TEST_SIZE     = 10    # minimum test rows in a fold (skip sparse months)

LGB_WEIGHT        = 0.30  # LightGBM blend weight in final predictions


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _xgb_quantile(alpha: float, params: dict | None = None) -> xgb.XGBRegressor:
    defaults = dict(
        objective='reg:quantileerror',
        quantile_alpha=alpha,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        missing=np.nan,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method='hist',
    )
    if params:
        defaults.update(params)
    return xgb.XGBRegressor(**defaults)


def _lgb_quantile(alpha: float) -> Any:
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            n_jobs=-1,
            verbosity=-1,
            random_state=42,
        )
    except ImportError:
        return None


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
    target:     str
    feat_cols:  list[str] = field(default_factory=list)
    xgb_params: dict      = field(default_factory=dict)

    # Fitted models (set after fit_final)
    q10: Optional[xgb.XGBRegressor] = None
    q50: Optional[xgb.XGBRegressor] = None
    q90: Optional[xgb.XGBRegressor] = None
    lgb_q50: Optional[Any] = None   # LightGBM blend model

    oof_records: list = field(default_factory=list)
    _winsor_lo: Optional[np.ndarray] = None
    _winsor_hi: Optional[np.ndarray] = None

    def _fit_one_xgb(self, alpha: float, X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
        X = np.where(np.isinf(X), np.nan, X)
        model = _xgb_quantile(alpha, self.xgb_params or None)
        model.fit(X, y)
        return model

    def walk_forward(self, panel: pd.DataFrame) -> dict:
        """
        Monthly walk-forward CV.
        Folds: train on signal_date < cutoff, test on [cutoff, cutoff+30d).
        Populates self.oof_records and returns per-fold + aggregate metrics.
        """
        signal_dates = panel.index.get_level_values('signal_date')
        min_sd = signal_dates.min()
        max_sd = signal_dates.max()

        # Monthly cutoffs starting after MIN_TRAIN_MONTHS of data
        start_cutoff = min_sd + pd.DateOffset(months=MIN_TRAIN_MONTHS)
        cutoffs = pd.date_range(start=start_cutoff, end=max_sd, freq='MS')

        ic_history  = []
        hr_history  = []
        ls_history  = []

        for cutoff in cutoffs:
            cutoff_end = cutoff + pd.DateOffset(days=30)

            train_mask = signal_dates < cutoff
            test_mask  = (signal_dates >= cutoff) & (signal_dates < cutoff_end)

            train = panel[train_mask].dropna(subset=[self.target])
            test  = panel[test_mask]

            if len(train) < MIN_TRAIN:
                continue
            # Check sufficient training history (months, not just rows)
            train_months = train.index.get_level_values('signal_date').to_period('M').nunique()
            if train_months < MIN_TRAIN_MONTHS:
                continue
            if test_mask.sum() < MIN_TEST_SIZE:
                continue

            X_tr = train[self.feat_cols].values.astype(float)
            y_tr = train[self.target].values.astype(float)
            X_te = test[self.feat_cols].values.astype(float)
            y_te = test[self.target].values.astype(float)

            X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
            X_te = np.where(np.isinf(X_te), np.nan, X_te)

            # Per-fold winsorization (computed on train only)
            lo = np.nanpercentile(X_tr, 1, axis=0)
            hi = np.nanpercentile(X_tr, 99, axis=0)
            X_tr = np.clip(X_tr, lo, hi)
            X_te = np.clip(X_te, lo, hi)
            self._winsor_lo = lo
            self._winsor_hi = hi

            m50 = self._fit_one_xgb(0.50, X_tr, y_tr)
            pred_mid = m50.predict(X_te)

            ic_val = ic(y_te, pred_mid)
            hr_val = hit_rate(y_te, pred_mid)
            ls_val = long_short_ret(y_te, pred_mid)

            if not math.isnan(ic_val):
                ic_history.append(ic_val)
                hr_history.append(hr_val)
                ls_history.append(ls_val)

            print(
                f"    [{self.target}] M {cutoff.strftime('%Y-%m')}"
                f"  n={len(test):3d}"
                f"  IC={ic_val:+.3f}"
                f"  HR@20%={hr_val:.2f}"
                f"  L/S={ls_val:+.3f}"
            )

            # Store OOF records — include signal_date + quarter_end metadata
            qe_col = test['quarter_end'] if 'quarter_end' in test.columns else pd.Series()
            for j, (tkr, sd) in enumerate(test.index):
                qe = qe_col.get((tkr, sd), pd.NaT) if not qe_col.empty else pd.NaT
                self.oof_records.append({
                    'ticker':      tkr,
                    'signal_date': sd,
                    'quarter_end': qe,
                    'horizon':     self.target,
                    'pred_q50':    float(pred_mid[j]),
                    'actual':      float(y_te[j]) if not math.isnan(float(y_te[j])) else float('nan'),
                })

        def _icir(arr):
            return float(np.mean(arr) / (np.std(arr) + 1e-9)) if len(arr) >= 3 else float('nan')

        return {
            'horizon':    self.target,
            'mean_IC':    float(np.nanmean(ic_history))  if ic_history else float('nan'),
            'median_IC':  float(np.nanmedian(ic_history)) if ic_history else float('nan'),
            'std_IC':     float(np.nanstd(ic_history))   if ic_history else float('nan'),
            'ICIR':       _icir(ic_history),
            'pct_pos_IC': float(np.mean([x > 0 for x in ic_history])) if ic_history else float('nan'),
            'mean_HR':    float(np.nanmean(hr_history))  if hr_history else float('nan'),
            'mean_LS':    float(np.nanmean(ls_history))  if ls_history else float('nan'),
            'n_folds':    len(ic_history),
        }

    def fit_final(self, panel: pd.DataFrame) -> None:
        """Retrain all quantile models (XGB + LGB) on ALL available data."""
        df = panel.dropna(subset=[self.target])
        X  = df[self.feat_cols].values.astype(float)
        y  = df[self.target].values.astype(float)
        X  = np.where(np.isinf(X), np.nan, X)

        # Winsorization bounds from full dataset
        self._winsor_lo = np.nanpercentile(X, 1, axis=0)
        self._winsor_hi = np.nanpercentile(X, 99, axis=0)
        X_clip = np.clip(X, self._winsor_lo, self._winsor_hi)

        # XGBoost — all 3 quantiles
        self.q10 = self._fit_one_xgb(0.10, X_clip, y)
        self.q50 = self._fit_one_xgb(0.50, X_clip, y)
        self.q90 = self._fit_one_xgb(0.90, X_clip, y)

        # LightGBM — q50 only (for blend)
        lgb_model = _lgb_quantile(0.50)
        if lgb_model is not None:
            try:
                lgb_model.fit(X_clip, y)
                self.lgb_q50 = lgb_model
            except Exception:
                self.lgb_q50 = None

    def predict(self, X: np.ndarray) -> dict:
        """Returns {'lower', 'point', 'upper'} with LGB blend on point estimate."""
        X = np.where(np.isinf(X.astype(float)), np.nan, X.astype(float))
        if self._winsor_lo is not None:
            X = np.clip(X, self._winsor_lo, self._winsor_hi)

        lower = float(self.q10.predict(X)[0])
        xgb_pt = float(self.q50.predict(X)[0])
        upper = float(self.q90.predict(X)[0])

        # Blend XGB + LGB for point estimate
        if self.lgb_q50 is not None:
            try:
                lgb_pt = float(self.lgb_q50.predict(X)[0])
                point  = (1 - LGB_WEIGHT) * xgb_pt + LGB_WEIGHT * lgb_pt
            except Exception:
                point = xgb_pt
        else:
            point = xgb_pt

        # Enforce quantile monotonicity
        lower = min(lower, point)
        upper = max(upper, point)

        return {'lower': lower, 'point': point, 'upper': upper}

    def feature_importance(self) -> pd.Series:
        if self.q50 is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.q50.feature_importances_, index=self.feat_cols
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Multi-horizon ensemble
# ---------------------------------------------------------------------------

class MultiHorizonModel:
    """Wraps three HorizonModels (1m, 3m, 12m)."""

    def __init__(self):
        self.models:    dict[str, HorizonModel] = {}
        self.feat_cols: list[str] = []
        self.oof_df:    pd.DataFrame = pd.DataFrame()

    def train(self, panel: pd.DataFrame,
              xgb_params: dict | None = None) -> dict[str, dict]:
        """
        Walk-forward train all three horizon models.
        Returns summary metrics dict keyed by target column.
        """
        from ml.collector_v2 import validate_panel_schema, _META_COLS
        validate_panel_schema(panel)

        meta = _META_COLS + TARGET_COLS
        self.feat_cols = [c for c in panel.columns if c not in meta]

        all_metrics = {}
        all_oof     = []

        for target in TARGET_COLS:
            print(f"\n{'─'*55}")
            print(f"  Horizon: {HORIZONS[target]}  ({target})")
            print(f"{'─'*55}")

            model = HorizonModel(
                target=target,
                feat_cols=self.feat_cols,
                xgb_params=xgb_params or {},
            )
            metrics = model.walk_forward(panel)
            model.fit_final(panel)

            self.models[target] = model
            all_metrics[target] = metrics
            all_oof.extend(model.oof_records)

        self.oof_df = pd.DataFrame(all_oof) if all_oof else pd.DataFrame()
        return all_metrics

    def predict_ticker(self, ticker_feats: pd.Series) -> dict[str, dict]:
        """Generate predictions for a single ticker (as a feature Series)."""
        X = ticker_feats.reindex(self.feat_cols).values.astype(float).reshape(1, -1)
        X = np.where(np.isinf(X), np.nan, X)
        return {target: m.predict(X) for target, m in self.models.items()}

    def walk_forward_summary(self, metrics: dict[str, dict]) -> str:
        lines = [
            "\n" + "═" * 72,
            "  WALK-FORWARD EVALUATION SUMMARY (v5 — monthly folds)",
            "═" * 72,
            f"  {'Horizon':<12} {'Folds':>6} {'Mean IC':>8} {'Med IC':>7} "
            f"{'ICIR':>7} {'%PosIC':>7} {'HR@20%':>7} {'L/S':>7}",
            "  " + "─" * 65,
        ]
        for target, m in metrics.items():
            lines.append(
                f"  {HORIZONS[target]:<12}"
                f"  {m['n_folds']:>5d}"
                f"  {m['mean_IC']:>+.4f}"
                f"  {m['median_IC']:>+.4f}"
                f"  {m['ICIR']:>+.3f}"
                f"  {m['pct_pos_IC']:>6.0%}"
                f"  {m['mean_HR']:>6.2f}"
                f"  {m['mean_LS']:>+.3f}"
            )
        lines.append("═" * 72)
        return "\n".join(lines)

    def top_feature_importance(self, n: int = 20) -> pd.DataFrame:
        rows = []
        for target, model in self.models.items():
            fi = model.feature_importance()
            for feat, imp in fi.head(n).items():
                rows.append({'horizon': target, 'feature': feat, 'importance': imp})
        return pd.DataFrame(rows)

    def prune_zero_importance_features(self) -> list[str]:
        """
        Remove features with zero importance across ALL horizon models.
        Updates self.feat_cols in place; returns pruned list.
        """
        zero_counts: dict[str, int] = {}
        for model in self.models.values():
            fi = model.feature_importance()
            for feat in fi[fi == 0].index:
                zero_counts[feat] = zero_counts.get(feat, 0) + 1

        n_horizons = len(self.models)
        to_prune = {f for f, cnt in zero_counts.items() if cnt == n_horizons}
        pruned = [f for f in self.feat_cols if f not in to_prune]

        if to_prune:
            print(f"[prune] Removed {len(to_prune)} zero-importance features; "
                  f"{len(pruned)} remain.")
        self.feat_cols = pruned
        for model in self.models.values():
            model.feat_cols = pruned
        return pruned
