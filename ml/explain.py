"""
Model explainability for Project Iris.

Outputs:
  1. SHAP summary  — which features drive predictions most strongly
  2. Per-feature IC table — which features individually correlate with fwd return
  3. Quintile return table — do model predictions actually rank stocks correctly?
  4. Long-only performance — cumulative return from going long top quintile each Q
"""

import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')


OUTPUT_DIR = Path('ml_output')


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Per-feature IC analysis
# ---------------------------------------------------------------------------

def feature_ic_table(panel: pd.DataFrame, feat_cols: list[str],
                     target: str = 'fwd_return') -> pd.DataFrame:
    """
    Compute Spearman IC between each feature and the forward return.
    Grouped by quarter to get mean IC and ICIR.

    Returns DataFrame sorted by |mean_IC| descending.
    """
    quarter_ends = sorted(panel.index.get_level_values('quarter_end').unique())
    ic_records = {f: [] for f in feat_cols}

    for qe in quarter_ends:
        qdf = panel.loc[panel.index.get_level_values('quarter_end') == qe]
        y   = qdf[target].values
        for feat in feat_cols:
            x = qdf[feat].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                continue
            rho, _ = stats.spearmanr(x[mask], y[mask])
            if not math.isnan(rho):
                ic_records[feat].append(rho)

    rows = []
    for feat, ics in ic_records.items():
        if not ics:
            rows.append({'feature': feat, 'mean_IC': float('nan'),
                         'std_IC': float('nan'), 'ICIR': float('nan'),
                         'pct_pos': float('nan'), 'n_quarters': 0})
            continue
        arr = np.array(ics)
        rows.append({
            'feature':    feat,
            'mean_IC':    float(np.mean(arr)),
            'std_IC':     float(np.std(arr)),
            'ICIR':       float(np.mean(arr) / (np.std(arr) + 1e-9)),
            'pct_pos':    float(np.mean(arr > 0)),
            'n_quarters': len(arr),
        })

    df = pd.DataFrame(rows).set_index('feature')
    df = df.sort_values('mean_IC', key=abs, ascending=False)
    return df


# ---------------------------------------------------------------------------
# 2. SHAP feature importance
# ---------------------------------------------------------------------------

def shap_importance(model_xgb,
                    preprocessor,
                    panel: pd.DataFrame,
                    feat_cols: list[str],
                    max_rows: int = 2000) -> pd.DataFrame:
    """
    Compute mean |SHAP| for each feature using the final XGBoost model.
    Returns DataFrame sorted by importance.
    """
    try:
        import shap
    except ImportError:
        print("[explain] shap not installed — skipping SHAP analysis")
        return pd.DataFrame()

    sample = panel[feat_cols].dropna(how='all')
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=42)

    X = preprocessor.transform(sample.values)
    explainer  = shap.TreeExplainer(model_xgb)
    shap_vals  = explainer.shap_values(X)

    mean_abs   = np.abs(shap_vals).mean(axis=0)
    df = pd.DataFrame({
        'feature':     feat_cols,
        'mean_abs_shap': mean_abs,
    }).set_index('feature').sort_values('mean_abs_shap', ascending=False)
    return df


# ---------------------------------------------------------------------------
# 3. Quintile return analysis
# ---------------------------------------------------------------------------

def quintile_analysis(oof_df: pd.DataFrame,
                      pred_col: str = 'pred_xgb') -> pd.DataFrame:
    """
    Divide predictions into quintiles (Q1 = bottom, Q5 = top) each quarter.
    Report average realised forward return per quintile.
    """
    if oof_df.empty or pred_col not in oof_df.columns:
        return pd.DataFrame()

    results = []
    for qe, grp in oof_df.groupby('quarter_end'):
        grp = grp.dropna(subset=[pred_col, 'fwd_return'])
        if len(grp) < 10:
            continue
        grp = grp.copy()
        grp['quintile'] = pd.qcut(grp[pred_col], 5,
                                   labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        for q, qdf in grp.groupby('quintile', observed=True):
            results.append({
                'quarter_end': qe,
                'quintile':    str(q),
                'mean_fwd_return': qdf['fwd_return'].mean(),
                'n': len(qdf),
            })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    summary = df.groupby('quintile')['mean_fwd_return'].agg(['mean', 'std', 'count'])
    summary.columns = ['mean_return', 'std_return', 'n_obs']
    summary['ICIR_proxy'] = summary['mean_return'] / (summary['std_return'] + 1e-9)
    return summary


# ---------------------------------------------------------------------------
# 4. Cumulative long-top-quintile equity curve
# ---------------------------------------------------------------------------

def equity_curve(oof_df: pd.DataFrame,
                 pred_col: str = 'pred_xgb',
                 top_pct: float = 0.20) -> pd.Series:
    """
    Simulate going long the top top_pct of predictions each quarter.
    Returns a cumulative return Series indexed by quarter_end.
    """
    quarters = sorted(oof_df['quarter_end'].unique())
    cum_ret  = 1.0
    curve    = {}

    for qe in quarters:
        grp = oof_df[oof_df['quarter_end'] == qe].dropna(subset=[pred_col, 'fwd_return'])
        if len(grp) < 5:
            continue
        n_top = max(1, int(len(grp) * top_pct))
        top   = grp.nlargest(n_top, pred_col)
        period_ret = top['fwd_return'].mean()
        cum_ret *= (1 + period_ret)
        curve[qe] = cum_ret

    return pd.Series(curve, name='cumulative_return')


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------

def full_report(oof_df: pd.DataFrame,
                model_xgb,
                preprocessor,
                panel: pd.DataFrame,
                feat_cols: list[str],
                save: bool = True) -> dict:
    """
    Run all analyses, print a human-readable report, optionally save CSVs.
    Returns dict with all analysis DataFrames.
    """
    _ensure_output_dir()
    print("\n" + "═" * 60)
    print("  PROJECT IRIS — MODEL EXPLAINABILITY REPORT")
    print("═" * 60)

    print("\n── 1. Per-feature Information Coefficient (IC) ──────────")
    ic_df = feature_ic_table(panel, feat_cols)
    print(ic_df.head(20).to_string())
    if save:
        ic_df.to_csv(OUTPUT_DIR / 'feature_ic.csv')

    print("\n── 2. SHAP Feature Importance ────────────────────────────")
    shap_df = shap_importance(model_xgb, preprocessor, panel, feat_cols)
    if not shap_df.empty:
        print(shap_df.head(20).to_string())
        if save:
            shap_df.to_csv(OUTPUT_DIR / 'shap_importance.csv')

    print("\n── 3. Quintile Return Analysis ───────────────────────────")
    quint_df = quintile_analysis(oof_df)
    if not quint_df.empty:
        print(quint_df.to_string())
        if save:
            quint_df.to_csv(OUTPUT_DIR / 'quintile_returns.csv')

    print("\n── 4. Long Top-Quintile Equity Curve ─────────────────────")
    curve = equity_curve(oof_df)
    if not curve.empty:
        total = curve.iloc[-1] - 1
        print(f"  Total return  : {total:.1%}")
        print(f"  Quarters      : {len(curve)}")
        ann_approx = (curve.iloc[-1] ** (4 / len(curve))) - 1
        print(f"  Ann. return ≈ : {ann_approx:.1%}")
        if save:
            curve.to_csv(OUTPUT_DIR / 'equity_curve.csv')

    if not oof_df.empty and save:
        oof_df.to_csv(OUTPUT_DIR / 'oof_predictions.csv', index=False)

    print("\n── Key insight: top predictors ───────────────────────────")
    if not ic_df.empty:
        top_pos = ic_df[ic_df['mean_IC'] > 0].head(5)
        top_neg = ic_df[ic_df['mean_IC'] < 0].head(5)
        print("  Bullish signals (higher feature → higher return):")
        for f, row in top_pos.iterrows():
            print(f"    {f:<35} IC={row['mean_IC']:+.4f}  ICIR={row['ICIR']:+.2f}")
        print("  Bearish signals (higher feature → lower return):")
        for f, row in top_neg.iterrows():
            print(f"    {f:<35} IC={row['mean_IC']:+.4f}  ICIR={row['ICIR']:+.2f}")

    return {
        'feature_ic':     ic_df,
        'shap':           shap_df,
        'quintile':       quint_df,
        'equity_curve':   curve,
    }
