"""
Enhanced feature engineering — v2.

Adds to the original ml/features.py:
  • Piotroski F-Score (9 components individually)
  • Beneish M-Score (8 components + flag)
  • Altman Z''-Score (4 components + zone)
  • DSO / DIO / DPO / Cash Conversion Cycle
  • Shareholding pattern features (India: NSE; US: partial via 13F)
  • SEC insider transaction features (US)
  • 3-tier news sentiment features (GDELT + FinBERT)

NaN policy:
  Every feature is either computed from real data or NaN.
  XGBoost handles NaN natively by learning the best split direction.
  No imputation, no assumptions.
"""

import math
import numpy as np
import pandas as pd

from data.financials import (
    safe_get, trailing_sum, yoy_growth, qoq_growth, cash_cycle_features
)
from ml.scores import piotroski_f_score, beneish_m_score, altman_z_score
# Original feature functions reused
from ml.features import (
    valuation_features, profitability_features, revenue_growth_features,
    earnings_growth_features, balance_sheet_features, cashflow_quality_features,
    income_detail_features, momentum_features,
    roic_features, asset_growth_features, gross_profit_assets,
    tech_features, earnings_surprise_features_v2, eps_acceleration_features,
)


def _nan():
    return float('nan')


def _safe_div(a, b):
    if math.isnan(a) or math.isnan(b) or b == 0:
        return _nan()
    return a / b


# ---------------------------------------------------------------------------
# Academic scores
# ---------------------------------------------------------------------------

def score_features(stmts: dict) -> dict:
    """Piotroski F, Beneish M, Altman Z — computed from annual statements."""
    ai = stmts.get('annual_income',   pd.DataFrame())
    ab = stmts.get('annual_balance',  pd.DataFrame())
    ac = stmts.get('annual_cashflow', pd.DataFrame())
    mkt_cap = float(stmts.get('info', {}).get('marketCap', 0) or 0)

    pf = piotroski_f_score(ai, ab, ac)
    bm = beneish_m_score(ai, ab, ac)
    az = altman_z_score(ab, ai, mkt_cap)

    feats = {}
    feats['piotroski_score']    = float(pf['score'])
    feats['piotroski_partial']  = float(pf['partial'])
    for k, v in pf.items():
        if k.startswith('F') and isinstance(v, (int, float)):
            feats[f'piotroski_{k}'] = float(v)

    feats['beneish_m']          = bm['m_score']
    feats['beneish_flag']       = bm['flag_manipulate']
    for k, v in bm.items():
        if k.startswith('beneish_') and k != 'beneish_flag':
            feats[k] = v

    feats['altman_z']           = az['altman_z']
    feats['altman_T1']          = az['altman_T1_wc_ta']
    feats['altman_T2']          = az['altman_T2_re_ta']
    feats['altman_T3']          = az['altman_T3_ebit_ta']
    feats['altman_T4']          = az['altman_T4_bve_tl']

    return feats


# ---------------------------------------------------------------------------
# Cash cycle
# ---------------------------------------------------------------------------

def cash_cycle_feat(stmts: dict) -> dict:
    return cash_cycle_features(stmts['income'], stmts['balance'])


# ---------------------------------------------------------------------------
# Analyst / consensus proxy from yfinance
# ---------------------------------------------------------------------------

def analyst_features(info: dict) -> dict:
    """
    Extract analyst-derived features from yfinance .info dict.
    These reflect consensus at the time of data pull (live only, not historical).
    """
    nan = _nan()

    target_mean  = float(info.get('targetMeanPrice')  or nan)
    target_high  = float(info.get('targetHighPrice')  or nan)
    target_low   = float(info.get('targetLowPrice')   or nan)
    current      = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)
    n_analysts   = float(info.get('numberOfAnalystOpinions') or nan)

    # Consensus upside = (target_mean - current) / current
    upside       = _safe_div(target_mean - current, current) if current > 0 else nan
    # Target range width (normalised) = dispersion proxy
    target_range = _safe_div(target_high - target_low, current) if not (
        math.isnan(target_high) or math.isnan(target_low) or current == 0
    ) else nan

    rec_key = info.get('recommendationKey', '')
    rec_score = {
        'strongBuy': 5.0, 'buy': 4.0, 'hold': 3.0,
        'underperform': 2.0, 'sell': 1.0, '': nan
    }.get(rec_key, nan)

    return {
        'analyst_upside':      upside,
        'analyst_target_range': target_range,
        'analyst_n':           n_analysts,
        'analyst_rec_score':   rec_score,
    }


# ---------------------------------------------------------------------------
# Short interest (US — from yfinance .info)
# ---------------------------------------------------------------------------

def short_interest_features(info: dict) -> dict:
    """
    Short interest metrics from yfinance (sourced from FINRA/exchange data).
    Only available for US stocks.
    """
    nan = _nan()
    short_pct   = float(info.get('shortPercentOfFloat') or nan)
    short_ratio = float(info.get('shortRatio') or nan)  # days to cover

    # Short interest change not directly available in yfinance info
    # We record what we have; the model will learn from level signals
    return {
        'short_pct_float':   short_pct,
        'short_ratio_dtc':   short_ratio,
    }


# ---------------------------------------------------------------------------
# Ownership concentration (from yfinance .info)
# ---------------------------------------------------------------------------

def ownership_features(info: dict) -> dict:
    nan = _nan()
    inst_pct = float(info.get('heldPercentInstitutions') or nan)
    insider_pct = float(info.get('heldPercentInsiders') or nan)
    return {
        'own_institutional_pct': inst_pct,
        'own_insider_pct':       insider_pct,
    }


# ---------------------------------------------------------------------------
# Earnings surprise proxy
# ---------------------------------------------------------------------------

def earnings_surprise_features(info: dict) -> dict:
    """
    Post-earnings drift (PEAD) proxy: actual EPS vs estimated EPS from yfinance.
    """
    nan = _nan()
    eps_actual   = float(info.get('trailingEps') or nan)
    eps_forward  = float(info.get('forwardEps')  or nan)
    eps_growth   = _safe_div(eps_forward - eps_actual, abs(eps_actual)) if not (
        math.isnan(eps_actual) or math.isnan(eps_forward) or eps_actual == 0
    ) else nan

    # PEG ratio = P/E / earnings growth rate
    pe  = float(info.get('trailingPE') or nan)
    peg = float(info.get('pegRatio')   or nan)

    return {
        'eps_growth_fwd':  eps_growth,
        'peg_ratio':       peg,
    }


# ---------------------------------------------------------------------------
# Master: compute all features for one snapshot
# ---------------------------------------------------------------------------

def compute_all_v2(stmts: dict,
                   as_of: pd.Timestamp,
                   shareholding: dict | None = None,
                   insider: dict | None = None,
                   news: dict | None = None,
                   corp_actions: dict | None = None,
                   fno: dict | None = None) -> dict:
    """
    Compute the full feature set for one (ticker, date) snapshot.
    Optional inputs are incorporated when available; NaN otherwise.

    Args:
        stmts        : financial statements dict (income, balance, cashflow, etc.)
        as_of        : signal_date — all price/momentum features clipped to this date
        shareholding : NSE shareholding pattern features (sh_* keys)
        insider      : SEC Form 4 insider transaction features
        news         : GDELT + FinBERT news sentiment features
        corp_actions : corporate actions features (corp_* keys)
        fno          : NSE F&O derivatives features (fo_* keys)

    Returns a flat dict of float features.
    """
    info = stmts.get('info', {})
    feats = {}

    # ── Original 60 features (from ml/features.py) ──────────────────────
    feats.update(valuation_features(info, stmts))
    feats.update(profitability_features(info, stmts))
    feats.update(revenue_growth_features(stmts))
    feats.update(earnings_growth_features(stmts))
    feats.update(balance_sheet_features(stmts))
    feats.update(cashflow_quality_features(stmts))
    feats.update(income_detail_features(stmts))
    feats.update(momentum_features(stmts.get('price_hist', pd.DataFrame()), as_of))

    # ── New: Academic scores ─────────────────────────────────────────────
    feats.update(score_features(stmts))

    # ── New: Cash cycle ──────────────────────────────────────────────────
    feats.update(cash_cycle_feat(stmts))

    # ── New: Analyst consensus ───────────────────────────────────────────
    feats.update(analyst_features(info))

    # ── New: Short interest ──────────────────────────────────────────────
    feats.update(short_interest_features(info))

    # ── New: Ownership concentration ─────────────────────────────────────
    feats.update(ownership_features(info))

    # ── New: EPS growth proxy / PEAD ─────────────────────────────────────
    feats.update(earnings_surprise_features(info))

    # ── Tier-1 additions ─────────────────────────────────────────────────
    feats.update(roic_features(stmts))
    feats.update(asset_growth_features(stmts))
    feats.update(gross_profit_assets(stmts))
    feats.update(tech_features(stmts.get('price_hist', pd.DataFrame()), as_of))
    feats.update(earnings_surprise_features_v2(stmts, as_of))
    feats.update(eps_acceleration_features(stmts))

    # ── Optional: Shareholding pattern (India NSE / US 13F) ─────────────
    if shareholding:
        feats.update(shareholding)
    else:
        # Ensure columns exist (NaN) so panel schema is consistent
        for k in ['sh_promoter_pct', 'sh_promoter_pledge_pct', 'sh_fii_pct',
                  'sh_dii_pct', 'sh_mf_pct', 'sh_retail_pct',
                  'sh_promoter_delta_qoq', 'sh_fii_delta_qoq',
                  'sh_dii_delta_qoq', 'sh_pledge_delta_qoq']:
            feats[k] = _nan()

    # ── Optional: Insider transactions ───────────────────────────────────
    if insider:
        feats.update(insider)
    else:
        for k in ['insider_net_value', 'insider_n_buyers', 'insider_n_sellers',
                  'insider_buy_count', 'insider_sell_count', 'insider_buy_sell_ratio']:
            feats[k] = _nan()

    # ── Optional: News sentiment ─────────────────────────────────────────
    _news_keys = [
        'news_direct_score', 'news_direct_n',
        'news_sector_score', 'news_sector_n',
        'news_macro_score',  'news_macro_n',
        'news_pulse_7d',     'news_vol_spike',   'news_event_flag',
    ]
    if news:
        feats.update(news)
        # Ensure all keys exist
        for k in _news_keys:
            feats.setdefault(k, _nan())
    else:
        for k in _news_keys:
            feats[k] = _nan()

    # ── Optional: Corporate actions ──────────────────────────────────────
    _corp_keys = [
        'corp_div_yield_ttm', 'corp_div_growth_3y', 'corp_div_consistency',
        'corp_buyback_flag',  'corp_rights_flag',   'corp_bonus_flag',
        'corp_promoter_buy_flag',
    ]
    if corp_actions:
        feats.update(corp_actions)
        for k in _corp_keys:
            feats.setdefault(k, _nan())
    else:
        for k in _corp_keys:
            feats[k] = _nan()

    # ── Optional: F&O derivatives ────────────────────────────────────────
    _fno_keys = [
        'fo_pcr', 'fo_pcr_trend', 'fo_iv_atm', 'fo_iv_pct_52w',
        'fo_oi_change_5d_pct', 'fo_long_buildup', 'fo_short_buildup',
        'fo_max_pain',
    ]
    if fno:
        feats.update(fno)
        for k in _fno_keys:
            feats.setdefault(k, _nan())
    else:
        for k in _fno_keys:
            feats[k] = _nan()

    # ── Market cap bucket (size proxy) ───────────────────────────────────
    mkt_cap = float(info.get('marketCap') or 0)
    # SEBI definitions (INR): large >₹20k Cr = 2e11, mid ₹5–20k Cr, small <₹5k Cr
    if mkt_cap >= 2e11:
        feats['market_cap_bucket'] = 2.0
    elif mkt_cap >= 5e10:
        feats['market_cap_bucket'] = 1.0
    elif mkt_cap > 0:
        feats['market_cap_bucket'] = 0.0
    else:
        feats['market_cap_bucket'] = _nan()

    return feats


# ---------------------------------------------------------------------------
# Feature name registry (for column ordering consistency)
# ---------------------------------------------------------------------------

def feature_names() -> list[str]:
    """Return the canonical ordered list of all feature names."""
    dummy_stmts = {
        'income': pd.DataFrame(), 'balance': pd.DataFrame(),
        'cashflow': pd.DataFrame(), 'price_hist': pd.DataFrame(),
        'annual_income': pd.DataFrame(), 'annual_balance': pd.DataFrame(),
        'annual_cashflow': pd.DataFrame(), 'info': {},
        'earnings_hist': pd.DataFrame(),
    }
    dummy = compute_all_v2(dummy_stmts, pd.Timestamp('2020-01-01'))
    return list(dummy.keys())


FEATURE_NAMES = feature_names()
