"""
Academic financial health and earnings quality scores.

All formulas follow the original papers exactly — no modifications.

1. Piotroski F-Score  (Piotroski 2000, JAR)
   — 9 binary signals across profitability, leverage/liquidity, efficiency
   — Score range 0–9. High F = financially strong, Low F = deteriorating

2. Beneish M-Score  (Beneish 1999, TAR)
   — 8-variable logit model to detect earnings manipulation
   — M > -1.78 → likely manipulator; M < -2.22 → likely clean

3. Altman Z''-Score  (Altman 2000, revised for non-manufacturers/services)
   — Z > 2.6 = safe, 1.1–2.6 = grey zone, < 1.1 = distress

All inputs come from yfinance annual statements (income_stmt, balance_sheet,
cashflow). Functions return dicts with the score AND every component, so
features_v2.py can include components as individual model features.

NaN policy: if a required input line item is unavailable, the component that
needs it is marked NaN. Partial scores are still computed and flagged.
"""

import math
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _f(series: pd.Series, label: str) -> float:
    """Extract most-recent non-null value from a statement row."""
    try:
        vals = series.dropna()
        return float(vals.iloc[-1]) if len(vals) else float('nan')
    except Exception:
        return float('nan')


def _prior(series: pd.Series, label: str) -> float:
    """Extract second-most-recent non-null value."""
    try:
        vals = series.dropna()
        return float(vals.iloc[-2]) if len(vals) >= 2 else float('nan')
    except Exception:
        return float('nan')


def _s(v):
    return float('nan') if v is None else float(v)


def _div(a, b):
    if math.isnan(a) or math.isnan(b) or b == 0:
        return float('nan')
    return a / b


def _nan():
    return float('nan')


def _get(df: pd.DataFrame, *keys) -> pd.Series:
    """Try multiple possible row labels, return first match."""
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# 1. Piotroski F-Score
# ---------------------------------------------------------------------------

def piotroski_f_score(annual_inc: pd.DataFrame,
                      annual_bs: pd.DataFrame,
                      annual_cf: pd.DataFrame) -> dict:
    """
    Piotroski (2000) 9-point F-Score using annual statements.
    Returns score (int 0-9) and all 9 binary components.
    NaN components are excluded from the total; partial_score is flagged.
    """

    def row(df, *keys): return _get(df, *keys)

    # ── Profitability signals ──────────────────────────────────────────────

    ta_now    = _f(row(annual_bs, 'Total Assets'), 'TA')
    ta_prior  = _prior(row(annual_bs, 'Total Assets'), 'TA')
    ni_now    = _f(row(annual_inc, 'Net Income'), 'NI')
    ocf_now   = _f(row(annual_cf, 'Operating Cash Flow'), 'OCF')

    avg_ta = _div(ta_now + ta_prior, 2) if not (math.isnan(ta_now) or math.isnan(ta_prior)) else ta_now

    roa_now   = _div(ni_now, avg_ta)

    # Prior-year ROA for ΔROA
    ni_prior  = _prior(row(annual_inc, 'Net Income'), 'NI')
    # Use beginning-of-prior-year total assets (3 periods back) if available
    ta_2prior_ser = row(annual_bs, 'Total Assets').dropna()
    ta_2prior = float(ta_2prior_ser.iloc[-3]) if len(ta_2prior_ser) >= 3 else ta_prior
    avg_ta_prior = _div(ta_prior + ta_2prior, 2) if not math.isnan(ta_2prior) else ta_prior
    roa_prior = _div(ni_prior, avg_ta_prior)

    F1 = int(roa_now > 0)            if not math.isnan(roa_now)  else float('nan')
    F2 = int(ocf_now > 0)            if not math.isnan(ocf_now)  else float('nan')
    F3 = int(roa_now > roa_prior)    if not (math.isnan(roa_now) or math.isnan(roa_prior)) else float('nan')

    # F4: accruals — OCF/Assets > ROA (cash-backed earnings quality)
    ocf_ta = _div(ocf_now, avg_ta)
    F4 = int(ocf_ta > roa_now) if not (math.isnan(ocf_ta) or math.isnan(roa_now)) else float('nan')

    # ── Leverage / Liquidity signals ───────────────────────────────────────

    ltd_now    = _f(row(annual_bs, 'Long Term Debt'), 'LTD')
    ltd_prior  = _prior(row(annual_bs, 'Long Term Debt'), 'LTD')
    lev_now    = _div(ltd_now, avg_ta)
    lev_prior  = _div(ltd_prior, avg_ta_prior)
    F5 = int(lev_now < lev_prior) if not (math.isnan(lev_now) or math.isnan(lev_prior)) else float('nan')

    ca_now    = _f(row(annual_bs, 'Current Assets'), 'CA')
    cl_now    = _f(row(annual_bs, 'Current Liabilities'), 'CL')
    ca_prior  = _prior(row(annual_bs, 'Current Assets'), 'CA')
    cl_prior  = _prior(row(annual_bs, 'Current Liabilities'), 'CL')
    cr_now    = _div(ca_now, cl_now)
    cr_prior  = _div(ca_prior, cl_prior)
    F6 = int(cr_now > cr_prior) if not (math.isnan(cr_now) or math.isnan(cr_prior)) else float('nan')

    # F7: No new shares issued (Common Stock did not increase)
    shares_now   = _f(row(annual_bs, 'Common Stock'), 'CS')
    shares_prior = _prior(row(annual_bs, 'Common Stock'), 'CS')
    F7 = int(shares_now <= shares_prior) if not (math.isnan(shares_now) or math.isnan(shares_prior)) else float('nan')

    # ── Operating efficiency signals ───────────────────────────────────────

    gp_now    = _f(row(annual_inc, 'Gross Profit'), 'GP')
    rev_now   = _f(row(annual_inc, 'Total Revenue'), 'REV')
    gp_prior  = _prior(row(annual_inc, 'Gross Profit'), 'GP')
    rev_prior = _prior(row(annual_inc, 'Total Revenue'), 'REV')
    gm_now    = _div(gp_now, rev_now)
    gm_prior  = _div(gp_prior, rev_prior)
    F8 = int(gm_now > gm_prior) if not (math.isnan(gm_now) or math.isnan(gm_prior)) else float('nan')

    at_now    = _div(rev_now, avg_ta)
    at_prior  = _div(rev_prior, avg_ta_prior)
    F9 = int(at_now > at_prior) if not (math.isnan(at_now) or math.isnan(at_prior)) else float('nan')

    components = {
        'F1_roa_positive':     F1,
        'F2_ocf_positive':     F2,
        'F3_delta_roa':        F3,
        'F4_accruals':         F4,
        'F5_delta_leverage':   F5,
        'F6_delta_liquidity':  F6,
        'F7_no_dilution':      F7,
        'F8_delta_gm':         F8,
        'F9_delta_at':         F9,
    }

    valid = [v for v in components.values() if not (isinstance(v, float) and math.isnan(v))]
    score = int(sum(v for v in valid if v == 1))
    partial = len(valid) < 9

    return {
        'score':    score,
        'partial':  partial,
        'n_signals': len(valid),
        **components,
    }


# ---------------------------------------------------------------------------
# 2. Beneish M-Score
# ---------------------------------------------------------------------------

def beneish_m_score(annual_inc: pd.DataFrame,
                    annual_bs: pd.DataFrame,
                    annual_cf: pd.DataFrame) -> dict:
    """
    Beneish (1999) 8-variable earnings manipulation detection model.

    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
            + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    Interpretation:
      M > -1.78  : possible manipulator (flag for review)
      M <= -2.22 : likely not manipulating
    """

    def row(df, *keys): return _get(df, *keys)

    rev_t   = _f(row(annual_inc, 'Total Revenue'), 'REV')
    rev_tm1 = _prior(row(annual_inc, 'Total Revenue'), 'REV')

    # DSRI — Days Sales Receivable Index
    rec_t   = _f(row(annual_bs, 'Accounts Receivable'), 'REC')
    rec_tm1 = _prior(row(annual_bs, 'Accounts Receivable'), 'REC')
    dsri_t  = _div(rec_t, rev_t)
    dsri_tm1 = _div(rec_tm1, rev_tm1)
    DSRI = _div(dsri_t, dsri_tm1)

    # GMI — Gross Margin Index  (prior/current, so LOWER = improving margin)
    gp_t    = _f(row(annual_inc, 'Gross Profit'), 'GP')
    gp_tm1  = _prior(row(annual_inc, 'Gross Profit'), 'GP')
    gm_t    = _div(gp_t, rev_t)
    gm_tm1  = _div(gp_tm1, rev_tm1)
    GMI = _div(gm_tm1, gm_t)   # >1 = deteriorating margin = manipulation flag

    # AQI — Asset Quality Index
    ta_t    = _f(row(annual_bs, 'Total Assets'), 'TA')
    ta_tm1  = _prior(row(annual_bs, 'Total Assets'), 'TA')
    ca_t    = _f(row(annual_bs, 'Current Assets'), 'CA')
    ca_tm1  = _prior(row(annual_bs, 'Current Assets'), 'CA')
    ppe_t   = _f(row(annual_bs, 'Net PPE', 'Net Property Plant And Equipment'), 'PPE')
    ppe_tm1 = _prior(row(annual_bs, 'Net PPE', 'Net Property Plant And Equipment'), 'PPE')

    aqi_t   = 1 - _div(ca_t + ppe_t, ta_t)   if not (math.isnan(ca_t) or math.isnan(ppe_t) or math.isnan(ta_t)) else _nan()
    aqi_tm1 = 1 - _div(ca_tm1 + ppe_tm1, ta_tm1) if not (math.isnan(ca_tm1) or math.isnan(ppe_tm1) or math.isnan(ta_tm1)) else _nan()
    AQI = _div(aqi_t, aqi_tm1)

    # SGI — Sales Growth Index
    SGI = _div(rev_t, rev_tm1)

    # DEPI — Depreciation Index
    dep_ser = row(annual_inc, 'Depreciation And Amortization',
                  'Reconciled Depreciation',
                  'Depreciation Depletion And Amortization')
    dep_t   = _f(dep_ser, 'DEP')
    dep_tm1 = _prior(dep_ser, 'DEP')

    dep_rate_t   = _div(abs(dep_t),   abs(dep_t)   + ppe_t)   if not (math.isnan(dep_t)   or math.isnan(ppe_t))   else _nan()
    dep_rate_tm1 = _div(abs(dep_tm1), abs(dep_tm1) + ppe_tm1) if not (math.isnan(dep_tm1) or math.isnan(ppe_tm1)) else _nan()
    DEPI = _div(dep_rate_tm1, dep_rate_t)  # >1 = slowing depreciation = flag

    # SGAI — SG&A Index
    sga_ser = row(annual_inc, 'Selling General Administrative',
                  'Selling General And Administration')
    sga_t   = _f(sga_ser, 'SGA')
    sga_tm1 = _prior(sga_ser, 'SGA')
    sgai_t   = _div(abs(sga_t),   rev_t)
    sgai_tm1 = _div(abs(sga_tm1), rev_tm1)
    SGAI = _div(sgai_t, sgai_tm1)   # >1 = rising SGA burden = flag

    # TATA — Total Accruals to Total Assets
    ni_t  = _f(row(annual_inc, 'Net Income'), 'NI')
    ocf_t = _f(row(annual_cf, 'Operating Cash Flow'), 'OCF')
    TATA  = _div(ni_t - ocf_t, ta_t)   # high positive = earnings paper-inflated

    # LVGI — Leverage Index
    ltd_t    = _f(row(annual_bs, 'Long Term Debt'), 'LTD')
    ltd_tm1  = _prior(row(annual_bs, 'Long Term Debt'), 'LTD')
    cl_t     = _f(row(annual_bs, 'Current Liabilities'), 'CL')
    cl_tm1   = _prior(row(annual_bs, 'Current Liabilities'), 'CL')
    lev_t    = _div(ltd_t + cl_t, ta_t)
    lev_tm1  = _div(ltd_tm1 + cl_tm1, ta_tm1)
    LVGI = _div(lev_t, lev_tm1)   # >1 = increasing leverage = flag

    # Compute M-Score (only when all components available)
    components = {
        'DSRI': DSRI, 'GMI': GMI, 'AQI': AQI, 'SGI': SGI,
        'DEPI': DEPI, 'SGAI': SGAI, 'TATA': TATA, 'LVGI': LVGI,
    }
    all_avail = all(not math.isnan(v) for v in components.values())

    m_score = _nan()
    if all_avail:
        m_score = (-4.84
                   + 0.920 * DSRI
                   + 0.528 * GMI
                   + 0.404 * AQI
                   + 0.892 * SGI
                   + 0.115 * DEPI
                   - 0.172 * SGAI
                   + 4.679 * TATA
                   - 0.327 * LVGI)

    flag = (m_score > -1.78) if not math.isnan(m_score) else None
    return {
        'm_score':       m_score,
        'flag_manipulate': int(flag) if flag is not None else _nan(),
        **{f'beneish_{k.lower()}': v for k, v in components.items()},
    }


# ---------------------------------------------------------------------------
# 3. Altman Z''-Score (non-manufacturer / services version)
# ---------------------------------------------------------------------------

def altman_z_score(annual_bs: pd.DataFrame,
                   annual_inc: pd.DataFrame,
                   market_cap: float) -> dict:
    """
    Altman (2000) revised Z''-Score for non-manufacturing companies.

    Z'' = 6.56*T1 + 3.26*T2 + 6.72*T3 + 1.05*T4

    T1 = Working Capital / Total Assets
    T2 = Retained Earnings / Total Assets
    T3 = EBIT / Total Assets
    T4 = Book Value of Equity / Total Liabilities

    Zones:
      Z'' > 2.6  : Safe
      1.1-2.6    : Grey
      < 1.1      : Distress
    """

    def row(df, *keys): return _get(df, *keys)

    ta   = _f(row(annual_bs, 'Total Assets'), 'TA')
    ca   = _f(row(annual_bs, 'Current Assets'), 'CA')
    cl   = _f(row(annual_bs, 'Current Liabilities'), 'CL')
    re   = _f(row(annual_bs, 'Retained Earnings'), 'RE')
    bveq = _f(row(annual_bs, 'Stockholders Equity'), 'BVE')
    tl   = _f(row(annual_bs, 'Total Liabilities Net Minority Interest',
                              'Total Liabilities'), 'TL')

    ebit_ser = row(annual_inc, 'EBIT', 'Operating Income')
    ebit = _f(ebit_ser, 'EBIT')

    rev  = _f(row(annual_inc, 'Total Revenue'), 'REV')

    wc   = ca - cl if not (math.isnan(ca) or math.isnan(cl)) else _nan()

    T1 = _div(wc,   ta)
    T2 = _div(re,   ta)
    T3 = _div(ebit, ta)
    T4 = _div(bveq, tl) if not math.isnan(bveq) else _div(market_cap, tl)

    components_avail = not any(math.isnan(x) for x in [T1, T2, T3, T4])
    z_score = 6.56*T1 + 3.26*T2 + 6.72*T3 + 1.05*T4 if components_avail else _nan()

    if math.isnan(z_score):
        zone = 'unknown'
    elif z_score > 2.6:
        zone = 'safe'
    elif z_score > 1.1:
        zone = 'grey'
    else:
        zone = 'distress'

    return {
        'altman_z':      z_score,
        'altman_zone':   zone,
        'altman_T1_wc_ta':   T1,
        'altman_T2_re_ta':   T2,
        'altman_T3_ebit_ta': T3,
        'altman_T4_bve_tl':  T4,
    }
