"""
Feature engineering for the Project Iris regression model.

Given a raw statements dict (from data.financials.fetch_statements), this
module computes ~60 features organised into eight categories:

  1.  Valuation           (7)   — market price vs intrinsic value signals
  2.  Profitability       (9)   — margin quality and return ratios
  3.  Revenue growth     (6)   — top-line momentum and acceleration
  4.  Earnings growth    (5)   — EPS and operating income dynamics
  5.  Balance sheet      (8)   — solvency, liquidity, capital structure
  6.  Cash flow quality  (7)   — how real are the reported profits?
  7.  Income stmt detail (8)   — fine-grained line-item relationships
  8.  Momentum / price   (6)   — market's own signal embedded in price

All outputs are plain Python floats (NaN for unavailable data).
The caller (collector.py) assembles them into one row per (ticker, date).
"""

import math
import numpy as np
import pandas as pd
from data.financials import safe_get, trailing_sum, yoy_growth, qoq_growth


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nan() -> float:
    return float('nan')


def _safe_div(a: float, b: float) -> float:
    if b == 0 or math.isnan(b) or math.isnan(a):
        return _nan()
    return a / b


def _pct_return(price_series: pd.Series, start: pd.Timestamp,
                end: pd.Timestamp) -> float:
    """Compute price return between two dates using nearest available prices."""
    try:
        s = price_series.loc[:start].iloc[-1]
        e = price_series.loc[:end].iloc[-1]
        return _safe_div(e - s, abs(s))
    except (IndexError, KeyError):
        return _nan()


# ---------------------------------------------------------------------------
# 1. Valuation ratios
# ---------------------------------------------------------------------------

def valuation_features(info: dict, stmts: dict) -> dict:
    mkt_cap    = float(info.get('marketCap', 0) or 0)
    shares     = float(info.get('sharesOutstanding', 0) or 0)
    price      = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)
    book_val   = float(info.get('bookValue', 0) or 0)
    enterprise = float(info.get('enterpriseValue', 0) or 0)

    inc = stmts['income']
    cf  = stmts['cashflow']

    # TTM figures from quarterly statements
    rev_ttm    = trailing_sum(inc, 'Total Revenue')
    ebitda_ttm = trailing_sum(inc, 'EBITDA') if 'EBITDA' in inc.index else _nan()
    ni_ttm     = trailing_sum(inc, 'Net Income')
    ocf_ttm    = trailing_sum(cf,  'Operating Cash Flow')
    capex_ttm  = trailing_sum(cf,  'Capital Expenditure')
    fcf_ttm    = ocf_ttm - abs(capex_ttm) if not (math.isnan(ocf_ttm) or math.isnan(capex_ttm)) else _nan()

    pe         = float(info.get('trailingPE', _nan()) or _nan())
    pb         = _safe_div(price, book_val) if book_val else float(info.get('priceToBook', _nan()) or _nan())
    ps         = _safe_div(mkt_cap, rev_ttm)
    ev_ebitda  = _safe_div(enterprise, ebitda_ttm)
    ev_sales   = _safe_div(enterprise, rev_ttm)
    p_fcf      = _safe_div(mkt_cap, fcf_ttm) if not math.isnan(fcf_ttm) else _nan()
    earnings_yield = _safe_div(ni_ttm, mkt_cap)  # inverse P/E — better for regression

    return {
        'val_pe':           pe,
        'val_pb':           pb,
        'val_ps':           ps,
        'val_ev_ebitda':    ev_ebitda,
        'val_ev_sales':     ev_sales,
        'val_p_fcf':        p_fcf,
        'val_earnings_yield': earnings_yield,
    }


# ---------------------------------------------------------------------------
# 2. Profitability & returns
# ---------------------------------------------------------------------------

def profitability_features(info: dict, stmts: dict) -> dict:
    inc = stmts['income']
    bs  = stmts['balance']
    cf  = stmts['cashflow']

    rev_ttm  = trailing_sum(inc, 'Total Revenue')
    gp_ttm   = trailing_sum(inc, 'Gross Profit')
    ebit_ttm = trailing_sum(inc, 'EBIT') if 'EBIT' in inc.index else trailing_sum(inc, 'Operating Income')
    ni_ttm   = trailing_sum(inc, 'Net Income')
    ocf_ttm  = trailing_sum(cf,  'Operating Cash Flow')

    total_assets   = safe_get(bs, 'Total Assets', bs.columns[-1]) if not bs.empty else _nan()
    total_equity   = safe_get(bs, 'Stockholders Equity', bs.columns[-1]) if not bs.empty else _nan()
    invested_cap   = _nan()
    if not (math.isnan(total_assets) or math.isnan(total_equity)):
        total_debt = safe_get(bs, 'Long Term Debt', bs.columns[-1])
        total_debt = 0 if math.isnan(total_debt) else total_debt
        invested_cap = total_equity + total_debt

    gross_margin  = _safe_div(gp_ttm, rev_ttm)
    ebit_margin   = _safe_div(ebit_ttm, rev_ttm)
    net_margin    = _safe_div(ni_ttm, rev_ttm)
    ocf_margin    = _safe_div(ocf_ttm, rev_ttm)
    roe           = float(info.get('returnOnEquity', _nan()) or _nan())
    roa           = _safe_div(ni_ttm, total_assets)
    roic          = _safe_div(ebit_ttm * (1 - 0.25), invested_cap)  # tax-adjusted approx
    asset_turnover = _safe_div(rev_ttm, total_assets)
    equity_mult   = _safe_div(total_assets, total_equity)  # DuPont lever

    return {
        'prof_gross_margin':   gross_margin,
        'prof_ebit_margin':    ebit_margin,
        'prof_net_margin':     net_margin,
        'prof_ocf_margin':     ocf_margin,
        'prof_roe':            roe,
        'prof_roa':            roa,
        'prof_roic':           roic,
        'prof_asset_turnover': asset_turnover,
        'prof_equity_mult':    equity_mult,
    }


# ---------------------------------------------------------------------------
# 3. Revenue growth
# ---------------------------------------------------------------------------

def revenue_growth_features(stmts: dict) -> dict:
    inc = stmts['income']

    rev_yoy       = yoy_growth(inc, 'Total Revenue')
    rev_qoq       = qoq_growth(inc, 'Total Revenue')
    gp_yoy        = yoy_growth(inc, 'Gross Profit')

    # Acceleration: current QoQ vs prior QoQ (is growth speeding up?)
    rev_accel = _nan()
    if 'Total Revenue' in inc.index:
        vals = inc.loc['Total Revenue'].dropna()
        if len(vals) >= 4:
            qoq_now   = _safe_div(float(vals.iloc[-1]) - float(vals.iloc[-2]), abs(float(vals.iloc[-2])))
            qoq_prior = _safe_div(float(vals.iloc[-2]) - float(vals.iloc[-3]), abs(float(vals.iloc[-3])))
            rev_accel = qoq_now - qoq_prior  # positive = accelerating

    # 2-year CAGR (less volatile than single-year)
    rev_cagr2 = _nan()
    if 'Total Revenue' in inc.index:
        vals = inc.loc['Total Revenue'].dropna()
        if len(vals) >= 9:
            r_now  = float(vals.iloc[-4:].sum())   # TTM
            r_2y   = float(vals.iloc[-8:-4].sum())  # prior year
            rev_cagr2 = _safe_div(r_now - r_2y, abs(r_2y))

    # Revenue estimate beat proxy: QoQ vs trailing avg QoQ
    rev_beat_proxy = _nan()
    if 'Total Revenue' in inc.index:
        vals = inc.loc['Total Revenue'].dropna()
        if len(vals) >= 6:
            hist_qoq = [_safe_div(float(vals.iloc[i]) - float(vals.iloc[i-1]),
                                  abs(float(vals.iloc[i-1])))
                        for i in range(-5, -1)]
            hist_qoq = [x for x in hist_qoq if not math.isnan(x)]
            if hist_qoq:
                avg_hist = np.mean(hist_qoq)
                rev_beat_proxy = rev_qoq - avg_hist if not math.isnan(rev_qoq) else _nan()

    return {
        'rev_yoy':        rev_yoy,
        'rev_qoq':        rev_qoq,
        'rev_gp_yoy':     gp_yoy,
        'rev_accel':      rev_accel,
        'rev_cagr2':      rev_cagr2,
        'rev_beat_proxy': rev_beat_proxy,
    }


# ---------------------------------------------------------------------------
# 4. Earnings growth
# ---------------------------------------------------------------------------

def earnings_growth_features(stmts: dict) -> dict:
    inc = stmts['income']

    ni_yoy    = yoy_growth(inc, 'Net Income')
    ni_qoq    = qoq_growth(inc, 'Net Income')
    ebit_yoy  = yoy_growth(inc, 'EBIT') if 'EBIT' in inc.index else yoy_growth(inc, 'Operating Income')
    ebit_qoq  = qoq_growth(inc, 'EBIT') if 'EBIT' in inc.index else qoq_growth(inc, 'Operating Income')

    # Operating leverage: EBIT growth / Revenue growth (amplification factor)
    op_lev = _nan()
    if not (math.isnan(ebit_yoy) or math.isnan(yoy_growth(inc, 'Total Revenue'))):
        rev_yoy = yoy_growth(inc, 'Total Revenue')
        op_lev  = _safe_div(ebit_yoy, rev_yoy) if abs(rev_yoy) > 0.001 else _nan()

    return {
        'earn_ni_yoy':   ni_yoy,
        'earn_ni_qoq':   ni_qoq,
        'earn_ebit_yoy': ebit_yoy,
        'earn_ebit_qoq': ebit_qoq,
        'earn_op_lev':   op_lev,
    }


# ---------------------------------------------------------------------------
# 5. Balance sheet health
# ---------------------------------------------------------------------------

def balance_sheet_features(stmts: dict) -> dict:
    bs  = stmts['balance']
    inc = stmts['income']

    if bs.empty:
        return {k: _nan() for k in [
            'bs_current_ratio', 'bs_quick_ratio', 'bs_cash_ratio',
            'bs_debt_equity', 'bs_interest_coverage', 'bs_net_debt_ebitda',
            'bs_working_cap_pct_rev', 'bs_debt_trend',
        ]}

    col = bs.columns[-1]  # most recent quarter

    def g(row): return safe_get(bs, row, col)

    current_assets  = g('Current Assets')
    current_liab    = g('Current Liabilities')
    cash            = g('Cash And Cash Equivalents')
    receivables     = g('Accounts Receivable') if 'Accounts Receivable' in bs.index else 0.0
    receivables     = 0.0 if math.isnan(receivables) else receivables
    total_debt      = g('Total Debt')
    total_equity    = g('Stockholders Equity')
    total_assets    = g('Total Assets')
    inventory       = g('Inventory') if 'Inventory' in bs.index else 0.0
    inventory       = 0.0 if math.isnan(inventory) else inventory

    current_ratio   = _safe_div(current_assets, current_liab)
    quick_ratio     = _safe_div(current_assets - inventory, current_liab)
    cash_ratio      = _safe_div(cash, current_liab)
    debt_equity     = _safe_div(total_debt, total_equity)
    net_debt        = total_debt - cash if not (math.isnan(total_debt) or math.isnan(cash)) else _nan()

    ebit_ttm        = trailing_sum(inc, 'EBIT') if 'EBIT' in inc.index else trailing_sum(inc, 'Operating Income')
    interest_exp    = trailing_sum(inc, 'Interest Expense')
    interest_cov    = _safe_div(ebit_ttm, abs(interest_exp)) if not math.isnan(interest_exp) else _nan()

    ebitda_ttm      = trailing_sum(inc, 'EBITDA') if 'EBITDA' in inc.index else _nan()
    net_debt_ebitda = _safe_div(net_debt, ebitda_ttm) if not math.isnan(net_debt) else _nan()

    rev_ttm         = trailing_sum(inc, 'Total Revenue')
    working_cap     = current_assets - current_liab if not (math.isnan(current_assets) or math.isnan(current_liab)) else _nan()
    wc_pct_rev      = _safe_div(working_cap, rev_ttm)

    # Debt trend: is total debt growing faster than equity? (1=increasing, -1=decreasing)
    debt_trend = _nan()
    if 'Total Debt' in bs.index and len(bs.columns) >= 5:
        debts = bs.loc['Total Debt'].dropna()
        if len(debts) >= 2:
            debt_trend = float(debts.iloc[-1]) - float(debts.iloc[-2])
            debt_trend = 1.0 if debt_trend > 0 else -1.0

    return {
        'bs_current_ratio':      current_ratio,
        'bs_quick_ratio':        quick_ratio,
        'bs_cash_ratio':         cash_ratio,
        'bs_debt_equity':        debt_equity,
        'bs_interest_coverage':  interest_cov,
        'bs_net_debt_ebitda':    net_debt_ebitda,
        'bs_working_cap_pct_rev': wc_pct_rev,
        'bs_debt_trend':         debt_trend,
    }


# ---------------------------------------------------------------------------
# 6. Cash flow quality
# ---------------------------------------------------------------------------

def cashflow_quality_features(stmts: dict) -> dict:
    inc = stmts['income']
    cf  = stmts['cashflow']
    bs  = stmts['balance']

    ni_ttm    = trailing_sum(inc, 'Net Income')
    ocf_ttm   = trailing_sum(cf,  'Operating Cash Flow')
    capex_ttm = trailing_sum(cf,  'Capital Expenditure')
    rev_ttm   = trailing_sum(inc, 'Total Revenue')

    capex_ttm_abs = abs(capex_ttm) if not math.isnan(capex_ttm) else _nan()
    fcf_ttm   = ocf_ttm - capex_ttm_abs if not (math.isnan(ocf_ttm) or math.isnan(capex_ttm_abs)) else _nan()

    # Cash conversion: OCF / Net Income  (>1 = earnings are cash-backed)
    cash_conv  = _safe_div(ocf_ttm, ni_ttm)

    # FCF margin
    fcf_margin = _safe_div(fcf_ttm, rev_ttm)

    # Capex intensity
    capex_int  = _safe_div(capex_ttm_abs, rev_ttm)

    # Sloan accruals ratio: (Net Income - OCF) / Avg Total Assets
    # High positive accruals = earnings overstated relative to cash (bearish)
    sloan = _nan()
    if not (math.isnan(ni_ttm) or math.isnan(ocf_ttm)) and not bs.empty:
        avg_assets = bs.loc['Total Assets'].dropna().astype(float).mean() if 'Total Assets' in bs.index else _nan()
        sloan = _safe_div(ni_ttm - ocf_ttm, avg_assets)

    # FCF growth YoY (proxy: OCF YoY since capex data can be noisy)
    ocf_yoy = yoy_growth(cf, 'Operating Cash Flow')

    # Dividend coverage: FCF / Dividends paid
    div_cov = _nan()
    if 'Common Stock Dividend Paid' in cf.index:
        div_ttm = abs(trailing_sum(cf, 'Common Stock Dividend Paid'))
        div_cov = _safe_div(fcf_ttm, div_ttm)

    return {
        'cfq_cash_conversion': cash_conv,
        'cfq_fcf_margin':      fcf_margin,
        'cfq_capex_intensity': capex_int,
        'cfq_sloan_accruals':  sloan,
        'cfq_ocf_yoy':         ocf_yoy,
        'cfq_div_coverage':    div_cov,
        'cfq_fcf_ttm':         fcf_ttm,   # absolute (normalised later)
    }


# ---------------------------------------------------------------------------
# 7. Income statement fine detail
# ---------------------------------------------------------------------------

def income_detail_features(stmts: dict) -> dict:
    inc = stmts['income']

    rev_ttm  = trailing_sum(inc, 'Total Revenue')

    # SG&A as % of revenue (rising = operating deleverage)
    sga_pct = _nan()
    for lbl in ['Selling General Administrative', 'Selling General And Administration']:
        if lbl in inc.index:
            sga_pct = _safe_div(abs(trailing_sum(inc, lbl)), rev_ttm)
            break

    # R&D as % of revenue
    rd_pct = _nan()
    for lbl in ['Research And Development', 'Research Development']:
        if lbl in inc.index:
            rd_pct = _safe_div(abs(trailing_sum(inc, lbl)), rev_ttm)
            break

    # Depreciation & amortisation as % of revenue (capital intensity proxy)
    da_pct = _nan()
    for lbl in ['Depreciation And Amortization', 'Reconciled Depreciation']:
        if lbl in inc.index:
            da_pct = _safe_div(abs(trailing_sum(inc, lbl)), rev_ttm)
            break

    # Gross margin expansion: current quarter vs 4 quarters ago
    gm_expansion = _nan()
    if 'Gross Profit' in inc.index and 'Total Revenue' in inc.index:
        gp  = inc.loc['Gross Profit'].dropna()
        rev = inc.loc['Total Revenue'].dropna()
        if len(gp) >= 5 and len(rev) >= 5:
            gm_now   = _safe_div(float(gp.iloc[-1]),  float(rev.iloc[-1]))
            gm_prior = _safe_div(float(gp.iloc[-5]),  float(rev.iloc[-5]))
            gm_expansion = gm_now - gm_prior

    # EBIT margin expansion vs prior year quarter
    ebit_margin_exp = _nan()
    ebit_lbl = 'EBIT' if 'EBIT' in inc.index else 'Operating Income'
    if ebit_lbl in inc.index and 'Total Revenue' in inc.index:
        eb  = inc.loc[ebit_lbl].dropna()
        rev = inc.loc['Total Revenue'].dropna()
        if len(eb) >= 5 and len(rev) >= 5:
            em_now   = _safe_div(float(eb.iloc[-1]),  float(rev.iloc[-1]))
            em_prior = _safe_div(float(eb.iloc[-5]),  float(rev.iloc[-5]))
            ebit_margin_exp = em_now - em_prior

    # COGS trend: rising COGS as % of revenue = margin pressure
    cogs_pct_rev = _nan()
    for lbl in ['Cost Of Revenue', 'Cost Of Goods And Services Sold']:
        if lbl in inc.index:
            cogs_pct_rev = _safe_div(abs(trailing_sum(inc, lbl)), rev_ttm)
            break

    # Tax rate trend (unusually low = one-time benefit, likely to reverse)
    tax_rate = _nan()
    for lbl in ['Tax Provision', 'Income Tax Expense']:
        if lbl in inc.index:
            pretax = trailing_sum(inc, 'Pretax Income') if 'Pretax Income' in inc.index else _nan()
            if not math.isnan(pretax) and pretax != 0:
                tax_rate = _safe_div(trailing_sum(inc, lbl), pretax)
            break

    # Non-operating income as % of net income (persistent vs one-time)
    nonop_pct = _nan()
    ni_ttm = trailing_sum(inc, 'Net Income')
    ebit_ttm = trailing_sum(inc, ebit_lbl)
    pretax_ttm = trailing_sum(inc, 'Pretax Income') if 'Pretax Income' in inc.index else _nan()
    if not (math.isnan(ebit_ttm) or math.isnan(pretax_ttm)):
        nonop_income = pretax_ttm - ebit_ttm
        nonop_pct = _safe_div(nonop_income, abs(ni_ttm))

    return {
        'inc_sga_pct_rev':      sga_pct,
        'inc_rd_pct_rev':       rd_pct,
        'inc_da_pct_rev':       da_pct,
        'inc_cogs_pct_rev':     cogs_pct_rev,
        'inc_gm_expansion':     gm_expansion,
        'inc_ebit_margin_exp':  ebit_margin_exp,
        'inc_tax_rate':         tax_rate,
        'inc_nonop_pct_ni':     nonop_pct,
    }


# ---------------------------------------------------------------------------
# 8. Momentum & price-derived features
# ---------------------------------------------------------------------------

def momentum_features(price_hist: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """Compute price-based momentum features as of a given date."""
    if price_hist.empty:
        return {k: _nan() for k in [
            'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m',
            'mom_vol_20d', 'mom_vol_adj_12m',
        ]}

    ph = price_hist['Close'].sort_index()

    def ret(days):
        try:
            end   = ph.loc[:as_of].iloc[-1]
            start = ph.loc[:as_of - pd.DateOffset(days=days)].iloc[-1]
            return _safe_div(end - start, abs(start))
        except IndexError:
            return _nan()

    m1  = ret(30)
    m3  = ret(91)
    m6  = ret(182)
    m12 = ret(365)

    # 20-day realised volatility (annualised)
    vol_20d = _nan()
    try:
        slice_ = ph.loc[:as_of].iloc[-21:]
        if len(slice_) >= 5:
            log_rets = np.log(slice_ / slice_.shift(1)).dropna()
            vol_20d  = float(log_rets.std() * math.sqrt(252))
    except Exception:
        pass

    # Vol-adjusted 12m momentum (Sharpe-like signal)
    vol_adj_12m = _safe_div(m12, vol_20d) if not math.isnan(vol_20d) else _nan()

    return {
        'mom_1m':         m1,
        'mom_3m':         m3,
        'mom_6m':         m6,
        'mom_12m':        m12,
        'mom_vol_20d':    vol_20d,
        'mom_vol_adj_12m': vol_adj_12m,
    }


# ---------------------------------------------------------------------------
# Master: compute all features for one (ticker, date) snapshot
# ---------------------------------------------------------------------------

def compute_all(stmts: dict, as_of: pd.Timestamp) -> dict:
    """
    Returns a flat dict of all features for a single snapshot.
    All values are floats (NaN if data unavailable).
    """
    info = stmts.get('info', {})
    feats = {}
    feats.update(valuation_features(info, stmts))
    feats.update(profitability_features(info, stmts))
    feats.update(revenue_growth_features(stmts))
    feats.update(earnings_growth_features(stmts))
    feats.update(balance_sheet_features(stmts))
    feats.update(cashflow_quality_features(stmts))
    feats.update(income_detail_features(stmts))
    feats.update(momentum_features(stmts.get('price_hist', pd.DataFrame()), as_of))
    return feats


FEATURE_NAMES = list(compute_all(
    {'income': pd.DataFrame(), 'balance': pd.DataFrame(),
     'cashflow': pd.DataFrame(), 'price_hist': pd.DataFrame(), 'info': {}},
    pd.Timestamp('2020-01-01')
).keys())
