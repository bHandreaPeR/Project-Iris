"""
Plain-English reasoning generator for IRIS ML predictions.

For each ticker, produces exactly 3 one-line reasons that explain the
model's directional call, drawn from the stock's actual feature values.
"""

import math
import pandas as pd


# ── Feature → sentence templates ──────────────────────────────────────────────

def _pct(v, decimals=1):
    return f"{v*100:.{decimals}f}%"

def _na(v):
    return math.isnan(float(v)) if isinstance(v, (int, float)) else True


def _revenue_growth_line(f):
    yoy = f.get('rev_yoy', float('nan'))
    qoq = f.get('rev_qoq', float('nan'))
    if not _na(yoy):
        adj = "surging" if yoy > 0.30 else ("strong" if yoy > 0.15 else ("solid" if yoy > 0.05 else ("flat" if yoy > -0.05 else "declining")))
        extra = f"; QoQ {_pct(qoq)}" if not _na(qoq) else ""
        return f"Revenue is {adj} at {_pct(yoy)} YoY{extra}."
    return None


def _earnings_growth_line(f):
    ni  = f.get('earn_ni_yoy', float('nan'))
    ebit = f.get('earn_ebit_yoy', float('nan'))
    if not _na(ni):
        tone = "accelerating" if ni > 0.40 else ("growing" if ni > 0.10 else ("contracting" if ni < -0.10 else "stable"))
        ebit_s = f", EBIT {_pct(ebit)} YoY" if not _na(ebit) else ""
        return f"Net income {tone} at {_pct(ni)} YoY{ebit_s}."
    return None


def _margin_line(f):
    gm_exp = f.get('inc_gm_expansion', float('nan'))
    ebit_exp = f.get('inc_ebit_margin_exp', float('nan'))
    ocf_m = f.get('prof_ocf_margin', float('nan'))
    if not _na(gm_exp):
        dir_gm = "expanding" if gm_exp > 0.01 else ("compressing" if gm_exp < -0.01 else "stable")
        dir_ebit = ""
        if not _na(ebit_exp):
            dir_ebit = f", EBIT margin {_pct(ebit_exp, 2)} YoY"
        ocf_s = f"; OCF margin {_pct(ocf_m)}" if not _na(ocf_m) else ""
        return f"Gross margin {dir_gm} ({_pct(gm_exp, 2)} YoY){dir_ebit}{ocf_s}."
    return None


def _piotroski_line(f):
    score = f.get('piotroski_score', float('nan'))
    if not _na(score):
        sc = int(score)
        grade = "excellent (8-9/9)" if sc >= 8 else ("strong (6-7/9)" if sc >= 6 else ("weak (3-5/9)" if sc >= 3 else "very weak (<3/9)"))
        f1 = f.get('piotroski_F1_roa_positive', float('nan'))
        f2 = f.get('piotroski_F2_ocf_positive', float('nan'))
        extras = []
        if not _na(f1) and f1 == 0: extras.append("ROA negative")
        if not _na(f2) and f2 == 0: extras.append("OCF negative")
        tail = f" — {', '.join(extras)}" if extras else ""
        return f"Piotroski F-Score {sc}/9 ({grade}){tail}."
    return None


def _valuation_line(f):
    ey  = f.get('val_earnings_yield', float('nan'))
    pe  = f.get('val_pe', float('nan'))
    pb  = f.get('val_pb', float('nan'))
    ev  = f.get('val_ev_ebitda', float('nan'))
    if not _na(ey):
        tone = "attractively valued" if ey > 0.06 else ("fairly valued" if ey > 0.03 else "expensive")
        pe_s = f"P/E {pe:.1f}x" if not _na(pe) and pe > 0 else ""
        pb_s = f"P/B {pb:.1f}x" if not _na(pb) else ""
        parts = [s for s in [pe_s, pb_s] if s]
        stats = f" ({', '.join(parts)})" if parts else ""
        return f"Valuation {tone} at earnings yield {_pct(ey)}{stats}."
    return None


def _fcf_line(f):
    fcf = f.get('cfq_fcf_margin', float('nan'))
    accruals = f.get('cfq_sloan_accruals', float('nan'))
    capex = f.get('cfq_capex_intensity', float('nan'))
    if not _na(fcf):
        tone = "strong" if fcf > 0.15 else ("positive" if fcf > 0.05 else ("thin" if fcf > 0 else "negative"))
        acc_s = ""
        if not _na(accruals):
            acc_s = "; clean accruals" if accruals < 0.05 else "; elevated accruals (earnings quality risk)"
        return f"Free cash flow margin {_pct(fcf)} ({tone}){acc_s}."
    return None


def _debt_line(f):
    de   = f.get('bs_debt_equity', float('nan'))
    cr   = f.get('bs_current_ratio', float('nan'))
    nd_eb = f.get('bs_net_debt_ebitda', float('nan'))
    if not _na(de):
        lev = "low leverage" if de < 0.3 else ("moderate leverage" if de < 1.0 else ("high leverage" if de < 2.5 else "very high leverage"))
        cr_s = f", current ratio {cr:.2f}" if not _na(cr) else ""
        nd_s = f", net debt/EBITDA {nd_eb:.1f}x" if not _na(nd_eb) else ""
        return f"Balance sheet: {lev} (D/E {de:.2f}x{nd_eb and f', ND/EBITDA {nd_eb:.1f}x' or ''}{cr_s})."
    return None


def _analyst_line(f):
    upside = f.get('analyst_upside', float('nan'))
    rec    = f.get('analyst_rec_score', float('nan'))
    n      = f.get('analyst_n', float('nan'))
    if not _na(upside):
        tone = "strong consensus buy" if upside > 0.25 else ("consensus buy" if upside > 0.10 else ("consensus hold" if upside > -0.05 else "consensus underperform"))
        n_s  = f" from {int(n)} analysts" if not _na(n) and n > 0 else ""
        return f"Analyst {tone}{n_s} with {_pct(upside)} upside to consensus target."
    return None


def _short_line(f):
    short = f.get('short_pct_float', float('nan'))
    if not _na(short) and short > 0:
        tone = "very high short interest" if short > 0.10 else ("elevated short interest" if short > 0.05 else "low short interest")
        direction = "contrarian bullish signal" if short > 0.08 else ("potential squeeze catalyst" if short > 0.05 else "clean float")
        return f"{tone.capitalize()} at {_pct(short)} of float — {direction}."
    return None


def _momentum_line(f):
    m12 = f.get('mom_12m', float('nan'))
    m3  = f.get('mom_3m', float('nan'))
    if not _na(m12):
        tone = "strong momentum" if m12 > 0.30 else ("positive momentum" if m12 > 0.10 else ("neutral" if m12 > -0.10 else "negative momentum"))
        m3_s = f"; 3m: {_pct(m3)}" if not _na(m3) else ""
        return f"Price momentum: {tone} (12m {_pct(m12)}{m3_s})."
    return None


def _altman_line(f):
    z = f.get('altman_z', float('nan'))
    if not _na(z):
        zone = "safe" if z > 2.6 else ("grey" if z > 1.1 else "distress")
        return f"Altman Z''-Score {z:.2f} — {zone} zone{'.' if zone == 'safe' else (' (monitor).' if zone == 'grey' else ' — financial stress risk.')}"
    return None


def _shareholding_line(f):
    prom  = f.get('sh_promoter_pct', float('nan'))
    fii   = f.get('sh_fii_pct', float('nan'))
    p_dq  = f.get('sh_promoter_delta_qoq', float('nan'))
    f_dq  = f.get('sh_fii_delta_qoq', float('nan'))
    if not _na(prom) or not _na(fii):
        parts = []
        if not _na(prom):
            parts.append(f"Promoter {prom:.1f}%")
            if not _na(p_dq) and abs(p_dq) > 0.5:
                parts[-1] += f" ({p_dq:+.1f}pp QoQ)"
        if not _na(fii):
            parts.append(f"FII {fii:.1f}%")
            if not _na(f_dq) and abs(f_dq) > 0.5:
                parts[-1] += f" ({f_dq:+.1f}pp QoQ)"
        return f"Shareholding: {', '.join(parts)}."
    return None


# ── Priority-ordered candidate generators ─────────────────────────────────────

_GENERATORS = [
    _analyst_line,
    _earnings_growth_line,
    _revenue_growth_line,
    _margin_line,
    _piotroski_line,
    _valuation_line,
    _fcf_line,
    _debt_line,
    _shareholding_line,
    _momentum_line,
    _short_line,
    _altman_line,
]


def generate_reasoning(feats: pd.Series, signal: str, n_lines: int = 3) -> list[str]:
    """
    Return exactly n_lines plain-English sentences explaining the prediction.
    Picks the most informative available features in priority order.
    """
    f = feats.to_dict() if hasattr(feats, 'to_dict') else feats
    lines = []
    for gen in _GENERATORS:
        if len(lines) >= n_lines:
            break
        try:
            line = gen(f)
            if line:
                lines.append(line)
        except Exception:
            pass
    while len(lines) < n_lines:
        lines.append("Insufficient data for additional reasoning.")
    return lines[:n_lines]
