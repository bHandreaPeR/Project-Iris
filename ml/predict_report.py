"""
Prediction report generator for Project Iris.

Outputs a rich terminal report per ticker:
  ┌ SHORT  TERM (1 month)  : predicted return, price target, confidence band
  ├ MID    TERM (3 months) : same
  └ LONG   TERM (12 months): same

  Plus: financial health scores, key feature drivers, news sentiment,
  shareholding summary, backtest validation stats, best/worst backtest calls.

Also generates a ranked watchlist of top picks across all tickers.
"""

import math
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

from ml.multi_horizon import MultiHorizonModel, HORIZONS, TARGET_COLS

_OUTPUT_DIR = Path("ml_output")
_OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

def _signal(point: float, lower: float, horizon: str) -> str:
    """Map quantile predictions to a directional signal."""
    if math.isnan(point):
        return "UNKNOWN"
    if horizon == 'fwd_1m':
        if point > 0.08 and lower > 0.02:   return "STRONG BUY"
        if point > 0.04:                     return "BUY"
        if point > -0.02:                    return "HOLD"
        if point > -0.08:                    return "SELL"
        return "STRONG SELL"
    elif horizon == 'fwd_3m':
        if point > 0.15 and lower > 0.05:   return "STRONG BUY"
        if point > 0.07:                     return "BUY"
        if point > -0.03:                    return "HOLD"
        if point > -0.12:                    return "SELL"
        return "STRONG SELL"
    else:  # 12m
        if point > 0.30 and lower > 0.10:   return "STRONG BUY"
        if point > 0.15:                     return "BUY"
        if point > -0.05:                    return "HOLD"
        if point > -0.20:                    return "SELL"
        return "STRONG SELL"


def _confidence(lower: float, upper: float) -> str:
    if math.isnan(lower) or math.isnan(upper):
        return "Unknown"
    width = upper - lower
    if width < 0.15:   return "High"
    if width < 0.35:   return "Medium"
    return "Low"


def _fmt_pct(v: float, sign: bool = True) -> str:
    if math.isnan(v):
        return "N/A"
    s = f"{v:+.1%}" if sign else f"{v:.1%}"
    return s


def _fmt_price(price: float, ret: float) -> str:
    if price <= 0 or math.isnan(ret):
        return "N/A"
    return f"{price * (1 + ret):,.2f}"


def _zone_icon(zone: str) -> str:
    return {"safe": "🟢", "grey": "🟡", "distress": "🔴", "unknown": "⚪"}.get(zone, "⚪")


# ---------------------------------------------------------------------------
# Single-ticker report
# ---------------------------------------------------------------------------

def ticker_report(ticker: str,
                  model: MultiHorizonModel,
                  panel: pd.DataFrame,
                  current_feats: pd.Series,
                  current_price: float,
                  company_name: str = "",
                  sector: str = "",
                  market: str = "") -> str:

    preds = model.predict_ticker(current_feats)
    lines = []

    W = 65
    def _line(s=""): lines.append(s)
    def _bar(c="═"): lines.append(c * W)
    def _hdr(s): lines.append(f"  {s}")

    lines.append("╔" + "═" * (W - 2) + "╗")
    _hdr(f"PROJECT IRIS — {ticker}  {company_name}")
    _hdr(f"Sector: {sector or 'N/A':<20}  Market: {market or 'N/A'}")
    _hdr(f"Current Price: {current_price:,.2f}   |   Report Date: {date.today().isoformat()}")
    lines.append("╚" + "═" * (W - 2) + "╝")
    _line()

    horizon_labels = {
        'fwd_1m':  ("SHORT TERM", "1 Month"),
        'fwd_3m':  ("MID TERM",   "3 Months"),
        'fwd_12m': ("LONG TERM",  "12 Months"),
    }

    for target, (hname, hperiod) in horizon_labels.items():
        p = preds.get(target, {})
        point = p.get('point', float('nan'))
        lower = p.get('lower', float('nan'))
        upper = p.get('upper', float('nan'))
        sig   = _signal(point, lower, target)
        conf  = _confidence(lower, upper)

        _line(f"  ┌{'─'*(W-4)}┐")
        _line(f"  │  {hname} — {hperiod}{' '*(W-6-len(hname)-len(hperiod)-4)}│")
        _line(f"  ├{'─'*(W-4)}┤")
        _line(f"  │  Predicted Return   : {_fmt_pct(point):<10}  Signal: {sig:<12}│")
        _line(f"  │  Price Target       : {_fmt_price(current_price, point):<10}  Confidence: {conf:<8}│")
        lo_str = _fmt_price(current_price, lower)
        hi_str = _fmt_price(current_price, upper)
        _line(f"  │  Range (10th–90th)  : {lo_str} → {hi_str}{' '*(W-5-4-len(lo_str)-len(hi_str)-4)}│")
        _line(f"  └{'─'*(W-4)}┘")
        _line()

    # ── Financial health scores ──────────────────────────────────────────
    _bar("─")
    _hdr("FINANCIAL HEALTH SCORES")
    _bar("─")
    f = current_feats

    def _feat(name, default=float('nan')):
        return float(f.get(name, default) if hasattr(f, 'get') else getattr(f, name, default))

    pscore  = _feat('piotroski_score')
    bscore  = _feat('beneish_m')
    altman  = _feat('altman_z')
    altzone = "safe" if altman > 2.6 else ("grey" if altman > 1.1 else ("distress" if not math.isnan(altman) else "unknown"))

    pstr = f"{int(pscore)}/9" if not math.isnan(pscore) else "N/A"
    bstr = f"{bscore:.2f}" if not math.isnan(bscore) else "N/A"
    binterp = ("⚠ POSSIBLE MANIPULATION" if (not math.isnan(bscore) and bscore > -1.78) else
               "✓ CLEAN" if not math.isnan(bscore) else "N/A")
    astr = f"{altman:.2f}" if not math.isnan(altman) else "N/A"

    _hdr(f"Piotroski F-Score   : {pstr:<8}  (0=weak → 9=strong)")
    _hdr(f"Beneish M-Score     : {bstr:<8}  {binterp}")
    _hdr(f"Altman Z''-Score    : {astr:<8}  {_zone_icon(altzone)} {altzone.upper()} ZONE")
    _line()

    # ── Key features driving prediction ──────────────────────────────────
    _bar("─")
    _hdr("TOP FEATURE DRIVERS  (3-month model)")
    _bar("─")

    if 'fwd_3m' in model.models:
        fi = model.models['fwd_3m'].feature_importance().head(10)
        feat_vals = []
        for feat, imp in fi.items():
            val = _feat(feat)
            val_str = f"{val:.3f}" if not math.isnan(val) else "N/A"
            feat_vals.append((feat, val_str, imp))
        for feat, val, imp in feat_vals:
            _hdr(f"  {feat:<40} = {val:<10} (imp: {imp:.4f})")
    _line()

    # ── News sentiment ────────────────────────────────────────────────────
    direct_s = _feat('news_direct_score')
    sector_s = _feat('news_sector_score')
    macro_s  = _feat('news_macro_score')
    direct_n = int(_feat('news_direct_n')) if not math.isnan(_feat('news_direct_n')) else 0
    sector_n = int(_feat('news_sector_n')) if not math.isnan(_feat('news_sector_n')) else 0
    macro_n  = int(_feat('news_macro_n'))  if not math.isnan(_feat('news_macro_n'))  else 0

    if not all(math.isnan(x) for x in [direct_s, sector_s, macro_s]):
        _bar("─")
        _hdr("NEWS SENTIMENT (GDELT + FinBERT, last 30 days)")
        _bar("─")

        def _tone(v):
            if math.isnan(v): return "N/A"
            if v > 0.2:  return f"Positive ({v:+.2f})"
            if v < -0.2: return f"Negative ({v:+.2f})"
            return f"Neutral  ({v:+.2f})"

        _hdr(f"  Tier 1 Direct  : {_tone(direct_s):<30} ({direct_n} articles)")
        _hdr(f"  Tier 2 Sector  : {_tone(sector_s):<30} ({sector_n} articles)")
        _hdr(f"  Tier 3 Macro   : {_tone(macro_s):<30} ({macro_n} articles)")
        _line()

    # ── Shareholding ──────────────────────────────────────────────────────
    prom   = _feat('sh_promoter_pct')
    pledge = _feat('sh_promoter_pledge_pct')
    fii    = _feat('sh_fii_pct')
    dii    = _feat('sh_dii_pct')
    p_dq   = _feat('sh_promoter_delta_qoq')
    f_dq   = _feat('sh_fii_delta_qoq')

    if not all(math.isnan(x) for x in [prom, fii, dii]):
        _bar("─")
        _hdr("SHAREHOLDING PATTERN")
        _bar("─")
        prom_s   = f"{prom:.1f}%" if not math.isnan(prom) else "N/A"
        pledge_s = f"{pledge:.1f}%" if not math.isnan(pledge) else "N/A"
        fii_s    = f"{fii:.1f}%" if not math.isnan(fii) else "N/A"
        dii_s    = f"{dii:.1f}%" if not math.isnan(dii) else "N/A"
        p_dq_s   = f"{p_dq:+.1f}pp QoQ" if not math.isnan(p_dq) else ""
        f_dq_s   = f"{f_dq:+.1f}pp QoQ" if not math.isnan(f_dq) else ""

        _hdr(f"  Promoter        : {prom_s:<8}  Pledge: {pledge_s}  {p_dq_s}")
        _hdr(f"  FII             : {fii_s:<8}  {f_dq_s}")
        _hdr(f"  DII             : {dii_s}")
        _line()

    # ── Insider activity ──────────────────────────────────────────────────
    ins_net  = _feat('insider_net_value')
    ins_buy  = _feat('insider_n_buyers')
    ins_sell = _feat('insider_n_sellers')

    if not math.isnan(ins_net):
        _bar("─")
        _hdr("INSIDER ACTIVITY (last 180 days, SEC Form 4)")
        _bar("─")
        net_s = f"${ins_net:+,.0f}" if not math.isnan(ins_net) else "N/A"
        _hdr(f"  Net insider value   : {net_s}")
        _hdr(f"  Distinct buyers     : {int(ins_buy) if not math.isnan(ins_buy) else 'N/A'}")
        _hdr(f"  Distinct sellers    : {int(ins_sell) if not math.isnan(ins_sell) else 'N/A'}")
        _line()

    _bar("═")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtest examples
# ---------------------------------------------------------------------------

def backtest_examples(oof_df: pd.DataFrame,
                      panel: pd.DataFrame,
                      n_examples: int = 5) -> str:
    """
    Select the most illustrative OOF predictions (mix of hits and misses)
    and format them as human-readable examples.
    """
    if oof_df.empty:
        return "No OOF predictions available."

    df = oof_df[oof_df['horizon'] == 'fwd_3m'].dropna(subset=['pred_q50', 'actual'])
    if df.empty:
        return "No 3-month OOF predictions available."

    # Correct: prediction and actual both in same direction + large move
    df = df.copy()
    df['correct']      = (df['pred_q50'] * df['actual']) > 0
    df['pred_abs']     = df['pred_q50'].abs()
    df['actual_abs']   = df['actual'].abs()
    df['error']        = (df['pred_q50'] - df['actual']).abs()

    # Best calls: correct direction + large actual move
    best  = df[df['correct']].nlargest(n_examples, 'actual_abs')
    worst = df[~df['correct']].nlargest(min(2, len(df[~df['correct']])), 'error')

    lines = ["\n" + "═" * 65,
             "  BACKTEST EXAMPLES  (3-month horizon, out-of-fold)",
             "═" * 65]

    def _example(row, label):
        ticker = row['ticker']
        qe     = pd.Timestamp(row['quarter_end']).strftime('%Y-Q%q') if hasattr(pd.Timestamp(row['quarter_end']), 'strftime') else str(row['quarter_end'])[:7]
        pred   = f"{row['pred_q50']:+.1%}"
        actual = f"{row['actual']:+.1%}"
        hit    = "✓ HIT" if row['correct'] else "✗ MISS"
        return (
            f"  {label}\n"
            f"    Ticker: {ticker:<20}  Quarter: {qe}\n"
            f"    Predicted: {pred:<10}  Actual: {actual:<10}  {hit}"
        )

    lines.append("\n  ── Best calls ─────────────────────────────────────────")
    for _, row in best.iterrows():
        lines.append(_example(row, ""))

    if len(worst):
        lines.append("\n  ── Missed calls (learning from errors) ───────────────")
        for _, row in worst.iterrows():
            lines.append(_example(row, ""))

    lines.append("═" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ranked watchlist
# ---------------------------------------------------------------------------

def ranked_watchlist(ticker_preds: dict[str, dict],
                     prices: dict[str, float],
                     horizon: str = 'fwd_3m') -> pd.DataFrame:
    """
    Build a ranked watchlist from predictions across all tickers.

    Args:
        ticker_preds : {ticker: {horizon: {lower, point, upper}}}
        prices       : {ticker: current_price}
        horizon      : which horizon to rank by

    Returns ranked DataFrame.
    """
    rows = []
    for tkr, preds in ticker_preds.items():
        p = preds.get(horizon, {})
        point = p.get('point', float('nan'))
        lower = p.get('lower', float('nan'))
        upper = p.get('upper', float('nan'))
        price = prices.get(tkr, 0)
        rows.append({
            'ticker':      tkr,
            'pred_return': point,
            'lower_bound': lower,
            'upper_bound': upper,
            'price_target': price * (1 + point) if price > 0 and not math.isnan(point) else float('nan'),
            'signal':      _signal(point, lower, horizon),
            'confidence':  _confidence(lower, upper),
        })

    df = pd.DataFrame(rows).sort_values('pred_return', ascending=False).reset_index(drop=True)
    df.index += 1
    return df
