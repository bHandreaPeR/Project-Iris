"""
Price / volume anomaly screening.
"""

import pandas as pd
import numpy as np


def screen_price_volume(df_ohlcv: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Args:
        df_ohlcv : DataFrame indexed by (ticker, date) with OHLCV columns.
        params   : PRICE_VOLUME_FILTERS dict from config.py

    Returns:
        DataFrame of flagged tickers (one row per ticker) with anomaly details.
    """
    if df_ohlcv.empty:
        return pd.DataFrame()

    vol_mult   = params['volume_multiplier']
    near_hi    = params['near_52w_high_pct'] / 100
    near_lo    = params['near_52w_low_pct'] / 100
    min_gap    = params['min_gap_pct'] / 100
    top_n      = params['top_movers_n']
    min_avg_vol = params['min_avg_volume']

    records = []
    for tkr, grp in df_ohlcv.groupby(level='ticker'):
        grp = grp.sort_index(level='date')
        if len(grp) < 22:
            continue

        close  = grp['Close']
        volume = grp['Volume']
        today  = grp.iloc[-1]
        prev   = grp.iloc[-2]

        avg_vol_20 = volume.iloc[-21:-1].mean()
        if avg_vol_20 < min_avg_vol:
            continue

        hi_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
        lo_52w = close.iloc[-252:].min() if len(close) >= 252 else close.min()

        day_chg_pct  = (today['Close'] - prev['Close']) / prev['Close']
        gap_pct      = (today['Open']  - prev['Close']) / prev['Close']
        vol_ratio    = today['Volume'] / avg_vol_20 if avg_vol_20 > 0 else 0
        near_hi_flag = (hi_52w - today['Close']) / hi_52w <= near_hi
        near_lo_flag = (today['Close'] - lo_52w) / lo_52w <= near_lo
        vol_flag     = vol_ratio >= vol_mult
        gap_flag     = abs(gap_pct) >= min_gap

        if not (vol_flag or near_hi_flag or near_lo_flag or gap_flag):
            continue

        flags = []
        if vol_flag:
            flags.append(f"Vol {vol_ratio:.1f}×")
        if gap_flag:
            direction = 'GapUp' if gap_pct > 0 else 'GapDn'
            flags.append(f"{direction} {gap_pct*100:.1f}%")
        if near_hi_flag:
            flags.append(f"Near52wHi")
        if near_lo_flag:
            flags.append(f"Near52wLo")

        records.append({
            'ticker':       tkr,
            'close':        round(today['Close'], 2),
            'day_chg_pct':  round(day_chg_pct * 100, 2),
            'vol_ratio':    round(vol_ratio, 2),
            'gap_pct':      round(gap_pct * 100, 2),
            '52w_high':     round(hi_52w, 2),
            '52w_low':      round(lo_52w, 2),
            'flags':        ' | '.join(flags),
            'screen':       'price_volume',
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).set_index('ticker')

    # Sort by absolute day change, keep top_n movers
    result = result.reindex(
        result['day_chg_pct'].abs().nlargest(top_n).index
    )
    return result
