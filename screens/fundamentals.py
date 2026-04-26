"""
Fundamental screening: filter stocks by valuation and quality metrics.
"""

import pandas as pd


def screen_fundamentals(df_info: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Args:
        df_info : DataFrame indexed by ticker with columns
                  [pe, pb, debt_equity, roe, revenue_growth, ...]
        params  : FUNDAMENTAL_FILTERS dict from config.py

    Returns:
        Filtered DataFrame of tickers that pass all non-null checks,
        plus a 'flags' column listing which criteria they hit.
    """
    if df_info.empty:
        return pd.DataFrame()

    df = df_info.copy()
    df['flags'] = ''

    mask = pd.Series(True, index=df.index)

    pe_mask = df['pe'].isna() | (df['pe'] < params['max_pe'])
    pb_mask = df['pb'].isna() | (df['pb'] < params['max_pb'])
    de_mask = df['debt_equity'].isna() | (df['debt_equity'] < params['max_debt_equity'])
    roe_mask = df['roe'].isna() | (df['roe'] > params['min_roe'])
    rg_mask = df['revenue_growth'].isna() | (df['revenue_growth'] > params['min_revenue_growth'])

    mask = pe_mask & pb_mask & de_mask & roe_mask & rg_mask

    result = df[mask].copy()

    def _build_flags(row):
        parts = []
        if pd.notna(row.get('pe')):
            parts.append(f"P/E {row['pe']:.1f}")
        if pd.notna(row.get('pb')):
            parts.append(f"P/B {row['pb']:.2f}")
        if pd.notna(row.get('roe')):
            parts.append(f"ROE {row['roe']*100:.1f}%")
        if pd.notna(row.get('revenue_growth')):
            parts.append(f"RevGr {row['revenue_growth']*100:.1f}%")
        return ' | '.join(parts)

    result['flags'] = result.apply(_build_flags, axis=1)
    result['screen'] = 'fundamental'
    return result[['name', 'sector', 'market_cap', 'pe', 'pb', 'roe',
                   'revenue_growth', 'debt_equity', 'flags', 'screen']]
