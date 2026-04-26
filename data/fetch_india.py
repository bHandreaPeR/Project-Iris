"""
Fetch OHLCV and fundamental data for Indian (NSE) stocks via yfinance.
"""

import time
import pandas as pd
import yfinance as yf


_CHUNK = 40  # tickers per batch download call


def fetch_ohlcv(tickers: list[str], period: str = '1y') -> pd.DataFrame:
    """
    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by (ticker, date). Batched to avoid yfinance rate limits.
    """
    frames = []
    for i in range(0, len(tickers), _CHUNK):
        chunk = tickers[i: i + _CHUNK]
        raw = yf.download(chunk, period=period, interval='1d',
                          group_by='ticker', auto_adjust=True, progress=False)
        if len(chunk) == 1:
            raw.columns = pd.MultiIndex.from_product([chunk, raw.columns])
        for tkr in chunk:
            try:
                df = raw[tkr].dropna(how='all').copy()
                df['ticker'] = tkr
                frames.append(df)
            except KeyError:
                pass
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    combined.index.name = 'date'
    return combined.reset_index().set_index(['ticker', 'date'])


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by ticker with fundamental fields:
    pe, pb, debt_equity, roe, revenue_growth, market_cap, sector.
    Missing values are NaN — the screen will skip those fields.
    """
    records = []
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).info
            records.append({
                'ticker':         tkr,
                'pe':             info.get('trailingPE'),
                'pb':             info.get('priceToBook'),
                'debt_equity':    info.get('debtToEquity'),
                'roe':            info.get('returnOnEquity'),
                'revenue_growth': info.get('revenueGrowth'),
                'market_cap':     info.get('marketCap'),
                'sector':         info.get('sector', ''),
                'name':           info.get('longName', tkr),
            })
        except Exception:
            pass
        time.sleep(0.2)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index('ticker')
