"""
Master orchestrator for Project Iris stock screener.

Flow per market:
  1. Load tickers from universe
  2. Fetch OHLCV + fundamentals
  3. Run fundamental screen
  4. Run price/volume screen
  5. Combine results (union or intersect — see config.ENGINE['combine_mode'])
  6. Format and broadcast Telegram watchlist
"""

import datetime
import pandas as pd

import config
from data import universe
from data import fetch_india, fetch_us
from screens.fundamentals import screen_fundamentals
from screens.price_volume import screen_price_volume
from alerts.telegram import broadcast


_FETCHERS = {
    'india': fetch_india,
    'us':    fetch_us,
}

_TICKERS = {
    'india': universe.INDIA_TICKERS,
    'us':    universe.US_TICKERS,
}

_MARKET_LABEL = {
    'india': 'India (NSE)',
    'us':    'US (NYSE/NASDAQ)',
}


def run(market: str) -> pd.DataFrame:
    """
    Run the full screener for one market. Returns the combined results DataFrame.
    """
    market = market.lower()
    if market not in _FETCHERS:
        raise ValueError(f"Unknown market: {market!r}. Use 'india' or 'us'.")

    tickers  = _TICKERS[market]
    fetcher  = _FETCHERS[market]
    label    = _MARKET_LABEL[market]
    today    = datetime.date.today().strftime('%a %d-%b-%Y')

    print(f"[iris] {label} — fetching {len(tickers)} tickers …")
    df_ohlcv = fetcher.fetch_ohlcv(tickers)
    df_fund  = fetcher.fetch_fundamentals(tickers)

    print(f"[iris] running fundamental screen …")
    fund_hits = screen_fundamentals(df_fund, config.FUNDAMENTAL_FILTERS)

    print(f"[iris] running price/volume screen …")
    pv_hits = screen_price_volume(df_ohlcv, config.PRICE_VOLUME_FILTERS)

    combined = _combine(fund_hits, pv_hits, config.ENGINE['combine_mode'])
    combined = combined.head(config.ENGINE['max_results_per_market'])

    msg = _format_message(label, today, combined, fund_hits, pv_hits)
    print(msg)

    if config.TELEGRAM['bot_token'] and config.TELEGRAM['chat_ids']:
        broadcast(config.TELEGRAM['bot_token'], config.TELEGRAM['chat_ids'], msg)
        print(f"[iris] alert sent to {len(config.TELEGRAM['chat_ids'])} chat(s).")
    else:
        print("[iris] TELEGRAM not configured — skipping alert.")

    return combined


def _combine(fund: pd.DataFrame, pv: pd.DataFrame, mode: str) -> pd.DataFrame:
    if fund.empty and pv.empty:
        return pd.DataFrame()
    if fund.empty:
        return pv
    if pv.empty:
        return fund

    if mode == 'intersect':
        common = fund.index.intersection(pv.index)
        return fund.loc[common]
    else:  # union
        extra_pv = pv.loc[pv.index.difference(fund.index), ['flags', 'screen']]
        return pd.concat([fund, extra_pv])


def _format_message(label: str, date: str, combined: pd.DataFrame,
                    fund: pd.DataFrame, pv: pd.DataFrame) -> str:
    lines = [
        f"<b>IRIS Screener — {label}</b>",
        f"<i>{date}</i>",
        f"",
        f"Fundamental hits : {len(fund)}",
        f"Price/vol hits   : {len(pv)}",
        f"Combined watchlist: {len(combined)}",
        f"",
    ]

    if combined.empty:
        lines.append("No stocks passed the screener today.")
        return '\n'.join(lines)

    for tkr, row in combined.iterrows():
        name  = row.get('name', tkr) or tkr
        flags = row.get('flags', '')
        chg   = row.get('day_chg_pct', '')
        chg_str = f" ({chg:+.2f}%)" if isinstance(chg, float) else ''
        lines.append(f"<b>{tkr}</b>{chg_str}  {name}")
        if flags:
            lines.append(f"  └ {flags}")

    return '\n'.join(lines)
