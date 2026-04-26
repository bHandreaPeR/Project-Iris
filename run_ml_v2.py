"""
Project Iris — ML pipeline v2 entry point.

Usage:
  # Full pipeline (collect → train → explain → predict + alert):
  python run_ml_v2.py --market us --collect --train --predict --alert

  # Fast rerun with cached panel:
  python run_ml_v2.py --market us --train --predict

  # Only show today's top picks (requires saved model):
  python run_ml_v2.py --market us --predict

  # India with shareholding data:
  python run_ml_v2.py --market india --collect --train --predict

  # Full universe + news (slow, ~3–6 hours first run):
  python run_ml_v2.py --market all --collect --train --predict --news --alert

Markets: us | india | all
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import numpy as np

import config
from data.universe import get_universe
from ml.collector_v2 import build_panel, panel_summary
from ml.multi_horizon import MultiHorizonModel, TARGET_COLS
from ml.predict_report import ticker_report, backtest_examples, ranked_watchlist
from ml.explain import full_report as explain_report

_OUT = Path("ml_output")
_OUT.mkdir(exist_ok=True)


def _cache_path(market: str) -> str:
    return str(_OUT / f"v2_panel_{market}.parquet")


def _model_path(market: str) -> Path:
    return _OUT / f"v2_model_{market}.pkl"


def _save_model(market: str, model: MultiHorizonModel):
    with open(_model_path(market), 'wb') as f:
        pickle.dump(model, f)
    print(f"[iris] Model saved → {_model_path(market)}")


def _load_model(market: str) -> MultiHorizonModel:
    p = _model_path(market)
    if not p.exists():
        sys.exit(f"[iris] No model at {p}. Run with --train first.")
    with open(p, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_collect(market: str, tickers: list[str],
                 use_news: bool = False) -> pd.DataFrame:
    print(f"\n[iris] Collecting data for {len(tickers)} {market} tickers …")
    panel = build_panel(
        tickers,
        cache_path=_cache_path(market),
        use_shareholding=(market in ('india', 'all')),
        use_insiders=(market in ('us', 'all')),
        use_news=use_news,
        verbose=True,
    )
    panel_summary(panel)
    return panel


def step_train(market: str, panel: pd.DataFrame) -> MultiHorizonModel:
    print(f"\n[iris] Training multi-horizon model ({len(panel)} rows) …")
    model = MultiHorizonModel()
    metrics = model.train(panel)
    print(model.walk_forward_summary(metrics))

    # Feature importance table
    fi = model.top_feature_importance(n=15)
    print("\n[iris] Top features by horizon:")
    for horizon in TARGET_COLS:
        sub = fi[fi['horizon'] == horizon].head(10)
        print(f"\n  {horizon}:")
        for _, row in sub.iterrows():
            print(f"    {row['feature']:<45} {row['importance']:.4f}")

    _save_model(market, model)
    return model


def step_explain(model: MultiHorizonModel, panel: pd.DataFrame):
    feat_cols = [c for c in panel.columns
                 if c not in TARGET_COLS + ['entry_date']]
    oof_df = model.oof_df if hasattr(model, 'oof_df') else pd.DataFrame()

    if '3m' in model.models or 'fwd_3m' in model.models:
        m_key = 'fwd_3m' if 'fwd_3m' in model.models else '3m'
        xgb_model   = model.models[m_key].q50
        preprocessor = None  # XGBoost needs no preprocessor (native NaN)

        if xgb_model:
            explain_report(oof_df, xgb_model, preprocessor, panel, feat_cols)


def step_predict(market: str,
                 panel: pd.DataFrame,
                 model: MultiHorizonModel,
                 top_n: int = 20) -> tuple[pd.DataFrame, dict]:
    """
    Generate predictions for all tickers in the most recent panel quarter.
    Returns (watchlist_df, raw_preds_dict).
    """
    latest_q = panel.index.get_level_values('quarter_end').max()
    current  = panel.loc[panel.index.get_level_values('quarter_end') == latest_q]

    feat_cols = model.feat_cols
    ticker_preds: dict[str, dict] = {}
    prices: dict[str, float] = {}

    # Get current prices from yfinance for all tickers
    tickers = current.index.get_level_values('ticker').tolist()
    try:
        import yfinance as yf
        price_data = yf.download(tickers[:200], period='2d',
                                 interval='1d', auto_adjust=True,
                                 progress=False)['Close']
        price_map = {}
        if isinstance(price_data, pd.Series):
            price_map[tickers[0]] = float(price_data.iloc[-1])
        else:
            for tkr in price_data.columns:
                vals = price_data[tkr].dropna()
                if len(vals):
                    price_map[tkr] = float(vals.iloc[-1])
    except Exception:
        price_map = {}

    for tkr, row in current.groupby(level='ticker'):
        feats = row.iloc[0][feat_cols]
        try:
            preds = model.predict_ticker(feats)
            ticker_preds[tkr] = preds
            prices[tkr] = price_map.get(tkr, 0)
        except Exception:
            pass

    watchlist = ranked_watchlist(ticker_preds, prices, horizon='fwd_3m')
    print(f"\n[iris] TOP {top_n} PREDICTED PICKS ({market.upper()}) — 3-month horizon")
    print("=" * 75)
    display = watchlist.head(top_n)[
        ['ticker', 'pred_return', 'price_target', 'signal', 'confidence']
    ].copy()
    display['pred_return']  = display['pred_return'].map(lambda x: f"{x:+.1%}" if not pd.isna(x) else "N/A")
    display['price_target'] = display['price_target'].map(lambda x: f"{x:,.2f}" if not pd.isna(x) else "N/A")
    print(display.to_string(index=True))

    # Detailed report for top 3 tickers
    print("\n" + "=" * 65)
    print("  DETAILED REPORTS — TOP 3 PICKS")
    print("=" * 65)
    for tkr in watchlist['ticker'].head(3):
        try:
            tkr_row = current.loc[tkr].iloc[-1]
            feats   = tkr_row.reindex(feat_cols)
            price   = prices.get(tkr, 0)
            info    = {}  # live info not re-fetched here
            report  = ticker_report(
                ticker=tkr,
                model=model,
                panel=panel,
                current_feats=feats,
                current_price=price,
                market=market,
            )
            print(report)
        except Exception as e:
            print(f"  [report] {tkr}: {e}")

    # Backtest examples
    if hasattr(model, 'oof_df') and not model.oof_df.empty:
        print(backtest_examples(model.oof_df, panel))

    # Save watchlist
    watchlist.to_csv(_OUT / f"watchlist_{market}.csv", index=True)
    print(f"\n[iris] Watchlist saved → {_OUT}/watchlist_{market}.csv")

    return watchlist, ticker_preds


def step_alert(market: str, watchlist: pd.DataFrame, top_n: int = 15):
    from alerts.telegram import broadcast
    if not config.TELEGRAM['bot_token'] or not config.TELEGRAM['chat_ids']:
        print("[iris] Telegram not configured — skipping alert.")
        return

    from datetime import date
    date_str = date.today().strftime('%d-%b-%Y')
    label = {'india': 'India (NSE)', 'us': 'US (NYSE/NASDAQ)', 'all': 'All Markets'}.get(market, market)

    lines = [
        f"<b>IRIS ML Watchlist — {label}</b>",
        f"<i>{date_str}  |  3-month predictions</i>",
        "",
    ]
    for rank, row in watchlist.head(top_n).iterrows():
        tkr    = row['ticker']
        ret    = row['pred_return']
        sig    = row['signal']
        conf   = row['confidence']
        ret_s  = f"{ret:+.1%}" if not pd.isna(ret) else "N/A"
        lines.append(f"<b>{rank}. {tkr}</b>  {ret_s}  [{sig}] ({conf})")

    lines += ["", "<i>Not financial advice. Model predictions only.</i>"]
    msg = "\n".join(lines)
    broadcast(config.TELEGRAM['bot_token'], config.TELEGRAM['chat_ids'], msg)
    print(f"[iris] Alert sent to {len(config.TELEGRAM['chat_ids'])} chat(s).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Project Iris v2 — ML Pipeline')
    parser.add_argument('--market',   choices=['india', 'us', 'all'], default='us')
    parser.add_argument('--collect',  action='store_true')
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--explain',  action='store_true')
    parser.add_argument('--predict',  action='store_true')
    parser.add_argument('--alert',    action='store_true')
    parser.add_argument('--news',     action='store_true',
                        help='Include GDELT + FinBERT sentiment (slow)')
    parser.add_argument('--top-n',    type=int, default=20)
    args = parser.parse_args()

    markets = ['india', 'us'] if args.market == 'all' else [args.market]

    for market in markets:
        print(f"\n{'═'*65}")
        print(f"  PROJECT IRIS v2  |  Market: {market.upper()}")
        print(f"{'═'*65}")

        tickers = get_universe(market)
        print(f"  Universe: {len(tickers)} tickers")

        # Collect
        panel = None
        if args.collect:
            panel = step_collect(market, tickers, use_news=args.news)
        else:
            cp = _cache_path(market)
            if Path(cp).exists():
                print(f"[iris] Loading cached panel from {cp}")
                panel = pd.read_parquet(cp)
                panel_summary(panel)
            else:
                sys.exit(f"[iris] No panel cache. Run with --collect first.")

        # Train
        model = None
        if args.train:
            model = step_train(market, panel)

        # Explain
        if args.explain:
            if model is None:
                model = _load_model(market)
            step_explain(model, panel)

        # Predict
        if args.predict:
            if model is None:
                model = _load_model(market)
            watchlist, _ = step_predict(market, panel, model, top_n=args.top_n)
            if args.alert:
                step_alert(market, watchlist, top_n=args.top_n)


if __name__ == '__main__':
    main()
