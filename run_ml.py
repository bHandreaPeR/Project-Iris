"""
Project Iris — ML regression pipeline entry point.

Steps:
  1. Build historical panel (ticker × quarter → features + forward return)
  2. Run walk-forward cross-validation (XGBoost + Ridge)
  3. Generate full explainability report (SHAP, feature IC, quintile analysis)
  4. Predict current picks and optionally broadcast via Telegram

Usage:
  # Build panel and train (first run — slow, ~20–60 min depending on universe):
  python run_ml.py --market us --collect --train --explain

  # Use cached panel (fast rerun):
  python run_ml.py --market us --train --explain

  # Only generate today's top picks from an already-trained model:
  python run_ml.py --market us --predict

  # Full pipeline with Telegram alert of top picks:
  python run_ml.py --market us --collect --train --explain --alert

Markets: 'us' | 'india' | 'all'
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from data.universe import INDIA_TICKERS, US_TICKERS
import config

CACHE_DIR   = Path('ml_output')
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(market: str, name: str) -> str:
    return str(CACHE_DIR / f"{market}_{name}.parquet")


def _model_path(market: str) -> Path:
    return CACHE_DIR / f"{market}_model.pkl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tickers(market: str) -> list[str]:
    if market == 'india':
        return INDIA_TICKERS
    elif market == 'us':
        return US_TICKERS
    elif market == 'all':
        return INDIA_TICKERS + US_TICKERS
    raise ValueError(f"Unknown market: {market}")


def _save_model(market: str, xgb_model, ridge_model, preprocessor, feat_cols):
    payload = {
        'xgb':         xgb_model,
        'ridge':       ridge_model,
        'preprocessor': preprocessor,
        'feat_cols':   feat_cols,
    }
    with open(_model_path(market), 'wb') as f:
        pickle.dump(payload, f)
    print(f"[iris-ml] Model saved to {_model_path(market)}")


def _load_model(market: str) -> dict:
    path = _model_path(market)
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run with --train first.")
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_collect(market: str, tickers: list[str]) -> pd.DataFrame:
    from ml.collector import build_panel, summary
    cache = _cache_path(market, 'panel')
    print(f"\n[iris-ml] Collecting historical panel for {len(tickers)} {market} tickers …")
    panel = build_panel(tickers, cache_path=cache, verbose=True)
    summary(panel)
    return panel


def step_train(market: str, panel: pd.DataFrame):
    from ml.model import walk_forward
    feat_cols = [c for c in panel.columns if c not in ('fwd_return', 'entry_date')]
    print(f"\n[iris-ml] Walk-forward training on {len(panel)} rows × {len(feat_cols)} features …")
    oof_df, xgb_model, ridge_model, preprocessor, feat_cols = walk_forward(panel)
    _save_model(market, xgb_model, ridge_model, preprocessor, feat_cols)
    return oof_df, xgb_model, preprocessor, feat_cols


def step_explain(oof_df, xgb_model, preprocessor, panel, feat_cols):
    from ml.explain import full_report
    full_report(oof_df, xgb_model, preprocessor, panel, feat_cols, save=True)


def step_predict(market: str, panel: pd.DataFrame, top_n: int = 20):
    from ml.model import predict_now
    m = _load_model(market)
    picks = predict_now(panel, m['xgb'], m['preprocessor'], m['feat_cols'], top_n=top_n)
    print(f"\n[iris-ml] Top {top_n} predicted picks for {market.upper()}:")
    print(picks.to_string())
    return picks


def step_alert(market: str, picks: pd.DataFrame):
    from alerts.telegram import broadcast
    if not config.TELEGRAM['bot_token'] or not config.TELEGRAM['chat_ids']:
        print("[iris-ml] Telegram not configured — skipping alert.")
        return

    import datetime
    date_str = datetime.date.today().strftime('%d-%b-%Y')
    label = 'India (NSE)' if market == 'india' else 'US (NYSE/NASDAQ)'

    lines = [
        f"<b>IRIS ML — Top Picks  {label}</b>",
        f"<i>{date_str} | Model: XGBoost walk-forward</i>",
        "",
    ]
    for rank, (tkr, row) in enumerate(picks.iterrows(), 1):
        pred = row.get('pred_fwd_return', float('nan'))
        pred_str = f"+{pred:.1%}" if pred >= 0 else f"{pred:.1%}"
        lines.append(f"<b>{rank}. {tkr}</b>  (pred 3m: {pred_str})")

    lines += [
        "",
        "<i>Not financial advice. Model output only.</i>",
    ]
    msg = '\n'.join(lines)
    broadcast(config.TELEGRAM['bot_token'], config.TELEGRAM['chat_ids'], msg)
    print(f"[iris-ml] Alert sent to {len(config.TELEGRAM['chat_ids'])} chat(s).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Project Iris — ML pipeline')
    parser.add_argument('--market',   choices=['india', 'us', 'all'], default='us')
    parser.add_argument('--collect',  action='store_true', help='Fetch/rebuild historical panel')
    parser.add_argument('--train',    action='store_true', help='Run walk-forward training')
    parser.add_argument('--explain',  action='store_true', help='Generate explainability report')
    parser.add_argument('--predict',  action='store_true', help='Predict current period picks')
    parser.add_argument('--alert',    action='store_true', help='Broadcast picks via Telegram')
    parser.add_argument('--top-n',    type=int, default=20, help='Number of picks to show/alert')
    args = parser.parse_args()

    markets = ['india', 'us'] if args.market == 'all' else [args.market]

    for market in markets:
        print(f"\n{'═'*60}")
        print(f"  Market: {market.upper()}")
        print(f"{'═'*60}")

        tickers = _tickers(market)

        # ── Collect ──────────────────────────────────────────────────
        panel = None
        if args.collect:
            panel = step_collect(market, tickers)
        else:
            cache = _cache_path(market, 'panel')
            from pathlib import Path as _P
            if _P(cache).exists():
                print(f"[iris-ml] Loading cached panel from {cache}")
                panel = pd.read_parquet(cache)
            else:
                print(f"[iris-ml] No cached panel found. Run with --collect first.")
                continue

        # ── Train ─────────────────────────────────────────────────────
        oof_df, xgb_model, preprocessor, feat_cols = None, None, None, None
        if args.train:
            oof_df, xgb_model, preprocessor, feat_cols = step_train(market, panel)

        # ── Explain ───────────────────────────────────────────────────
        if args.explain:
            if xgb_model is None:
                m = _load_model(market)
                xgb_model, preprocessor, feat_cols = m['xgb'], m['preprocessor'], m['feat_cols']
                oof_path = CACHE_DIR / f"{market}_oof.csv"
                if oof_path.exists():
                    oof_df = pd.read_csv(oof_path)
                else:
                    print("[iris-ml] No OOF predictions found — skipping explain. Run --train first.")
                    args.explain = False
            if args.explain:
                step_explain(oof_df, xgb_model, preprocessor, panel, feat_cols)

        # ── Predict ───────────────────────────────────────────────────
        if args.predict or args.alert:
            picks = step_predict(market, panel, top_n=args.top_n)
            if args.alert:
                step_alert(market, picks)


if __name__ == '__main__':
    main()
