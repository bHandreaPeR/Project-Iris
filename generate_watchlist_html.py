"""
Generate live HTML watchlist:
  - 5-year Chart.js price chart (inline data from yfinance)
  - Company name (not ticker ID) as primary header
  - Stock-specific news + sector/macro news (Google News RSS)
  - ML signals with reasoning
Outputs: ml_output/watchlist_live.html
"""

import sys, pickle, warnings, math, json, time, re
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from ml.reasoning import generate_reasoning

_OUT      = Path("ml_output")
_SESS     = requests.Session()
_SESS.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"})

# ---------------------------------------------------------------------------
# Sector → Google News query map
# ---------------------------------------------------------------------------
SECTOR_QUERIES = {
    "Technology":             "India IT technology sector government PLI export",
    "Healthcare":             "India healthcare pharma CDSCO PLI policy budget",
    "Financial Services":     "India banking NBFC RBI SEBI regulation credit growth",
    "Consumer Cyclical":      "India consumer discretionary retail demand GST",
    "Industrials":            "India capital goods manufacturing infrastructure PLI scheme",
    "Basic Materials":        "India metals mining chemicals commodity budget",
    "Energy":                 "India energy oil gas renewable power policy budget",
    "Real Estate":            "India real estate RERA housing affordable policy",
    "Utilities":              "India power electricity tariff CERC renewable",
    "Communication Services": "India telecom TRAI spectrum 5G broadband policy",
    "Consumer Defensive":     "India FMCG staples rural demand inflation budget",
    "Automobile":             "India automobile EV PLI FAME scheme policy",
    "Textiles":               "India textiles PLI export yarn apparel sector",
    "Chemicals":              "India specialty chemicals export PLI budget",
    "Pharmaceuticals":        "India pharma USFDA drug approval PLI API",
}
MACRO_QUERY = "India stock market RBI budget FII DII Nifty economy 2025"

# ---------------------------------------------------------------------------
# Signal loader
# ---------------------------------------------------------------------------

def load_signals(panel: pd.DataFrame, model) -> pd.DataFrame:
    feat_cols = model.feat_cols
    today = pd.Timestamp.now().normalize()
    latest_per_ticker = (panel
        .sort_index(level="quarter_end")
        .groupby(level="ticker").tail(1))

    rows = []
    for tkr, grp in latest_per_ticker.groupby(level="ticker"):
        try:
            row_data = grp.iloc[-1]
            feats = row_data[feat_cols]
            p = model.predict_ticker(feats)
            entry_date = pd.Timestamp(row_data.get("entry_date", pd.NaT))
            # If the panel was built before the filing actually occurred, the
            # entry_date may be future-dated (lag estimate).  If yfinance already
            # shows the new numbers (i.e. we're reading them right now), treat
            # the signal as current — clamp future entry_dates to today.
            if not pd.isnull(entry_date) and entry_date > today:
                entry_date = today
            age_days = int((today - entry_date).days) if not pd.isnull(entry_date) else 999
            rows.append({
                "ticker":     tkr,
                "pred_3m":   p["fwd_3m"]["point"],
                "lower_3m":  p["fwd_3m"]["lower"],
                "upper_3m":  p["fwd_3m"]["upper"],
                "pred_1m":   p["fwd_1m"]["point"],
                "pred_12m":  p["fwd_12m"]["point"],
                "_feats":    feats,
                "entry_date": entry_date,
                "age_days":   age_days,
            })
        except Exception:
            pass

    df = (pd.DataFrame(rows)
          .sort_values("pred_3m", ascending=False)
          .reset_index(drop=True))
    n = len(df)
    df["pct_rank"]   = ((n - df.index) / n * 100)
    df["signal"]     = "HOLD"
    df.loc[df["pct_rank"] >= 90, "signal"] = "BUY"
    df.loc[df["pct_rank"] <= 10, "signal"] = "SELL"
    df["confidence"] = "Medium"
    df.loc[df["pct_rank"] >= 97, "confidence"] = "High"
    df.loc[df["pct_rank"] <= 3,  "confidence"] = "High"

    # Sort: freshest first (age_days asc), then by bias strength within same age bucket
    # BUYs:  highest pred_3m first (strongest bullish bias)
    # SELLs: lowest pred_3m first  (strongest bearish bias)
    buys  = df[df["signal"] == "BUY"].sort_values(["age_days", "pred_3m"], ascending=[True, False])
    sells = df[df["signal"] == "SELL"].sort_values(["age_days", "pred_3m"], ascending=[True, True])
    holds = df[df["signal"] == "HOLD"]
    df = pd.concat([buys, sells, holds]).reset_index(drop=True)

    print(f"Scored {n} tickers | BUYs: {(df['signal']=='BUY').sum()} | SELLs: {(df['signal']=='SELL').sum()}")
    print(f"Signal ages — <30d: {(df['age_days']<=30).sum()}  <60d: {(df['age_days']<=60).sum()}  <90d: {(df['age_days']<=90).sum()}")
    return df

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def screener_url(ticker: str) -> str:
    if ticker.endswith(".NS"):
        return f"https://www.screener.in/company/{ticker[:-3]}/consolidated/"
    if ticker.endswith(".BO"):
        return f"https://www.screener.in/company/{ticker[:-3]}/"
    return "https://www.screener.in"

def tv_symbol(ticker: str) -> str:
    if ticker.endswith(".NS"): return f"NSE:{ticker[:-3]}"
    if ticker.endswith(".BO"): return f"BSE:{ticker[:-3]}"
    return ticker

def fmt_mktcap(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    v = float(v)
    if v >= 1e12: return f"₹{v/1e12:.1f}T"
    if v >= 1e9:  return f"₹{v/1e9:.0f}B"
    if v >= 1e7:  return f"₹{v/1e7:.0f}Cr"
    return f"₹{v:,.0f}"

def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return f"{v*100:+.1f}%"

def fmt_price(v) -> str:
    if not v: return "—"
    return f"₹{v:,.2f}"

def fmt_ratio(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return f"{v:.1f}x"

def pct_class(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "neu"
    return "pos" if v > 0.005 else ("neg" if v < -0.005 else "neu")

# ---------------------------------------------------------------------------
# Name resolver (handles bad BSE names from yfinance)
# ---------------------------------------------------------------------------
_NAME_CACHE: dict[str, str] = {}

def resolve_name(ticker: str, yf_name: str) -> str:
    """Return a clean company name; falls back to screener.in for BSE codes."""
    if ticker in _NAME_CACHE:
        return _NAME_CACHE[ticker]
    name = yf_name or ""
    # yfinance returns garbage like "523888.BO,0P0000E7JZ,0" for unknown BSE tickers
    if "," in name or not name or name == ticker:
        code = ticker.replace(".BO", "").replace(".NS", "")
        try:
            r = _SESS.get(f"https://www.screener.in/api/company/search/?q={code}&v=3", timeout=6)
            data = r.json()
            if data:
                name = data[0]["name"]
        except Exception:
            name = code
    _NAME_CACHE[ticker] = name
    return name

# ---------------------------------------------------------------------------
# Live data fetcher
# ---------------------------------------------------------------------------

def fetch_ticker_data(ticker: str) -> dict:
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}

        # Use history() for price — fast_info/info.regularMarketPrice is unreliable for BSE tickers
        hist  = t.history(period="5y", interval="1wk")
        price = 0
        prev  = 0
        if not hist.empty:
            closes = hist["Close"].dropna()
            if len(closes) >= 2:
                price = float(closes.iloc[-1])
                prev  = float(closes.iloc[-2])
            elif len(closes) == 1:
                price = float(closes.iloc[-1])
                prev  = price
        chg = ((price - prev) / prev * 100) if prev else 0
        chart_dates  = []
        chart_prices = []
        if not hist.empty:
            hist = hist.dropna(subset=["Close"])
            chart_dates  = [d.strftime("%Y-%m-%d") for d in hist.index]
            chart_prices = [round(float(v), 2) for v in hist["Close"]]

        # News
        news_raw = t.news or []
        news = []
        for item in news_raw[:5]:
            c     = item.get("content", {})
            title = c.get("title", "")
            url   = ((c.get("canonicalUrl") or {}).get("url") or
                     (c.get("clickThroughUrl") or {}).get("url") or "")
            pub   = (c.get("provider") or {}).get("displayName", "")
            thumb = None
            try: thumb = c["thumbnail"]["resolutions"][1]["url"]
            except Exception: pass
            if title and url:
                news.append({"title": title, "url": url, "pub": pub, "thumb": thumb})

        raw_name = info.get("longName") or info.get("shortName") or ticker
        name     = resolve_name(ticker, raw_name)
        sector   = info.get("sector", "") or info.get("industry", "")

        return {
            "price":         price,
            "chg_pct":       chg,
            "name":          name,
            "sector":        sector,
            "industry":      info.get("industry", ""),
            "pe":            info.get("trailingPE"),
            "pb":            info.get("priceToBook"),
            "mktcap":        info.get("marketCap"),
            "52w_high":      info.get("fiftyTwoWeekHigh"),
            "52w_low":       info.get("fiftyTwoWeekLow"),
            "news":          news,
            "chart_dates":   chart_dates,
            "chart_prices":  chart_prices,
        }
    except Exception:
        return {"price": 0, "chg_pct": 0, "name": ticker, "sector": "", "industry": "",
                "pe": None, "pb": None, "mktcap": None, "52w_high": None, "52w_low": None,
                "news": [], "chart_dates": [], "chart_prices": []}

# ---------------------------------------------------------------------------
# Historical signal markers for chart (cross-sectional OOF rankings)
# ---------------------------------------------------------------------------

def build_cross_sectional_signals(model, panel: pd.DataFrame) -> pd.DataFrame:
    """
    Use the model's OOF (out-of-fold) predictions to compute cross-sectional
    percentile ranks at each quarter_end. Returns a DataFrame with columns:
    [ticker, quarter_end, entry_date, pred_q50, pct_rank, signal]

    BUY  = top 10% of all tickers at that time period
    SELL = bottom 10% of all tickers at that time period

    These are honest, look-ahead-free rankings computed during walk-forward training.
    Only signals with entry_date <= today are included (no future-dated rows).
    """
    TODAY = pd.Timestamp.now().normalize()

    oof = model.oof_df
    oof3m = oof[oof["horizon"] == "fwd_3m"].copy()

    # Cross-sectional percentile rank at each quarter_end
    oof3m["pct_rank"] = (
        oof3m.groupby("quarter_end")["pred_q50"]
             .rank(pct=True) * 100
    )
    oof3m["signal"] = "HOLD"
    oof3m.loc[oof3m["pct_rank"] >= 90, "signal"] = "BUY"
    oof3m.loc[oof3m["pct_rank"] <= 10, "signal"] = "SELL"

    # Attach entry_date from panel
    panel_dates = (panel.reset_index()[["ticker", "quarter_end", "entry_date"]]
                        .drop_duplicates())
    oof3m = oof3m.merge(panel_dates, on=["ticker", "quarter_end"], how="left")

    # QC: drop future-dated entries (annual rows filed > today)
    oof3m["entry_date"] = pd.to_datetime(oof3m["entry_date"])
    n_before = len(oof3m)
    oof3m = oof3m[oof3m["entry_date"] <= TODAY]
    n_dropped = n_before - len(oof3m)
    if n_dropped:
        print(f"  [QC] Dropped {n_dropped} future-dated OOF rows (entry_date > {TODAY.date()})")

    return oof3m


def get_signal_history(ticker: str, oof_signals: pd.DataFrame,
                       chart_dates: list, chart_prices: list) -> dict:
    """
    Look up cross-sectional BUY/SELL history for a ticker from the pre-computed
    OOF signals DataFrame, and map each signal entry_date to the nearest weekly
    price point on the 5Y chart.

    QC rules applied here:
      - entry_date must be within the chart's date range (no extrapolation)
      - nearest match must be within 60 days (no spurious snapping across gaps)
      - one marker per chart week (deduplicated)
    """
    sub = oof_signals[
        (oof_signals["ticker"] == ticker) &
        (oof_signals["signal"].isin(["BUY", "SELL"]))
    ].copy()

    if sub.empty or not chart_dates:
        return {"buy": [], "sell": []}

    # Build chart date → price lookup and timestamp list
    date_price  = {d: p for d, p in zip(chart_dates, chart_prices)}
    chart_ts    = [pd.Timestamp(d) for d in chart_dates]
    chart_start = chart_ts[0]
    chart_end   = chart_ts[-1]

    def nearest_price(target_date):
        if pd.isnull(target_date):
            return None
        t = pd.Timestamp(target_date)
        # Must fall within chart range — no clamping to edges
        if t < chart_start or t > chart_end:
            return None
        diffs = [abs((ct - t).days) for ct in chart_ts]
        min_diff = min(diffs)
        # Reject if nearest chart point is more than 60 days away
        if min_diff > 60:
            return None
        idx = diffs.index(min_diff)
        best_d = chart_dates[idx]
        return best_d, date_price[best_d]

    buy_markers, sell_markers = [], []
    for _, row in sub.iterrows():
        entry_date = row.get("entry_date")
        result = nearest_price(entry_date)
        if result is None:
            continue
        date_str, price = result
        marker = {"x": date_str, "y": price}
        if row["signal"] == "BUY":
            buy_markers.append(marker)
        else:
            sell_markers.append(marker)

    # Deduplicate: one marker per chart date
    def dedup(markers):
        seen = {}
        for m in markers:
            seen[m["x"]] = m
        return list(seen.values())

    return {"buy": dedup(buy_markers), "sell": dedup(sell_markers)}


# ---------------------------------------------------------------------------
# Sector + macro news (Google News RSS)
# ---------------------------------------------------------------------------
_SECTOR_NEWS_CACHE: dict[str, list] = {}

def fetch_rss_news(query: str, n: int = 4) -> list:
    """Fetch and parse Google News RSS for a query."""
    try:
        url = (f"https://news.google.com/rss/search?q="
               f"{urllib.parse.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en")
        r   = _SESS.get(url, timeout=8)
        root = ET.fromstring(r.content)
        items = root.findall(".//item")[:n]
        out = []
        for it in items:
            title  = (it.find("title").text or "").strip()
            link   = (it.find("link").text or "").strip()
            src_el = it.find("source")
            source = src_el.text.strip() if src_el is not None and src_el.text else ""
            pub_el = it.find("pubDate")
            pub    = pub_el.text.strip() if pub_el is not None and pub_el.text else ""
            # Clean up Google redirect URL
            if "news.google.com" in link:
                link = urllib.parse.unquote(link.split("url=")[-1]) if "url=" in link else link
            if title:
                out.append({"title": title, "url": link, "source": source, "pub": pub})
        return out
    except Exception:
        return []

def get_sector_news(sector: str) -> list:
    """Return cached sector news for a given sector string."""
    key = sector or "General"
    if key not in _SECTOR_NEWS_CACHE:
        # Map to closest known sector
        mapped = next((v for k, v in SECTOR_QUERIES.items()
                       if k.lower() in key.lower()), f"India {key} sector policy budget")
        _SECTOR_NEWS_CACHE[key] = fetch_rss_news(mapped, n=4)
        time.sleep(0.3)
    return _SECTOR_NEWS_CACHE[key]

def get_macro_news() -> list:
    return fetch_rss_news(MACRO_QUERY, n=3)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg:       #090909;
  --s1:       #121212;
  --s2:       #1a1a1a;
  --s3:       #222;
  --border:   #242424;
  --text:     #e6e6e6;
  --muted:    #666;
  --muted2:   #999;
  --green:    #00e676;
  --red:      #ff5252;
  --blue:     #5c9eff;
  --amber:    #ffab40;
  --purple:   #b39ddb;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;line-height:1.5}

/* ── TOP BAR ─────────────────────────── */
.topbar{background:#0e0e0e;border-bottom:1px solid var(--border);padding:13px 28px;
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:200;backdrop-filter:blur(8px)}
.logo{font-size:17px;font-weight:800;letter-spacing:-.4px}
.logo em{color:var(--green);font-style:normal}
.topbar-right{display:flex;gap:20px;align-items:center;color:var(--muted);font-size:11px}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);display:inline-block;
  margin-right:4px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* ── STATS BAR ───────────────────────── */
.stats-bar{max-width:1640px;margin:20px auto 0;padding:0 20px;display:flex;gap:10px;flex-wrap:wrap}
.stat-pill{background:var(--s1);border:1px solid var(--border);border-radius:10px;padding:10px 18px}
.sv{font-size:19px;font-weight:700;display:block}
.sl{font-size:11px;color:var(--muted)}
.sv.g{color:var(--green)} .sv.r{color:var(--red)}

/* ── SECTION ─────────────────────────── */
.section{max-width:1640px;margin:28px auto 0;padding:0 20px}
.stitle{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1.3px;margin-bottom:3px}
.stitle.buy{color:var(--green)} .stitle.sell{color:var(--red)}
.ssub{color:var(--muted);font-size:11px;margin-bottom:14px}

/* ── CARDS GRID ──────────────────────── */
.cards-grid{max-width:1640px;margin:0 auto;padding:0 20px 32px;
  display:grid;grid-template-columns:repeat(auto-fill,minmax(900px,1fr));gap:14px}

/* ── CARD ────────────────────────────── */
.card{background:var(--s1);border:1px solid var(--border);border-radius:12px;overflow:hidden;
  transition:box-shadow .15s,transform .15s}
.card:hover{box-shadow:0 10px 40px rgba(0,0,0,.5);transform:translateY(-1px)}
.card.buy {border-left:3px solid var(--green)}
.card.sell{border-left:3px solid var(--red)}

/* ── CARD HEADER ─────────────────────── */
.card-header{display:flex;justify-content:space-between;align-items:flex-start;
  padding:14px 18px 12px;border-bottom:1px solid var(--border);background:var(--s2)}
.name-block{display:flex;flex-direction:column;gap:2px;max-width:480px}
.co-name{font-size:16px;font-weight:700;line-height:1.3;letter-spacing:-.2px}
.co-ticker{font-size:11px;color:var(--muted);margin-top:1px}
.co-ticker .exch{background:var(--s3);color:var(--muted2);font-size:10px;
  padding:1px 7px;border-radius:8px;margin-left:6px}
.tag-row{display:flex;gap:5px;margin-top:5px;flex-wrap:wrap}
.tag{font-size:10px;padding:2px 8px;border-radius:9px;background:var(--s3);color:var(--muted2)}
.signal-block{display:flex;flex-direction:column;align-items:flex-end;gap:5px;flex-shrink:0}
.sig-badge{font-size:12px;font-weight:800;padding:5px 16px;border-radius:20px;letter-spacing:.8px}
.sig-badge.buy {background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.35)}
.sig-badge.sell{background:rgba(255,82,82,.1); color:var(--red);  border:1px solid rgba(255,82,82,.35)}
.sig-badge.hold{background:rgba(255,171,64,.1);color:var(--amber);border:1px solid rgba(255,171,64,.35)}
.conf-badge{font-size:10px;padding:2px 9px;border-radius:9px}
.conf-high  {background:rgba(92,158,255,.12);color:var(--blue)}
.conf-medium{background:var(--s3);color:var(--muted)}
.freshness-badge{font-size:10px;padding:2px 9px;border-radius:9px;font-weight:600}
.freshness-badge.fresh {background:rgba(0,230,118,.1);color:var(--green)}
.freshness-badge.recent{background:rgba(255,171,64,.1);color:var(--amber)}
.freshness-badge.stale {background:rgba(255,82,82,.1); color:var(--red)}

/* ── CARD BODY — 3 columns ────────────── */
.card-body{display:grid;grid-template-columns:1fr 220px 220px}

/* ── LEFT COL (KPIs + chart) ─────────── */
.left-col{padding:13px 16px 0;border-right:1px solid var(--border);display:flex;flex-direction:column;gap:10px}
.chart-section{border-top:1px solid var(--border);padding:10px 0 12px;display:flex;flex-direction:column;gap:6px;margin-top:4px}
.chart-title{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}
.chart-wrap{position:relative;min-height:150px}
.chart-wrap canvas{width:100%!important}
.screener-btn{display:flex;align-items:center;justify-content:center;gap:5px;
  padding:5px 10px;font-size:11px;color:var(--muted);text-decoration:none;
  border:1px solid var(--border);border-radius:7px;transition:all .2s;margin-top:2px;align-self:flex-start}
.screener-btn:hover{color:var(--text);border-color:#444;background:var(--s2)}
.price-row{display:flex;align-items:baseline;gap:10px}
.live-price{font-size:22px;font-weight:800}
.price-chg{font-size:12px;font-weight:600}
.price-chg.up{color:var(--green)} .price-chg.down{color:var(--red)}

.return-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}
.rbox{background:var(--s2);border-radius:8px;padding:7px 9px;text-align:center}
.rbox.main{border:1px solid rgba(255,255,255,.06)}
.rl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;margin-bottom:3px}
.rv{font-size:16px;font-weight:700}
.rv.pos{color:var(--green)} .rv.neg{color:var(--red)} .rv.neu{color:var(--muted)}

.range-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;
  margin-bottom:5px;display:block}
.range-bar{position:relative;height:4px;background:rgba(255,255,255,.07);border-radius:2px;margin-bottom:4px}
.range-fill{position:absolute;top:0;height:100%;background:rgba(140,140,140,.2);border-radius:2px}
.range-dot{position:absolute;top:-4px;width:12px;height:12px;border-radius:50%;
  transform:translateX(-50%);border:2px solid var(--bg)}
.range-dot.pos{background:var(--green)} .range-dot.neg{background:var(--red)}
.range-dot.neu{background:var(--muted)}
.range-nums{display:flex;justify-content:space-between;font-size:10px;color:var(--muted)}

.fund-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.fi{background:var(--s2);border-radius:6px;padding:5px 9px;
  display:flex;justify-content:space-between;align-items:center}
.fl{font-size:10px;color:var(--muted)} .fv{font-size:11px;font-weight:600}

/* ── REASONING ───────────────────────── */
.reasoning{border-top:1px solid var(--border);padding:9px 0;background:rgba(255,255,255,.015)}
.rlabel{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px}
.rlist{display:flex;flex-direction:column;gap:4px}
.ritem{display:flex;gap:7px;align-items:flex-start;font-size:11px;color:#b8b8b8;line-height:1.45}
.rdot{width:5px;height:5px;border-radius:50%;margin-top:5px;flex-shrink:0}
.rdot.buy{background:var(--green)} .rdot.sell{background:var(--red)} .rdot.hold{background:var(--muted)}

/* ── NEWS COL ────────────────────────── */
.news-col{padding:11px;border-right:1px solid var(--border);
  overflow-y:auto;max-height:380px;display:flex;flex-direction:column;gap:5px}
.news-col-title{font-size:10px;text-transform:uppercase;letter-spacing:.9px;
  color:var(--muted);margin-bottom:3px;padding-bottom:5px;border-bottom:1px solid var(--border)}

/* ── SECTOR COL ──────────────────────── */
.sector-col{padding:11px;overflow-y:auto;max-height:380px;display:flex;flex-direction:column;gap:5px}
.sector-col-title{font-size:10px;text-transform:uppercase;letter-spacing:.9px;
  color:var(--purple);margin-bottom:3px;padding-bottom:5px;border-bottom:1px solid var(--border)}

.nitem{display:flex;gap:7px;text-decoration:none;color:var(--text);
  padding:7px;border-radius:7px;border:1px solid var(--border);
  transition:background .12s;margin-bottom:1px}
.nitem:hover{background:var(--s2)}
.nthumb{width:50px;height:36px;object-fit:cover;border-radius:4px;flex-shrink:0;background:var(--s2)}
.ntext{flex:1;min-width:0}
.ntitle{font-size:11px;line-height:1.4;font-weight:500;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.npub{font-size:10px;color:var(--muted);margin-top:2px}
.no-news{font-size:11px;color:var(--muted);padding:6px}

/* ── FOOTER ──────────────────────────── */
.footer{text-align:center;padding:30px 20px;color:var(--muted);font-size:11px;
  border-top:1px solid var(--border);margin-top:16px}
.footer a{color:var(--blue);text-decoration:none}

@media(max-width:1000px){
  .cards-grid{grid-template-columns:1fr}
  .card-body{grid-template-columns:1fr}
  .left-col,.news-col{border-right:none;border-bottom:1px solid var(--border)}
}
"""

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Project Iris — Live Watchlist</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>{css}</style>
</head>
<body>

<div class="topbar">
  <div class="logo">Project <em>Iris</em></div>
  <div class="topbar-right">
    <span><span class="live-dot"></span>Live · India</span>
    <span>NSE + BSE · {total_tickers:,} stocks scored</span>
    <span>Updated {updated}</span>
  </div>
</div>

<div class="stats-bar">
  <div class="stat-pill"><span class="sv g">{n_buys}</span><span class="sl">BUY Signals</span></div>
  <div class="stat-pill"><span class="sv r">{n_sells}</span><span class="sl">SELL / Avoid</span></div>
  <div class="stat-pill"><span class="sv">{top_pred}</span><span class="sl">Best 3M Forecast</span></div>
  <div class="stat-pill"><span class="sv g">{n_fresh}</span><span class="sl">Fresh (&lt;30d)</span></div>
  <div class="stat-pill"><span class="sv">66</span><span class="sl">Backtest Folds</span></div>
  <div class="stat-pill"><span class="sv">2015–2026</span><span class="sl">Training History</span></div>
  <div class="stat-pill"><span class="sv">133</span><span class="sl">Features</span></div>
</div>

<div class="section">
  <div class="stitle buy">▲ Strong Buy — Top Decile</div>
  <div class="ssub">Ranked by 3-month return forecast · XGBoost quantile ensemble · {updated}</div>
</div>
<div class="cards-grid">{buy_cards}</div>

<div class="section">
  <div class="stitle sell">▼ Avoid / Short — Bottom Decile</div>
  <div class="ssub">Highest predicted underperformers — consider avoiding or reducing exposure</div>
</div>
<div class="cards-grid">{sell_cards}</div>

<div class="footer">
  <strong>Project Iris</strong> &middot;
  ML signals for India (NSE + BSE) &middot;
  Charts by <a href="https://www.chartjs.org" target="_blank">Chart.js</a> &middot;
  Fundamentals via <a href="https://screener.in" target="_blank">Screener.in</a> &middot;
  News via Google News &amp; Yahoo Finance<br>
  <em>Not financial advice. Probabilistic forecasts only.</em>
</div>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Card builder
# ---------------------------------------------------------------------------

def build_card(row: pd.Series, live: dict, sig_class: str,
               sector_news: list, macro_news: list,
               signal_markers: dict | None = None) -> str:
    ticker   = row["ticker"]
    signal   = row["signal"]
    conf     = row["confidence"]
    pred_3m  = float(row["pred_3m"])
    pred_1m  = float(row["pred_1m"])
    pred_12m = float(row["pred_12m"])
    lower    = float(row["lower_3m"])
    upper    = float(row["upper_3m"])
    age_days = int(row.get("age_days", 999))
    entry_dt = row.get("entry_date")
    entry_str = entry_dt.strftime("%b %d, %Y") if pd.notnull(entry_dt) else "—"
    if age_days <= 30:
        freshness_cls = "fresh"
        freshness_lbl = f"🟢 {age_days}d ago"
    elif age_days <= 90:
        freshness_cls = "recent"
        freshness_lbl = f"🟡 {age_days}d ago"
    else:
        freshness_cls = "stale"
        freshness_lbl = f"🔴 {age_days}d ago"

    safe_id  = ticker.replace(".", "_").replace("-", "_")
    sym_disp = ticker.replace(".NS", "").replace(".BO", "")
    exch     = "NSE" if ticker.endswith(".NS") else "BSE"
    name     = live.get("name") or sym_disp
    sector   = live.get("sector") or live.get("industry") or ""
    industry = live.get("industry") or ""
    scr_url  = screener_url(ticker)

    # ── 5Y price chart data ─────────────────────────────────────────────
    dates  = live.get("chart_dates",  [])
    prices = live.get("chart_prices", [])
    has_chart = len(prices) >= 4

    if has_chart:
        start_p = prices[0]
        end_p   = prices[-1]
        is_up   = end_p >= start_p
        chart_color  = "rgba(0,230,118,1)"   if is_up else "rgba(255,82,82,1)"
        chart_fill   = "rgba(0,230,118,0.08)" if is_up else "rgba(255,82,82,0.08)"
        # Downsample to max 130 points for HTML size
        step = max(1, len(dates) // 130)
        d_ds = dates[::step]
        p_ds = prices[::step]

        # Signal marker datasets (BUY = green ▲, SELL = red ▼)
        # All signals — including current period — plotted as triangles at their
        # actual entry_date position on the price line. No special treatment.
        sm = signal_markers or {"buy": [], "sell": []}

        # CRITICAL: Chart.js uses a categorical axis with labels = d_ds (downsampled).
        # Scatter marker x-values must exactly match one of those labels, otherwise
        # Chart.js cannot find the label and defaults the point to index 0 (wrong date).
        # Fix: snap each marker's x-value to the nearest date that exists in d_ds.
        d_ds_set = set(d_ds)
        d_ds_ts  = [(d, pd.Timestamp(d)) for d in d_ds]

        def snap_to_ds(marker):
            x = marker["x"]
            if x in d_ds_set:
                return marker
            t = pd.Timestamp(x)
            best = min(d_ds_ts, key=lambda dt: abs((dt[1] - t).days))
            return {"x": best[0], "y": marker["y"]}

        buy_pts  = json.dumps([snap_to_ds(m) for m in sm.get("buy",  [])])
        sell_pts = json.dumps([snap_to_ds(m) for m in sm.get("sell", [])])

        chart_js = f"""
<script>
(function(){{
  var ctx = document.getElementById('ch-{safe_id}').getContext('2d');
  var grad = ctx.createLinearGradient(0, 0, 0, 165);
  grad.addColorStop(0, '{chart_fill.replace("0.08","0.18")}');
  grad.addColorStop(1, 'rgba(0,0,0,0)');

  var signalPlugin = {{
    id:'signalPlugin',
    afterDatasetsDraw(chart){{
      const ctx2=chart.ctx;
      chart.data.datasets.forEach((ds,i)=>{{
        if(!ds._isBuyMarker && !ds._isSellMarker) return;
        const meta=chart.getDatasetMeta(i);
        meta.data.forEach(pt=>{{
          const x=pt.x, y=pt.y, s=7;
          ctx2.save();
          ctx2.fillStyle=ds._isBuyMarker?'rgba(0,230,118,0.9)':'rgba(255,82,82,0.9)';
          ctx2.strokeStyle='rgba(0,0,0,0.6)';ctx2.lineWidth=0.8;
          ctx2.beginPath();
          if(ds._isBuyMarker){{ctx2.moveTo(x,y-s);ctx2.lineTo(x+s,y+s*0.6);ctx2.lineTo(x-s,y+s*0.6);}}
          else{{ctx2.moveTo(x,y+s);ctx2.lineTo(x+s,y-s*0.6);ctx2.lineTo(x-s,y-s*0.6);}}
          ctx2.closePath();ctx2.fill();ctx2.stroke();
          ctx2.restore();
        }});
      }});
    }}
  }};

  new Chart(ctx, {{
    type: 'line',
    plugins: [signalPlugin],
    data: {{
      labels: {json.dumps(d_ds)},
      datasets: [
        {{
          data: {json.dumps(p_ds)},
          borderColor: '{chart_color}',
          backgroundColor: grad,
          fill: true,
          tension: 0.35,
          pointRadius: 0,
          borderWidth: 1.8,
          order: 3
        }},
        {{
          type: 'scatter',
          data: {buy_pts},
          _isBuyMarker: true,
          pointRadius: 0,
          borderWidth: 0,
          backgroundColor: 'transparent',
          parsing: {{ xAxisKey:'x', yAxisKey:'y' }},
          order: 1
        }},
        {{
          type: 'scatter',
          data: {sell_pts},
          _isSellMarker: true,
          pointRadius: 0,
          borderWidth: 0,
          backgroundColor: 'transparent',
          parsing: {{ xAxisKey:'x', yAxisKey:'y' }},
          order: 2
        }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          mode: 'index',
          intersect: false,
          backgroundColor: '#1a1a1a',
          borderColor: '#333',
          borderWidth: 1,
          titleColor: '#999',
          bodyColor: '#e6e6e6',
          callbacks: {{
            label: ctx => {{
              if(ctx.dataset._isBuyMarker) return '▲ BUY signal';
              if(ctx.dataset._isSellMarker) return '▼ SELL signal';
              return '₹' + ctx.parsed.y.toLocaleString('en-IN', {{minimumFractionDigits:2}});
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{ maxTicksLimit: 6, color: '#555', font: {{ size: 9 }} }},
          grid: {{ color: 'rgba(255,255,255,0.04)' }},
          border: {{ color: 'rgba(255,255,255,0.06)' }}
        }},
        y: {{
          ticks: {{ maxTicksLimit: 4, color: '#555', font: {{ size: 9 }},
            callback: v => '₹' + v.toLocaleString('en-IN', {{maximumFractionDigits:0}}) }},
          grid: {{ color: 'rgba(255,255,255,0.04)' }},
          border: {{ color: 'rgba(255,255,255,0.06)' }}
        }}
      }}
    }}
  }});
}})();
</script>"""
        chart_html = f'<canvas id="ch-{safe_id}" height="165"></canvas>{chart_js}'
    else:
        chart_html = '<div style="height:165px;display:flex;align-items:center;justify-content:center;color:#444;font-size:11px">No price history</div>'

    # ── Range bar ─────────────────────────────────────────────────────
    span     = 3.1
    bar_left  = round(max(0, min(100, (max(lower, -0.6) + 0.6) / span * 100)), 1)
    bar_right = round(max(0, min(100, (min(upper,  2.5) + 0.6) / span * 100)), 1)
    bar_width = round(max(0, bar_right - bar_left), 1)
    bar_point = round(max(0, min(100, (pred_3m + 0.6) / span * 100)), 1)

    # ── Reasoning ─────────────────────────────────────────────────────
    feats = row.get("_feats")
    reasons = []
    if feats is not None:
        try:
            reasons = generate_reasoning(feats, signal, n_lines=4)
        except Exception:
            pass
    reason_html = "".join(
        f'<div class="ritem"><div class="rdot {sig_class}"></div><span>{r}</span></div>'
        for r in reasons
    ) or '<div class="ritem"><span style="color:var(--muted)">Insufficient data.</span></div>'

    # ── Stock news ─────────────────────────────────────────────────────
    def news_items_html(items, limit=4):
        if not items:
            return '<div class="no-news">No recent news found</div>'
        html = ""
        for n in items[:limit]:
            th = (f'<img class="nthumb" src="{n["thumb"]}" loading="lazy" '
                  f'onerror="this.style.display=\'none\'">' if n.get("thumb") else "")
            pub = n.get("pub") or n.get("source") or ""
            html += (f'<a class="nitem" href="{n["url"]}" target="_blank" rel="noopener">'
                     f'{th}<div class="ntext">'
                     f'<div class="ntitle">{n["title"][:90]}</div>'
                     f'<div class="npub">{pub}</div>'
                     f'</div></a>\n')
        return html

    stock_news_html  = news_items_html(live.get("news", []), 4)
    sector_news_html = news_items_html(sector_news + macro_news, 5)

    # ── Fundamentals ──────────────────────────────────────────────────
    price = live.get("price") or 0
    chg   = live.get("chg_pct") or 0
    wk52  = (f"₹{live['52w_low']:,.0f} / ₹{live['52w_high']:,.0f}"
             if live.get("52w_high") and live.get("52w_low") else "—")

    tags_html = ""
    if sector:
        tags_html += f'<span class="tag">{sector}</span>'
    if industry and industry != sector:
        tags_html += f'<span class="tag">{industry}</span>'

    return f"""
<div class="card {sig_class}" id="card-{safe_id}">
  <div class="card-header">
    <div class="name-block">
      <div class="co-name">{name}</div>
      <div class="co-ticker">{sym_disp} <span class="exch">{exch}</span></div>
      <div class="tag-row">{tags_html}</div>
    </div>
    <div class="signal-block">
      <span class="sig-badge {sig_class}">{signal}</span>
      <span class="conf-badge conf-{conf.lower()}">{conf} Confidence</span>
      <span class="freshness-badge {freshness_cls}" title="Based on financials available as of {entry_str}">{freshness_lbl}</span>
    </div>
  </div>

  <div class="card-body">
    <!-- LEFT: KPIs + chart -->
    <div class="left-col">
      <div class="price-row">
        <span class="live-price">{fmt_price(price) if price else '—'}</span>
        <span class="price-chg {'up' if chg >= 0 else 'down'}">{f'{chg:+.2f}%' if price else ''}</span>
      </div>

      <div class="return-grid">
        <div class="rbox"><div class="rl">1M Forecast</div>
          <div class="rv {pct_class(pred_1m)}">{fmt_pct(pred_1m)}</div></div>
        <div class="rbox main"><div class="rl">3M Forecast</div>
          <div class="rv {pct_class(pred_3m)}">{fmt_pct(pred_3m)}</div></div>
        <div class="rbox"><div class="rl">12M Forecast</div>
          <div class="rv {pct_class(pred_12m)}">{fmt_pct(pred_12m)}</div></div>
      </div>

      <div>
        <span class="range-lbl">3M Confidence Range</span>
        <div class="range-bar">
          <div class="range-fill" style="left:{bar_left}%;width:{bar_width}%"></div>
          <div class="range-dot {pct_class(pred_3m)}" style="left:{bar_point}%"></div>
        </div>
        <div class="range-nums"><span>{fmt_pct(lower)}</span><span>{fmt_pct(upper)}</span></div>
      </div>

      <div class="fund-grid">
        <div class="fi"><span class="fl">Mkt Cap</span><span class="fv">{fmt_mktcap(live.get('mktcap'))}</span></div>
        <div class="fi"><span class="fl">P/E</span><span class="fv">{fmt_ratio(live.get('pe'))}</span></div>
        <div class="fi"><span class="fl">P/B</span><span class="fv">{fmt_ratio(live.get('pb'))}</span></div>
        <div class="fi"><span class="fl">52W H/L</span><span class="fv" style="font-size:10px">{wk52}</span></div>
        <div class="fi"><span class="fl">Signal Date</span><span class="fv" style="font-size:10px;color:var(--muted)">{entry_str} ({age_days}d ago)</span></div>
      </div>

      <div class="reasoning">
        <div class="rlabel">Why this signal</div>
        <div class="rlist">{reason_html}</div>
      </div>

      <!-- 5Y chart below KPIs -->
      <div class="chart-section">
        <div class="chart-title">5-Year Price (Weekly)</div>
        <div class="chart-wrap">{chart_html}</div>
        <a href="{scr_url}" target="_blank" rel="noopener" class="screener-btn">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
          Screener.in
        </a>
      </div>
    </div>

    <!-- STOCK NEWS -->
    <div class="news-col">
      <div class="news-col-title">Company News</div>
      {stock_news_html}
    </div>

    <!-- SECTOR + MACRO NEWS -->
    <div class="sector-col">
      <div class="sector-col-title">&#127758; Sector &amp; Macro</div>
      {sector_news_html}
    </div>
  </div>
</div>"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model and panel …")
    model = pickle.load(open(_OUT / "v2_model_india_v4.pkl", "rb"))
    panel = pd.read_parquet(_OUT / "v2_panel_india_v4.parquet")

    print("Scoring all tickers …")
    signals = load_signals(panel, model)

    buys  = signals[signals["signal"] == "BUY"].head(20)
    sells = signals[signals["signal"] == "SELL"].head(10)
    all_rows = list(buys.iterrows()) + list(sells.iterrows())

    print(f"Fetching live data + 5Y charts for {len(all_rows)} tickers …")
    live_data: dict[str, dict] = {}
    for i, (_, row) in enumerate(all_rows, 1):
        tkr = row["ticker"]
        print(f"  {i:2d}/{len(all_rows)}  {tkr:<22}", end="  ", flush=True)
        live_data[tkr] = fetch_ticker_data(tkr)
        p = live_data[tkr]
        print(f"{fmt_price(p['price'])}  {p['name'][:35]}")

    print("Fetching macro news …")
    macro_news = get_macro_news()
    print(f"  Got {len(macro_news)} macro items")

    print("Fetching sector news …")
    sector_set = {live_data[r["ticker"]].get("sector", "") for _, r in all_rows}
    for sec in sector_set:
        if sec:
            sn = get_sector_news(sec)
            print(f"  {sec}: {len(sn)} items")

    print("Building HTML …")

    print("Computing cross-sectional OOF signal history …")
    oof_signals = build_cross_sectional_signals(model, panel)
    signal_history: dict[str, dict] = {}
    for _, row in list(buys.iterrows()) + list(sells.iterrows()):
        tkr = row["ticker"]
        live = live_data.get(tkr, {})
        signal_history[tkr] = get_signal_history(
            tkr, oof_signals,
            live.get("chart_dates", []),
            live.get("chart_prices", []),
        )
        n_b = len(signal_history[tkr]["buy"])
        n_s = len(signal_history[tkr]["sell"])
        print(f"  {tkr:<22}  BUY×{n_b}  SELL×{n_s}")

    def make_card(row, sig_class):
        tkr  = row["ticker"]
        live = live_data.get(tkr, {})
        sec  = live.get("sector", "")
        sn   = get_sector_news(sec) if sec else []
        sm   = signal_history.get(tkr)
        return build_card(row, live, sig_class, sn, macro_news, sm)

    buy_cards  = "\n".join(make_card(r, "buy")  for _, r in buys.iterrows())
    sell_cards = "\n".join(make_card(r, "sell") for _, r in sells.iterrows())

    n_fresh = int((signals["age_days"] <= 30).sum())
    html = PAGE_HTML.format(
        css           = CSS,
        updated       = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
        total_tickers = len(signals),
        n_buys        = (signals["signal"] == "BUY").sum(),
        n_sells       = (signals["signal"] == "SELL").sum(),
        top_pred      = fmt_pct(signals["pred_3m"].max()),
        n_fresh       = n_fresh,
        buy_cards     = buy_cards,
        sell_cards    = sell_cards,
    )

    out = _OUT / "watchlist_live.html"
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"\nSaved → {out}  ({size_kb} KB)")
    print(f"Open:   file://{out.absolute()}")
