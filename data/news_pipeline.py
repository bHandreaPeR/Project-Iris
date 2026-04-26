"""
3-tier news sentiment pipeline.

Tier 1 — Direct    : news directly about the ticker (company-specific events)
Tier 2 — Sectoral  : news about the ticker's industry sector
Tier 3 — Macro     : broad market / economic / geopolitical news

Data source: GDELT Project (https://www.gdeltproject.org)
  — Largest open news database in the world, completely free, no API key.
  — 65+ languages, updated every 15 minutes.
  — Doc API v2: returns article titles, URLs, tones, themes.

Sentiment model: ProsusAI/finbert (HuggingFace)
  — Fine-tuned BERT on financial news (FiQA, Financial PhraseBank).
  — Outputs: positive / negative / neutral with confidence scores.
  — First run downloads ~440 MB model; cached locally by HuggingFace.
  — Falls back to VADER lexicon-based scorer if transformers unavailable.

Output per (ticker, date window):
  news_direct_score    : weighted avg sentiment, Tier 1  (-1 to +1)
  news_direct_n        : number of articles scored
  news_sector_score    : weighted avg sentiment, Tier 2
  news_sector_n        : article count
  news_macro_score     : weighted avg sentiment, Tier 3
  news_macro_n         : article count
"""

import time
import math
import hashlib
import json
from pathlib import Path
from urllib.parse import quote_plus

import requests
import pandas as pd

_CACHE_DIR = Path("ml_output/news_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

_MACRO_QUERY = (
    '"stock market" OR "interest rates" OR "Federal Reserve" '
    'OR "inflation" OR "GDP" OR "recession" OR "RBI" '
    'OR "SEBI" OR "fiscal" OR "monetary policy"'
)

_SECTOR_QUERIES = {
    "Technology":       "technology software cloud AI semiconductor",
    "Financial":        "banking finance insurance credit loan",
    "Healthcare":       "pharmaceutical biotech hospital healthcare drug",
    "Energy":           "oil gas energy crude petroleum refinery",
    "Consumer":         "retail consumer FMCG e-commerce",
    "Industrials":      "manufacturing infrastructure construction logistics",
    "Materials":        "steel metals mining chemicals commodities",
    "Utilities":        "power electricity utility grid renewable",
    "Real Estate":      "real estate property housing construction REIT",
    "Communication":    "telecom media streaming advertising",
    "Automobile":       "automobile EV electric vehicle auto parts",
}


# ---------------------------------------------------------------------------
# GDELT fetcher
# ---------------------------------------------------------------------------

def _gdelt_fetch(query: str, timespan: str = "30d",
                 max_records: int = 25) -> list[dict]:
    """
    Query GDELT Doc API v2.  Returns list of article dicts with 'title', 'url',
    'seendate', 'domain', 'language'.  Caches results for 6 hours.
    """
    cache_key = hashlib.md5(f"{query}{timespan}".encode()).hexdigest()[:12]
    cache_file = _CACHE_DIR / f"gdelt_{cache_key}.json"

    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 21600:
        with open(cache_file) as f:
            return json.load(f)

    params = {
        "query":       query,
        "mode":        "artlist",
        "maxrecords":  str(max_records),
        "timespan":    timespan,
        "sort":        "date",
        "format":      "json",
        "sourcelang":  "english",
    }
    try:
        resp = requests.get(_GDELT_DOC_API, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("articles", [])
    except Exception as e:
        print(f"[news] GDELT fetch failed: {e}")
        articles = []

    with open(cache_file, "w") as f:
        json.dump(articles, f)
    time.sleep(0.5)
    return articles


# ---------------------------------------------------------------------------
# Sentiment scorer
# ---------------------------------------------------------------------------

_finbert_pipeline = None
_use_vader = False


def _get_scorer():
    global _finbert_pipeline, _use_vader
    if _finbert_pipeline is not None or _use_vader:
        return _finbert_pipeline

    try:
        from transformers import pipeline as hf_pipeline
        import torch
        device = 0 if torch.cuda.is_available() else (
            0 if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else -1
        )
        _finbert_pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=device,
            truncation=True,
            max_length=512,
        )
        print("[news] FinBERT loaded.")
    except ImportError:
        print("[news] transformers/torch not installed — using VADER fallback.")
        _use_vader = True

    return _finbert_pipeline


def _score_texts(texts: list[str]) -> list[float]:
    """
    Score a list of texts → float sentiment in [-1, +1].
    +1 = very positive, -1 = very negative, 0 = neutral.
    """
    if not texts:
        return []

    scorer = _get_scorer()

    if _use_vader:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            return [sia.polarity_scores(t)["compound"] for t in texts]
        except ImportError:
            return [0.0] * len(texts)

    scores = []
    # FinBERT batch scoring (chunk to avoid OOM)
    chunk_size = 16
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i: i + chunk_size]
        try:
            results = scorer(chunk)
            for r in results:
                label = r["label"].lower()
                conf  = r["score"]
                if label == "positive":
                    scores.append(conf)
                elif label == "negative":
                    scores.append(-conf)
                else:
                    scores.append(0.0)
        except Exception:
            scores.extend([0.0] * len(chunk))
    return scores


def _weighted_avg(scores: list[float]) -> float:
    """Average sentiment score; NaN if empty."""
    valid = [s for s in scores if not math.isnan(s)]
    return float(sum(valid) / len(valid)) if valid else float('nan')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news_sentiment(ticker: str,
                         company_name: str = "",
                         sector: str = "",
                         timespan: str = "30d") -> dict:
    """
    Fetch and score 3-tier news sentiment for a ticker.

    Args:
        ticker       : yfinance-format ticker (e.g. 'AAPL' or 'RELIANCE.NS')
        company_name : full company name for better GDELT queries
        sector       : GICS-style sector (matched to _SECTOR_QUERIES)
        timespan     : GDELT timespan string ('7d', '30d', '90d')

    Returns dict with keys: news_direct_score, news_direct_n,
    news_sector_score, news_sector_n, news_macro_score, news_macro_n.
    """
    nan = float('nan')
    base_ticker = ticker.replace(".NS", "").replace(".BO", "")

    # ── Tier 1: Direct ────────────────────────────────────────────────────
    direct_query = f'"{base_ticker}"'
    if company_name:
        short_name = company_name.split()[0] if company_name else ""
        direct_query = f'"{base_ticker}" OR "{short_name}"' if short_name else direct_query

    direct_articles = _gdelt_fetch(direct_query, timespan, max_records=25)
    direct_texts    = [a.get("title", "") for a in direct_articles if a.get("title")]
    direct_scores   = _score_texts(direct_texts)

    # ── Tier 2: Sectoral ─────────────────────────────────────────────────
    sector_q  = _SECTOR_QUERIES.get(sector, "")
    sector_articles, sector_scores = [], []
    if sector_q:
        sector_articles = _gdelt_fetch(sector_q, timespan, max_records=30)
        sector_texts    = [a.get("title", "") for a in sector_articles if a.get("title")]
        sector_scores   = _score_texts(sector_texts)

    # ── Tier 3: Macro ─────────────────────────────────────────────────────
    macro_articles = _gdelt_fetch(_MACRO_QUERY, timespan, max_records=40)
    macro_texts    = [a.get("title", "") for a in macro_articles if a.get("title")]
    macro_scores   = _score_texts(macro_texts)

    return {
        'news_direct_score':  _weighted_avg(direct_scores),
        'news_direct_n':      float(len(direct_scores)),
        'news_sector_score':  _weighted_avg(sector_scores),
        'news_sector_n':      float(len(sector_scores)),
        'news_macro_score':   _weighted_avg(macro_scores),
        'news_macro_n':       float(len(macro_scores)),
    }
