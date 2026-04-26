"""
SEC EDGAR Form 4 insider transaction fetcher (US stocks only).

Data source: SEC EDGAR — https://www.sec.gov/cgi-bin/browse-edgar
Official US government data, completely free, no API key required.

What we extract:
  - Open-market purchases (code P) — strongest bullish insider signal
  - Open-market sales   (code S) — bearish signal
  - Awards/grants (A, F) — excluded (compensation, not conviction)

Output metrics per ticker per lookback window:
  insider_net_value     : dollar value of P minus S (positive = net buying)
  insider_n_buyers      : distinct insiders making open-market purchases
  insider_n_sellers     : distinct insiders making open-market sales
  insider_buy_count     : number of buy transactions
  insider_sell_count    : number of sell transactions
  insider_buy_sell_ratio: n_buyers / (n_buyers + n_sellers), NaN if zero

Rate limiting: SEC asks for ≤10 requests/second. We sleep between calls.
"""

import re
import time
import math
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

_HEADERS = {
    "User-Agent": "ProjectIris/1.0 subhraba01@gmail.com",  # SEC requires identification
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
_HEADERS_WWW = {
    "User-Agent": "ProjectIris/1.0 subhraba01@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

_CIK_MAP: dict | None = None
_CACHE_DIR = Path("ml_output/sec_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_cik_map() -> dict:
    """Load the SEC ticker→CIK mapping (cached locally)."""
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP

    cache_file = _CACHE_DIR / "company_tickers.json"
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400 * 7:
        with open(cache_file) as f:
            raw = json.load(f)
    else:
        print("[sec_insiders] Downloading SEC company ticker map …")
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_HEADERS_WWW, timeout=30
        )
        resp.raise_for_status()
        raw = resp.json()
        with open(cache_file, "w") as f:
            json.dump(raw, f)
        time.sleep(0.2)

    _CIK_MAP = {v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                for v in raw.values()}
    return _CIK_MAP


def _get_cik(ticker: str) -> str | None:
    clean = ticker.upper().replace(".NS", "").replace(".BO", "")
    return _load_cik_map().get(clean)


def _fetch_submissions(cik: str) -> dict:
    """Fetch the EDGAR submissions JSON for a CIK."""
    cache_file = _CACHE_DIR / f"sub_{cik}.json"
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 3600 * 12:
        with open(cache_file) as f:
            return json.load(f)

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    with open(cache_file, "w") as f:
        json.dump(data, f)
    time.sleep(0.12)
    return data


def _parse_form4_xml(xml_text: str) -> list[dict]:
    """
    Parse a Form 4 XML document.
    Returns list of dicts with: date, code, shares, price_per_share,
    transaction_type ('buy' | 'sell'), reporter_name.
    Only open-market buys (P) and sells (S) are returned.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    def _text(el, tag, default=None):
        node = el.find(f".//{tag}")
        return node.text.strip() if node is not None and node.text else default

    reporter = _text(root, "rptOwnerName") or "Unknown"
    rows = []

    for txn in root.findall(".//nonDerivativeTransaction"):
        code   = _text(txn, "transactionCode", "")
        if code not in ("P", "S"):
            continue
        date_s = _text(txn, "transactionDate/value", "")
        try:
            date   = datetime.strptime(date_s, "%Y-%m-%d").date()
        except ValueError:
            continue
        shares_s = _text(txn, "transactionShares/value", "")
        price_s  = _text(txn, "transactionPricePerShare/value", "")
        try:
            shares = float(shares_s)
            price  = float(price_s)
        except (ValueError, TypeError):
            continue

        rows.append({
            "date":           date,
            "code":           code,
            "shares":         shares,
            "price":          price,
            "dollar_value":   shares * price,
            "transaction_type": "buy" if code == "P" else "sell",
            "reporter":       reporter,
        })
    return rows


def fetch_insider_transactions(ticker: str,
                               lookback_days: int = 180) -> pd.DataFrame:
    """
    Fetch open-market insider transactions (Form 4) for a US ticker
    from SEC EDGAR for the past `lookback_days`.

    Returns DataFrame with columns:
      date, code, shares, price, dollar_value, transaction_type, reporter
    Empty DataFrame if no data or ticker not found.
    """
    cik = _get_cik(ticker)
    if cik is None:
        return pd.DataFrame()

    try:
        subs = _fetch_submissions(cik)
    except Exception as e:
        print(f"[sec_insiders] {ticker}: submissions fetch failed — {e}")
        return pd.DataFrame()

    cutoff = datetime.today().date() - timedelta(days=lookback_days)

    # Recent filings are in subs["filings"]["recent"]
    filings = subs.get("filings", {}).get("recent", {})
    forms     = filings.get("form", [])
    acc_nums  = filings.get("accessionNumber", [])
    doc_dates = filings.get("filingDate", [])

    all_rows = []
    for form, acc, dated in zip(forms, acc_nums, doc_dates):
        if form != "4":
            continue
        try:
            filing_date = datetime.strptime(dated, "%Y-%m-%d").date()
        except ValueError:
            continue
        if filing_date < cutoff:
            break  # filings are newest-first

        # Build the filing index URL
        acc_clean = acc.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{acc_clean}/{acc}.txt"
        )
        # Try the XML filing directly
        xml_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{acc_clean}/"
        )
        try:
            # Fetch filing index to find the XML document name
            idx_resp = requests.get(
                f"https://data.sec.gov/submissions/CIK{cik}.json",
                headers=_HEADERS, timeout=20
            )
            # Use the primary document search path
            primary_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{acc_clean}/{acc}-index.htm"
            )
            idx_r = requests.get(primary_url, headers=_HEADERS_WWW, timeout=20)
            # Find XML link in index HTML
            matches = re.findall(
                r'href="(/Archives/edgar/data/[^"]+\.xml)"',
                idx_r.text
            )
            if not matches:
                time.sleep(0.12)
                continue
            xml_full_url = "https://www.sec.gov" + matches[0]
            xml_resp = requests.get(xml_full_url, headers=_HEADERS_WWW, timeout=20)
            xml_resp.raise_for_status()
            rows = _parse_form4_xml(xml_resp.text)
            all_rows.extend(rows)
            time.sleep(0.12)
        except Exception:
            time.sleep(0.12)
            continue

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"].dt.date >= cutoff].reset_index(drop=True)


def insider_summary(ticker: str, lookback_days: int = 180) -> dict:
    """
    Summarise insider activity into model-ready float features.
    Returns dict with keys: insider_net_value, insider_n_buyers,
    insider_n_sellers, insider_buy_count, insider_sell_count,
    insider_buy_sell_ratio.
    All NaN if data unavailable.
    """
    nan = float('nan')
    df  = fetch_insider_transactions(ticker, lookback_days)

    if df.empty:
        return {
            'insider_net_value':      nan,
            'insider_n_buyers':       nan,
            'insider_n_sellers':      nan,
            'insider_buy_count':      nan,
            'insider_sell_count':     nan,
            'insider_buy_sell_ratio': nan,
        }

    buys  = df[df["transaction_type"] == "buy"]
    sells = df[df["transaction_type"] == "sell"]

    buy_val  = buys["dollar_value"].sum()
    sell_val = sells["dollar_value"].sum()

    n_buyers  = buys["reporter"].nunique()
    n_sellers = sells["reporter"].nunique()
    denom = n_buyers + n_sellers

    return {
        'insider_net_value':      float(buy_val - sell_val),
        'insider_n_buyers':       float(n_buyers),
        'insider_n_sellers':      float(n_sellers),
        'insider_buy_count':      float(len(buys)),
        'insider_sell_count':     float(len(sells)),
        'insider_buy_sell_ratio': float(n_buyers / denom) if denom > 0 else nan,
    }
