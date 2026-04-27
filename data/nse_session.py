"""
Shared NSE session module.

All code that needs to call NSE APIs should import from here:
  from data.nse_session import nse_get

Features:
  - Auto-initialises session cookie from NSE homepage
  - Exponential backoff with 3 retries on 401/403/timeout
  - Graceful return of None on persistent failure
"""

import time
import requests

_SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}

_session = requests.Session()
_session_initialized = False
GRACEFUL_FAIL_ON_NSE_ERROR = True   # return None instead of raising


def _init_session() -> None:
    global _session_initialized
    if _session_initialized:
        return
    try:
        _session.get("https://www.nseindia.com",
                     headers=_SESSION_HEADERS, timeout=15)
        time.sleep(1.0)
        _session_initialized = True
    except Exception:
        pass


def nse_get(url: str, params: dict | None = None,
            max_retries: int = 3) -> dict | list | None:
    """
    Make an authenticated NSE API GET request.
    Retries up to max_retries times with exponential back-off (1s, 2s, 4s).
    Returns None on persistent failure when GRACEFUL_FAIL_ON_NSE_ERROR=True.
    """
    global _session_initialized
    _init_session()

    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = _session.get(url, params=params,
                                headers=_SESSION_HEADERS, timeout=20)
            if resp.status_code in (401, 403):
                _session_initialized = False
                _init_session()
                resp = _session.get(url, params=params,
                                    headers=_SESSION_HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                if not GRACEFUL_FAIL_ON_NSE_ERROR:
                    raise
                return None
    return None
