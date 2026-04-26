import os
from dotenv import load_dotenv

load_dotenv('.env')

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', '').strip(),
    'chat_ids': [c.strip() for c in os.getenv('TELEGRAM_CHAT_IDS', '').split(',') if c.strip()],
}

# ---------------------------------------------------------------------------
# Scheduler — times are local machine time (set TZ appropriately)
# ---------------------------------------------------------------------------
SCHEDULE = {
    'india_run_time': '08:45',   # IST, before NSE open (09:15)
    'us_run_time':    '08:30',   # ET,  before NYSE open (09:30)
    'weekdays_only': True,
}

# ---------------------------------------------------------------------------
# Fundamental filter thresholds
# ---------------------------------------------------------------------------
FUNDAMENTAL_FILTERS = {
    'max_pe':           30.0,    # Trailing P/E
    'max_pb':           5.0,     # Price-to-Book
    'max_debt_equity':  1.5,     # Debt / Equity
    'min_roe':          0.12,    # Return on Equity (12%)
    'min_revenue_growth': 0.05,  # YoY revenue growth (5%)
}

# ---------------------------------------------------------------------------
# Price / volume anomaly thresholds
# ---------------------------------------------------------------------------
PRICE_VOLUME_FILTERS = {
    'volume_multiplier':   2.0,   # Flag if today's vol > 2× 20-day avg
    'near_52w_high_pct':   5.0,   # Flag if price within 5% of 52-week high
    'near_52w_low_pct':    5.0,   # Flag if price within 5% of 52-week low
    'min_gap_pct':         2.0,   # Flag gap up/down > 2% from prev close
    'top_movers_n':        10,    # Number of top gainers/losers to include
    'min_avg_volume':   100_000,  # Ignore illiquid stocks below this avg vol
}

# ---------------------------------------------------------------------------
# Engine — how to combine fundamental + price_volume hits
# ---------------------------------------------------------------------------
ENGINE = {
    # 'union'     → flag any ticker that passes EITHER screen
    # 'intersect' → flag only tickers that pass BOTH screens
    'combine_mode': 'union',
    'max_results_per_market': 20,
}

# ---------------------------------------------------------------------------
# Stock universe
# (imported from data/universe.py at runtime to keep this file concise)
# ---------------------------------------------------------------------------
