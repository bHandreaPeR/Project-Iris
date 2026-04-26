"""
Entry point for Project Iris.

Manual run:
    python run_screener.py --market india
    python run_screener.py --market us
    python run_screener.py --market all

Scheduled (daily pre-market) daemon:
    python run_screener.py --daemon
    (runs indefinitely; uses times from config.SCHEDULE)
"""

import argparse
import datetime
import time

import config
import engine


def _run_market(market: str) -> None:
    print(f"\n{'='*50}")
    print(f"[iris] Starting screen: {market.upper()}")
    print(f"[iris] Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    try:
        engine.run(market)
    except Exception as e:
        print(f"[iris] ERROR running {market}: {e}")


def _daemon() -> None:
    """Simple day-level scheduler using a polling loop."""
    import zoneinfo

    india_tz = zoneinfo.ZoneInfo('Asia/Kolkata')
    us_tz    = zoneinfo.ZoneInfo('America/New_York')

    india_hm = tuple(int(x) for x in config.SCHEDULE['india_run_time'].split(':'))
    us_hm    = tuple(int(x) for x in config.SCHEDULE['us_run_time'].split(':'))

    fired_today: set[str] = set()
    print("[iris] Daemon started. Waiting for scheduled times …")

    while True:
        now_india = datetime.datetime.now(india_tz)
        now_us    = datetime.datetime.now(us_tz)
        date_key  = now_india.date().isoformat()

        if date_key not in fired_today:
            fired_today.clear()

        if config.SCHEDULE['weekdays_only'] and now_india.weekday() >= 5:
            time.sleep(3600)
            continue

        india_key = f"india_{date_key}"
        if (india_key not in fired_today
                and (now_india.hour, now_india.minute) >= india_hm):
            _run_market('india')
            fired_today.add(india_key)

        us_key = f"us_{date_key}"
        if (us_key not in fired_today
                and (now_us.hour, now_us.minute) >= us_hm):
            _run_market('us')
            fired_today.add(us_key)

        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description='Project Iris Stock Screener')
    parser.add_argument('--market', choices=['india', 'us', 'all'],
                        help='Market to screen (run once)')
    parser.add_argument('--daemon', action='store_true',
                        help='Run as a daily scheduled daemon')
    args = parser.parse_args()

    if args.daemon:
        _daemon()
    elif args.market == 'all':
        _run_market('india')
        _run_market('us')
    elif args.market:
        _run_market(args.market)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
