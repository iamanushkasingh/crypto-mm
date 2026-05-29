"""Download historical tick data for Binance and Bybit BTC-USDT perp.

Binance offers public archives at https://data.binance.vision/  — we use the
USDⓂ-perp aggTrades and bookTicker CSVs. Bybit offers similar archives at
https://public.bybit.com/.

This script just downloads + caches them under ./data/cache. Tick-by-tick
L2 depth archives are large (gigabytes per day), so by default we use the
aggTrades + bookTicker streams, which is sufficient for our queue-position
fill model: bookTicker gives best-bid/ask updates, aggTrades gives every
print.

Run:
    python data/fetch_historical.py --date 2024-05-01 --hours 1
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)


def _download(url: str, dest: Path) -> bool:
    """Returns True on success (or already cached), False on 404."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache] {dest.name}")
        return True
    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"  -> {dest}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  [warn] 404 — not available, will fall back")
            return False
        raise


def fetch_binance(date: str) -> dict[str, Path]:
    """date format: YYYY-MM-DD. Returns paths to whichever streams we got."""
    sym = "BTCUSDT"
    out = {}
    # Try daily first, then monthly (some streams aren't always published daily)
    for stream in ("aggTrades", "bookTicker"):
        dest = CACHE / f"binance_{stream}_{date}.zip"
        # daily archive
        daily_url = (f"https://data.binance.vision/data/futures/um/daily/"
                     f"{stream}/{sym}/{sym}-{stream}-{date}.zip")
        if _download(daily_url, dest):
            out[stream] = dest
            continue
        # monthly archive fallback (YYYY-MM)
        ym = date[:7]
        monthly_url = (f"https://data.binance.vision/data/futures/um/monthly/"
                       f"{stream}/{sym}/{sym}-{stream}-{ym}.zip")
        monthly_dest = CACHE / f"binance_{stream}_{ym}.zip"
        if _download(monthly_url, monthly_dest):
            out[stream] = monthly_dest
            continue
        print(f"  [skip] no archive for binance {stream}; "
              "reader will synthesize from aggTrades if possible.")
    return out


def fetch_bybit(date: str) -> dict[str, Path]:
    """Bybit public archive layout. Files are gzipped CSVs."""
    sym = "BTCUSDT"
    out = {}
    for stream in ("trading",):    # "kline_1m" also available
        fn = f"{sym}{date}.csv.gz"
        url = f"https://public.bybit.com/{stream}/{sym}/{fn}"
        dest = CACHE / f"bybit_{stream}_{date}.csv.gz"
        try:
            _download(url, dest)
            out[stream] = dest
        except Exception as e:
            print(f"  [warn] bybit {stream} for {date} not available: {e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    fetch_binance(args.date)
    fetch_bybit(args.date)


if __name__ == "__main__":
    main()
