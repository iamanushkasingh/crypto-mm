"""Synthetic tick generator for fast offline smoke tests.

Mid-price follows a mean-reverting GBM. Each tick produces either a book
update on a randomly chosen venue (most events) or a trade (small fraction
of events). Book updates refresh the top 5 levels per side with a slight
random walk around mid.
"""
from __future__ import annotations

import math
import random
from typing import Iterator, Tuple, Union

from core.events import BookUpdate, Trade, Venue, Side


def generate(
    minutes: float = 5.0,
    tps: int = 400,
    mid0: float = 65000.0,
    sigma_per_min: float = 25.0,
    spread_ticks: int = 1,
    tick_size: float = 0.1,
    seed: int = 7,
    sweep_prob: float = 0.005,    # fraction of trades that aggressively sweep
    sweep_max_bps: float = 8.0,    # how far into the book sweeps go
) -> Iterator[Union[BookUpdate, Trade]]:
    rng = random.Random(seed)
    n_events = int(minutes * 60 * tps)
    dt_ns = int(1e9 / tps)
    t0 = 1_700_000_000_000_000_000   # nominal nanosecond epoch
    mid = mid0
    sigma_per_event = sigma_per_min / math.sqrt(60.0 * tps)
    # Emit initial snapshots
    for venue in (Venue.BINANCE, Venue.BYBIT):
        yield _snapshot(t0, venue, mid, spread_ticks, tick_size, rng)
    ts = t0
    for i in range(n_events):
        ts += dt_ns
        mid = max(1.0, mid + rng.gauss(0, sigma_per_event)
                       - 0.0005 * (mid - mid0))
        # 8% trade, 92% book update
        if rng.random() < 0.08:
            aggressor = Side.BID if rng.random() < 0.5 else Side.ASK
            venue = Venue.BINANCE if rng.random() < 0.5 else Venue.BYBIT
            half = spread_ticks * tick_size / 2
            if rng.random() < sweep_prob:
                # Aggressive sweep: trade prints deep into the book and is large.
                depth_bps = rng.uniform(1.0, sweep_max_bps)
                offset = mid * depth_bps * 1e-4
                price = mid + offset if aggressor == Side.BID else mid - offset
                size = max(0.5, rng.expovariate(0.5))
            else:
                price = mid + (half if aggressor == Side.BID else -half)
                size = max(0.001, rng.expovariate(50.0))
            yield Trade(ts=ts, venue=venue, price=round(price / tick_size) * tick_size,
                        size=size, aggressor=aggressor)
        else:
            venue = Venue.BINANCE if rng.random() < 0.5 else Venue.BYBIT
            # small incremental refresh of the top few levels
            half = spread_ticks * tick_size / 2
            bids = []
            asks = []
            for k in range(5):
                bp = round((mid - half - k * tick_size) / tick_size) * tick_size
                ap = round((mid + half + k * tick_size) / tick_size) * tick_size
                bids.append((bp, max(0.001, rng.expovariate(5.0))))
                asks.append((ap, max(0.001, rng.expovariate(5.0))))
            # Send as snapshot so stale levels don't accumulate.
            yield BookUpdate(ts=ts, venue=venue, bids=bids, asks=asks, is_snapshot=True)


def _snapshot(ts: int, venue: Venue, mid: float, spread_ticks: int,
              tick_size: float, rng: random.Random) -> BookUpdate:
    half = spread_ticks * tick_size / 2
    bids = []
    asks = []
    for k in range(20):
        bp = round((mid - half - k * tick_size) / tick_size) * tick_size
        ap = round((mid + half + k * tick_size) / tick_size) * tick_size
        bids.append((bp, max(0.01, rng.expovariate(2.0))))
        asks.append((ap, max(0.01, rng.expovariate(2.0))))
    return BookUpdate(ts=ts, venue=venue, bids=bids, asks=asks, is_snapshot=True)
