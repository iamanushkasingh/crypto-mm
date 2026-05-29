"""Generic tick replay harness.

Takes any iterable of (BookUpdate | Trade) events (sorted by ts) and feeds
them into the Engine in event-time order.
"""
from __future__ import annotations

import time
from typing import Iterable, Optional, Union

from core.events import BookUpdate, Trade
from strategy.engine import Engine


def replay(events: Iterable[Union[BookUpdate, Trade]], engine: Engine,
           verbose: bool = False, max_events: Optional[int] = None,
           progress_every: int = 50_000) -> int:
    """Stream `events` into `engine`. Prints progress every `progress_every`
    events so a long historical replay doesn't look hung.
    """
    n = 0
    t0 = time.time()
    last_log = t0
    for ev in events:
        if isinstance(ev, BookUpdate):
            engine.on_book(ev)
        else:
            engine.on_trade(ev)
        n += 1
        if n % progress_every == 0:
            now = time.time()
            rate = progress_every / max(now - last_log, 1e-6)
            pnl = engine.inv.total_pnl()
            print(f"[replay] {n:>10,} events  {rate:>7,.0f} ev/s  "
                  f"pnl={pnl:+.4f}  fills={engine.inv.n_maker_fills}M/"
                  f"{engine.inv.n_taker_fills}T", flush=True)
            last_log = now
        if max_events and n >= max_events:
            print(f"[replay] reached --max-events={max_events}, stopping early.")
            break
    print(f"[replay] done. {n:,} events in {time.time() - t0:.2f}s.")
    return n
