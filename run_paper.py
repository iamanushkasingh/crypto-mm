"""Live paper-trading driver.

Connects to Binance + Bybit public WS feeds and feeds events into the same
Engine. Nothing is sent to exchanges — fills are paper-simulated.

Run:
    pip install websockets pyyaml sortedcontainers
    python run_paper.py
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.events import BookUpdate, Trade
from strategy.engine import Engine
from backtest.analyze import print_summary, plot_equity


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def _pump(stream, queue: asyncio.Queue, label: str) -> None:
    async for ev in stream:
        await queue.put(ev)


async def _consume(engine: Engine, queue: asyncio.Queue, args) -> None:
    from core.events import Venue
    n = 0
    last_log = time.time()
    while True:
        ev = await queue.get()
        if isinstance(ev, BookUpdate):
            engine.on_book(ev)
        else:
            engine.on_trade(ev)
        n += 1
        now = time.time()
        if now - last_log > 5.0:
            s = engine.inv.summary()
            # Show strategy state, not just fill counts — so it's obvious
            # the engine IS quoting even when nothing's filled yet.
            bb = engine.books[Venue.BINANCE].best_bid()
            ba = engine.books[Venue.BINANCE].best_ask()
            n_resting = len(engine.sim.resting)
            quoter_b = engine.quoter.bid_quote
            quoter_a = engine.quoter.ask_quote
            mid = engine.books[Venue.BINANCE].mid() or 0
            bid_off = ((mid - quoter_b.price) / mid * 1e4) if (quoter_b and mid) else 0
            ask_off = ((quoter_a.price - mid) / mid * 1e4) if (quoter_a and mid) else 0
            spread_bps = ((ba[0] - bb[0]) / mid * 1e4) if (bb and ba and mid) else 0
            print(f"[paper] ev={n:>7} pnl={s['pnl_total']:+.4f} "
                  f"d={s['net_delta']:+.5f} "
                  f"fills={s['n_maker_fills']}M/{s['n_taker_fills']}T "
                  f"| bnc_spread={spread_bps:.2f}bps "
                  f"quotes={n_resting} bid_off={bid_off:.2f}bps ask_off={ask_off:.2f}bps",
                  flush=True)
            last_log = now


async def main_async(args) -> None:
    cfg = _load_cfg(args.config)
    engine = Engine(cfg)

    from exec.binance_ws import stream as binance_stream
    from exec.bybit_ws import stream as bybit_stream

    queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
    tasks = [
        asyncio.create_task(_pump(binance_stream(), queue, "binance")),
        asyncio.create_task(_pump(bybit_stream(), queue, "bybit")),
        asyncio.create_task(_consume(engine, queue, args)),
    ]

    stop = asyncio.Event()
    def _on_signal(*_):
        stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            pass  # Windows

    try:
        if args.seconds:
            await asyncio.wait_for(stop.wait(), timeout=args.seconds)
        else:
            await stop.wait()
    except asyncio.TimeoutError:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print_summary(engine)
        plot_equity(engine, str(Path(__file__).resolve().parent / "plots" / "equity_paper.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"),
                    help="path to config.yaml; use live_demo_config.yaml to quote at the touch")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="auto-stop after N seconds; 0 = run until Ctrl-C")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
