"""Binance USDⓂ-perp public WS client → yields BookUpdate + Trade events.

Streams used:
  - btcusdt@bookTicker     (best bid/ask updates, every change)
  - btcusdt@aggTrade       (aggregated trades)
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Union

from core.events import BookUpdate, Trade, Venue, Side


WS_URL = "wss://fstream.binance.com/stream?streams=btcusdt@bookTicker/btcusdt@aggTrade"


async def stream() -> AsyncIterator[Union[BookUpdate, Trade]]:
    import websockets
    async for ws in websockets.connect(WS_URL, ping_interval=20, ping_timeout=10):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                stream_name = msg.get("stream", "")
                data = msg.get("data", {})
                ts = int(time.time_ns())
                if "bookTicker" in stream_name:
                    yield BookUpdate(
                        ts=ts, venue=Venue.BINANCE,
                        bids=[(float(data["b"]), float(data["B"]))],
                        asks=[(float(data["a"]), float(data["A"]))],
                        is_snapshot=True,
                    )
                elif "aggTrade" in stream_name:
                    buyer_is_maker = bool(data.get("m"))
                    yield Trade(
                        ts=ts, venue=Venue.BINANCE,
                        price=float(data["p"]), size=float(data["q"]),
                        aggressor=Side.ASK if buyer_is_maker else Side.BID,
                    )
        except Exception as e:
            print(f"[binance_ws] reconnecting: {e}")
            await asyncio.sleep(1.0)
            continue
