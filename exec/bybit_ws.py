"""Bybit linear-perp public WS client → yields BookUpdate + Trade events.

Streams used (v5):
  - orderbook.1.BTCUSDT    (top-of-book, full snapshot every update)
  - publicTrade.BTCUSDT
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator, Union

from core.events import BookUpdate, Trade, Venue, Side


WS_URL = "wss://stream.bybit.com/v5/public/linear"


async def stream() -> AsyncIterator[Union[BookUpdate, Trade]]:
    import websockets
    sub = {"op": "subscribe",
           "args": ["orderbook.1.BTCUSDT", "publicTrade.BTCUSDT"]}
    async for ws in websockets.connect(WS_URL, ping_interval=20, ping_timeout=10):
        try:
            await ws.send(json.dumps(sub))
            async for raw in ws:
                msg = json.loads(raw)
                topic = msg.get("topic", "")
                ts = int(time.time_ns())
                if topic.startswith("orderbook.1."):
                    d = msg.get("data", {})
                    bids = [(float(p), float(q)) for p, q in d.get("b", [])]
                    asks = [(float(p), float(q)) for p, q in d.get("a", [])]
                    yield BookUpdate(ts=ts, venue=Venue.BYBIT,
                                     bids=bids, asks=asks, is_snapshot=True)
                elif topic.startswith("publicTrade."):
                    for t in msg.get("data", []):
                        side = t.get("S", "").lower()
                        yield Trade(
                            ts=ts, venue=Venue.BYBIT,
                            price=float(t["p"]), size=float(t["v"]),
                            aggressor=Side.BID if side == "buy" else Side.ASK,
                        )
        except Exception as e:
            print(f"[bybit_ws] reconnecting: {e}")
            await asyncio.sleep(1.0)
            continue
