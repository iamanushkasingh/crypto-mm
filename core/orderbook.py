"""Sorted L2 order book.

Apply BookUpdate events; query best bid/ask, depth, and volume at a price.
Uses SortedDict so insertion / deletion / best-price lookups are O(log n).
"""
from __future__ import annotations

from typing import Optional, Tuple

from sortedcontainers import SortedDict

from .events import BookUpdate, Side


class OrderBook:
    def __init__(self, venue: str, tick_size: float = 0.1) -> None:
        self.venue = venue
        self.tick_size = tick_size
        # bids: price desc; asks: price asc. We store both as price->size and
        # rely on .peekitem with explicit index for the best price.
        self.bids: SortedDict[float, float] = SortedDict()
        self.asks: SortedDict[float, float] = SortedDict()
        self.last_ts: int = 0
        self.ready: bool = False

    # ----- mutation -----

    def apply(self, u: BookUpdate) -> None:
        if u.is_snapshot:
            self.bids.clear()
            self.asks.clear()
        for price, size in u.bids:
            if size <= 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size
        for price, size in u.asks:
            if size <= 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size
        self.last_ts = u.ts
        if u.is_snapshot:
            self.ready = True

    # ----- queries -----

    def best_bid(self) -> Optional[Tuple[float, float]]:
        if not self.bids:
            return None
        p = self.bids.keys()[-1]    # max
        return p, self.bids[p]

    def best_ask(self) -> Optional[Tuple[float, float]]:
        if not self.asks:
            return None
        p = self.asks.keys()[0]     # min
        return p, self.asks[p]

    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return 0.5 * (bb[0] + ba[0])

    def microprice(self) -> Optional[float]:
        """Size-weighted top-of-book price; better mid for predicting next trade."""
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        bp, bs = bb
        ap, asz = ba
        return (bp * asz + ap * bs) / (bs + asz)

    def spread_bps(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        m = self.mid()
        if bb is None or ba is None or m is None or m == 0:
            return None
        return 1e4 * (ba[0] - bb[0]) / m

    def size_at(self, side: Side, price: float) -> float:
        book = self.bids if side == Side.BID else self.asks
        return book.get(price, 0.0)

    def cum_size_to_price(self, side: Side, price: float) -> float:
        """Sum of sizes from best up to (and including) price.

        For BID side: sum of all bid sizes at prices >= price.
        For ASK side: sum of all ask sizes at prices <= price.
        Used to estimate queue depth ahead of our quote.
        """
        if side == Side.BID:
            total = 0.0
            for p in reversed(self.bids.keys()):
                if p < price:
                    break
                total += self.bids[p]
            return total
        else:
            total = 0.0
            for p in self.asks.keys():
                if p > price:
                    break
                total += self.asks[p]
            return total
