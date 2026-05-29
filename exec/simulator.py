"""Paper-trading execution simulator.

Keeps track of our resting orders per venue and emits Fill events when:
  - A public trade walks through one of our maker price levels
    (we estimate queue position and fill the contracts that hit us).
  - A taker order is submitted: it fills immediately at best opposite quote.

This is intentionally pessimistic about maker fills: a maker order at the
touch is only filled if the trade volume after subtracting the queue ahead
of us actually reaches our order. Conservative is the right default for a
deep-book MM backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from core.events import Fill, Side, Trade, Venue
from core.orderbook import OrderBook
from strategy.fill_model import QueuePosition
from strategy.quoter import Quote
from strategy.hedger import HedgeOrder


@dataclass
class _Resting:
    order_id: int
    venue: Venue
    side: Side
    price: float
    size_remaining: float
    is_maker: bool
    queue: Optional[QueuePosition]
    role: str = "quoter"   # "quoter" or "hedger" — keeps the two from cancelling each other


class ExecSimulator:
    def __init__(self, queue_position_fraction: float = 1.0) -> None:
        """
        queue_position_fraction:
            0.0  → assume our order is at the FRONT (queue_ahead = 0).
                   Optimistic; useful for demos so fills happen on live data.
            1.0  → assume our order is at the BACK (queue_ahead = full reported
                   level size). Pessimistic but realistic for joining a busy BBO.
            0.5  → middle of the queue (a reasonable default).
        Also: if our quote is at a NEW price (no size reported at that level),
        queue_ahead is always 0 regardless of this knob.
        """
        self.resting: Dict[int, _Resting] = {}
        self.qpos_frac = max(0.0, min(1.0, queue_position_fraction))

    # --- order lifecycle ---

    def place_maker(self, q, book: OrderBook, ts: int, role: str = "quoter") -> None:
        """Place a maker quote. q is either Quote or HedgeOrder with is_maker=True.

        Enforces the "one resting order per venue+side per role" invariant:
        any pre-existing resting order owned by the same role at the same
        venue+side is cancelled first. This is what prevents partial-fill
        phantom orders from accumulating.
        """
        # cancel any stale order from the same role at the same venue+side
        stale = [oid for oid, r in self.resting.items()
                 if r.venue == q.venue and r.side == q.side and r.role == role
                 and oid != q.order_id]
        for oid in stale:
            self.resting.pop(oid, None)
        # If our price is BETTER than the touch (improving the BBO), we're alone
        # at that price → queue_ahead is exactly 0, regardless of qpos_frac.
        # Otherwise, queue_ahead = qpos_frac * reported_size_at_level.
        bb, ba = book.best_bid(), book.best_ask()
        if q.side == Side.BID:
            if bb and q.price > bb[0]:
                queue_ahead = 0.0          # improving BBO
            else:
                level_size = book.size_at(Side.BID, q.price)
                queue_ahead = self.qpos_frac * level_size
        else:
            if ba and q.price < ba[0]:
                queue_ahead = 0.0          # improving BBO
            else:
                level_size = book.size_at(Side.ASK, q.price)
                queue_ahead = self.qpos_frac * level_size
        qp = QueuePosition(venue=str(q.venue.value), side=str(q.side.value),
                           price=q.price, own_size=q.size, queue_ahead=queue_ahead,
                           placed_ts=ts)
        self.resting[q.order_id] = _Resting(q.order_id, q.venue, q.side, q.price,
                                            q.size, True, qp, role=role)

    def cancel(self, order_id: int) -> None:
        self.resting.pop(order_id, None)

    def fill_taker(self, h: HedgeOrder, book: OrderBook, ts: int) -> Optional[Fill]:
        """Cross with a taker. Fill at best opposite."""
        if h.side == Side.ASK:           # we sell as taker → hit best bid
            top = book.best_bid()
        else:                            # we buy as taker → lift best ask
            top = book.best_ask()
        if top is None:
            return None
        px = top[0]
        fill_size = min(h.size, top[1])
        return Fill(ts=ts, venue=h.venue, side=h.side, price=px,
                    size=fill_size, is_maker=False, order_id=h.order_id)

    # --- driven by trade ticks ---

    def on_trade(self, t: Trade) -> List[Fill]:
        """When a public trade hits a side, walk through any of our resting
        orders that are at-or-better than the trade price and that the queue
        traversal actually reaches.
        """
        fills: List[Fill] = []
        # Trade aggressor BID means a buyer crossed the spread → trade hit the
        # ASK side of the book. Resting ASK orders at price <= trade.price can fill.
        # Trade aggressor ASK means a seller crossed → hit the BID side.
        affected_side = Side.ASK if t.aggressor == Side.BID else Side.BID
        remaining_volume = t.size
        # Find our resting orders on the affected side at venue, sorted by price priority
        candidates = [r for r in self.resting.values()
                      if r.venue == t.venue and r.side == affected_side]
        # For ASK side, lower price has priority; for BID, higher price has priority
        if affected_side == Side.ASK:
            candidates.sort(key=lambda r: r.price)
            # Only orders at price <= trade.price get hit
            candidates = [r for r in candidates if r.price <= t.price]
        else:
            candidates.sort(key=lambda r: -r.price)
            candidates = [r for r in candidates if r.price >= t.price]

        for r in candidates:
            if remaining_volume <= 0:
                break
            if r.queue is None:
                continue
            hit = r.queue.on_trade(remaining_volume)
            # the queue.on_trade method already decremented its own_size; we
            # need to consume from remaining_volume the queue_ahead it ate
            # plus the hit itself.
            consumed_ahead = min(r.queue.queue_ahead + hit, remaining_volume)
            # NOTE: on_trade already deducted ahead inside the method; recompute
            # consumed as min(initial_ahead_used + hit, remaining_volume).
            # Simpler: just clamp remaining_volume by hit + previous ahead it ate.
            remaining_volume = max(0.0, remaining_volume - (consumed_ahead))
            if hit > 0:
                fills.append(Fill(ts=t.ts, venue=r.venue, side=r.side,
                                  price=r.price, size=hit, is_maker=True,
                                  order_id=r.order_id))
                r.size_remaining -= hit
                if r.size_remaining <= 1e-12:
                    self.resting.pop(r.order_id, None)
        return fills

    def on_book_update(self, book: OrderBook) -> None:
        """When a level our order sits at shrinks (cancels), update queue_ahead.

        Only meaningful when our quote price is actually in-book: if our
        price is below the deepest reported level (or above the highest),
        the book feed simply doesn't have info about our level, so we
        leave queue_ahead untouched.
        """
        for r in list(self.resting.values()):
            if r.queue is None:
                continue
            if r.venue.value != book.venue:
                continue
            # Skip if our price is outside the reported book window
            if r.side == Side.BID:
                if not book.bids or r.price < book.bids.keys()[0]:
                    continue   # deeper than the deepest reported level
            else:
                if not book.asks or r.price > book.asks.keys()[-1]:
                    continue
            level_size = book.size_at(r.side, r.price)
            est = r.queue.queue_ahead + r.queue.own_size
            if level_size < est and level_size > 0:
                shrinkage = est - level_size
                r.queue.on_level_reduction(shrinkage)
