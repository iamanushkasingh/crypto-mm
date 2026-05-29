"""Maker quoter for the quote venue.

Posts a bid and an ask deep in the book. As time passes without a fill, it
walks the quote closer to the mid (linearly in `requote_alpha_bps_per_sec`).
The quote is skewed by inventory so the side that would flatten our delta
is tighter.

This is the *quote* leg — it never crosses the spread as a taker. If the
inventory builds beyond `max_inventory_contracts`, that's handled by the
hedger, not by crossing on the quote venue.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.events import Side, Venue
from strategy.fill_model import QueuePosition


@dataclass
class Quote:
    order_id: int
    venue: Venue
    side: Side
    price: float
    size: float
    placed_ts: int
    queue: Optional[QueuePosition] = None


class Quoter:
    def __init__(self, cfg: dict, venue: Venue) -> None:
        self.venue = venue
        self.size = float(cfg["quote_size_contracts"])
        self.d0_bps = float(cfg["base_depth_bps"])
        self.d_min_bps = float(cfg["min_depth_bps"])
        self.skew = float(cfg["inventory_skew_bps_per_contract"])
        self.alpha = float(cfg["requote_alpha_bps_per_sec"])
        self.max_t = float(cfg["quote_max_time_in_book_sec"])
        self.max_inv = float(cfg["max_inventory_contracts"])
        # Pricing anchor: "mid" = mid*(1±depth_bps) ; "touch" = best_bid/ask ± N ticks.
        # Touch mode is what real MMs do — it tracks the visible BBO regardless of
        # how wide/tight the spread is.
        self.anchor = str(cfg.get("quote_anchor", "mid")).lower()
        self.ticks_from_touch = int(cfg.get("ticks_from_touch", 0))   # +1 = improve, 0 = join, -N = N behind
        self.tick_size = float(cfg.get("tick_size", 0.1))
        self.bid_quote: Optional[Quote] = None
        self.ask_quote: Optional[Quote] = None
        self._next_id = 1

    def _gen_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _skew_ticks(self, inv: float, mid: float) -> int:
        """Inventory skew in ticks. Positive when long (push quotes down)."""
        skew_bps = self.skew * inv
        return int(round(skew_bps * mid * 1e-4 / self.tick_size))

    def _price_bid(self, mid: float, best_bid: float, depth_bps: float,
                   inv: float = 0.0) -> float:
        # touch mode: anchor to best_bid; ticks_from_touch=0 joins BBO, +N improves,
        # -N posts N ticks behind. depth_bps is ignored in pure touch mode — the
        # whole point is to track the visible BBO regardless of its width.
        # Inventory skew is applied as a tick offset (long → bid deeper).
        if self.anchor == "touch" and best_bid > 0:
            skew_ticks = self._skew_ticks(inv, mid)
            return best_bid + (self.ticks_from_touch - skew_ticks) * self.tick_size
        return mid * (1 - depth_bps * 1e-4)

    def _price_ask(self, mid: float, best_ask: float, depth_bps: float,
                   inv: float = 0.0) -> float:
        if self.anchor == "touch" and best_ask > 0:
            skew_ticks = self._skew_ticks(inv, mid)
            return best_ask - (self.ticks_from_touch + skew_ticks) * self.tick_size
        return mid * (1 + depth_bps * 1e-4)

    def desired_quotes(self, ts: int, mid: float, inv: float,
                       best_bid: float = 0.0, best_ask: float = 0.0
                       ) -> tuple[Quote, Quote]:
        """Compute target bid/ask given current mid, inventory, and touch."""
        depth = self.d0_bps
        depth_bid = max(self.d_min_bps, depth + self.skew * inv)
        depth_ask = max(self.d_min_bps, depth - self.skew * inv)
        if inv >= self.max_inv:
            depth_bid = float("inf")
        if inv <= -self.max_inv:
            depth_ask = float("inf")
        bid_price = (self._price_bid(mid, best_bid, depth_bid, inv)
                     if depth_bid != float("inf") else 0.0)
        ask_price = (self._price_ask(mid, best_ask, depth_ask, inv)
                     if depth_ask != float("inf") else 0.0)
        bid = Quote(self._gen_id(), self.venue, Side.BID, bid_price, self.size, ts)
        ask = Quote(self._gen_id(), self.venue, Side.ASK, ask_price, self.size, ts)
        return bid, ask

    def step(self, ts: int, mid: float, inv: float,
             best_bid: float = 0.0, best_ask: float = 0.0
             ) -> list[tuple[str, Quote]]:
        """Return list of (action, quote) where action in {'place', 'replace', 'cancel'}."""
        actions = []
        target_bid, target_ask = self.desired_quotes(ts, mid, inv, best_bid, best_ask)

        # Bid side
        if target_bid.price == 0.0:
            if self.bid_quote:
                actions.append(("cancel", self.bid_quote))
                self.bid_quote = None
        else:
            if self.bid_quote is None:
                actions.append(("place", target_bid))
                self.bid_quote = target_bid
            else:
                # Requote when the target price has drifted "enough" from our resting price.
                # Threshold: 1 tick in touch mode, 0.2 bps in mid mode.
                price_diff = abs(target_bid.price - self.bid_quote.price)
                threshold = (self.tick_size if self.anchor == "touch"
                             else mid * 0.2 * 1e-4)
                if price_diff > threshold * 0.5:
                    actions.append(("replace", target_bid))
                    self.bid_quote = target_bid

        # Ask side — symmetric
        if target_ask.price == 0.0:
            if self.ask_quote:
                actions.append(("cancel", self.ask_quote))
                self.ask_quote = None
        else:
            if self.ask_quote is None:
                actions.append(("place", target_ask))
                self.ask_quote = target_ask
            else:
                price_diff = abs(target_ask.price - self.ask_quote.price)
                threshold = (self.tick_size if self.anchor == "touch"
                             else mid * 0.2 * 1e-4)
                if price_diff > threshold * 0.5:
                    actions.append(("replace", target_ask))
                    self.ask_quote = target_ask

        return actions

    def on_fill(self, side: Side) -> None:
        if side == Side.BID:
            self.bid_quote = None
        else:
            self.ask_quote = None
