"""Hedger for the hedge venue.

After a fill on the quote venue creates inventory `q`, the hedger tries to
flatten on the hedge venue. It uses a two-stage policy:

  Stage 1 (maker): post a hedge order just through the touch (effectively
                   a marketable-limit, but tagged as maker if it rests).
                   This may earn a maker rebate if the book is thick enough.
  Stage 2 (taker): after `hedge_timeout_ms`, cancel the maker hedge and
                   cross with a taker order. We pay `taker_bps` but the
                   delta is gone.

The hedger always sizes the hedge exactly to the residual delta on that
side (modulo `hedge_size_multiplier` if you want over- or under-hedge).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.events import Side, Venue


@dataclass
class HedgeOrder:
    order_id: int
    venue: Venue
    side: Side           # which side WE want to trade (opposite of inventory sign)
    size: float
    price: float
    is_maker: bool
    placed_ts: int


class Hedger:
    def __init__(self, cfg: dict, venue: Venue) -> None:
        self.venue = venue
        self.depth_bps = float(cfg["hedge_initial_depth_bps"])
        self.timeout_ms = float(cfg["hedge_timeout_ms"])
        self.size_mult = float(cfg["hedge_size_multiplier"])
        # When True, post the maker hedge AT the touch (best_bid for sells,
        # best_ask for buys). This is the "passive but joinable" stance — we
        # earn the maker rebate when filled instead of paying through the
        # spread. depth_bps above is then ignored.
        self.maker_at_touch = bool(cfg.get("hedge_maker_at_touch", True))
        # Dynamic timeout: if inventory is small (well below max_inv), wait
        # MUCH longer for a maker fill. Only fire taker when inventory is
        # large or urgent.
        self.timeout_low_inv_ms = float(cfg.get("hedge_timeout_low_inv_ms",
                                                 self.timeout_ms * 5))
        self.urgent_inv_fraction = float(cfg.get("hedge_urgent_inv_fraction", 0.5))
        self.max_inv = float(cfg.get("max_inventory_contracts", 0.05))
        self.tick_size = float(cfg.get("tick_size", 0.1))
        # Internal-hedging threshold: don't hedge externally when |delta|
        # is below this fraction of max_inv — let the opposite-side maker
        # quote flatten naturally (much cheaper than firing a Bybit hedge).
        self.no_hedge_below_fraction = float(cfg.get("no_hedge_below_fraction", 0.0))
        self.active: Optional[HedgeOrder] = None
        self._next_id = 100000

    def _gen_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def step(self, ts: int, net_delta: float, best_bid: float, best_ask: float
             ) -> list[tuple[str, HedgeOrder]]:
        """Decide hedge action.

        net_delta > 0 → we're long → need to SELL → hedge.side = ASK
        net_delta < 0 → we're short → need to BUY → hedge.side = BID
        """
        actions: list[tuple[str, HedgeOrder]] = []
        target_size = abs(net_delta) * self.size_mult

        # Internal hedging: skip external hedge while inventory is small.
        hedge_floor = self.no_hedge_below_fraction * self.max_inv
        if abs(net_delta) < hedge_floor:
            target_size = 0.0

        if target_size < 1e-9:
            if self.active is not None:
                actions.append(("cancel", self.active))
                self.active = None
            return actions

        target_side = Side.ASK if net_delta > 0 else Side.BID

        if self.active is None:
            # Maker hedge price:
            #   maker_at_touch=True  → join the BBO on the side we want to trade
            #   maker_at_touch=False → original behaviour (a bit through the touch)
            if self.maker_at_touch:
                price = best_ask if target_side == Side.ASK else best_bid
            else:
                if target_side == Side.ASK:
                    price = best_bid * (1 + self.depth_bps * 1e-4)
                else:
                    price = best_ask * (1 - self.depth_bps * 1e-4)
            h = HedgeOrder(self._gen_id(), self.venue, target_side, target_size,
                           price, is_maker=True, placed_ts=ts)
            actions.append(("place", h))
            self.active = h
            return actions

        # We already have a hedge in flight.
        age_ms = (ts - self.active.placed_ts) * 1e-6
        # If side flipped, replace AND reset the timer (the position direction
        # actually changed). If just the size grew (more inventory accumulated
        # while we waited), keep the original placed_ts so the timeout still
        # fires — otherwise rapid-fire fills can hold the hedger in maker
        # mode forever and inventory blows out.
        def _maker_price(side):
            if self.maker_at_touch:
                return best_ask if side == Side.ASK else best_bid
            if side == Side.ASK:
                return best_bid * (1 + self.depth_bps * 1e-4)
            return best_ask * (1 - self.depth_bps * 1e-4)

        if self.active.side != target_side:
            actions.append(("cancel", self.active))
            h = HedgeOrder(self._gen_id(), self.venue, target_side, target_size,
                           _maker_price(target_side), is_maker=True, placed_ts=ts)
            actions.append(("place", h))
            self.active = h
            return actions
        elif abs(self.active.size - target_size) > 1e-9:
            actions.append(("cancel", self.active))
            preserved_ts = self.active.placed_ts
            h = HedgeOrder(self._gen_id(), self.venue, target_side, target_size,
                           _maker_price(target_side), is_maker=True,
                           placed_ts=preserved_ts)
            actions.append(("place", h))
            self.active = h
            # fall through to the timeout check below

        # Adaptive timeout: more inventory → fire taker sooner.
        inv_fraction = abs(net_delta) / max(self.max_inv, 1e-9)
        if inv_fraction >= self.urgent_inv_fraction:
            effective_timeout = self.timeout_ms
        else:
            # interpolate between long timeout (low inv) and short (urgent inv)
            t = inv_fraction / self.urgent_inv_fraction
            effective_timeout = (1 - t) * self.timeout_low_inv_ms + t * self.timeout_ms

        # Timed out → escalate to taker (cross the spread)
        if self.active.is_maker and age_ms > effective_timeout:
            actions.append(("cancel", self.active))
            taker_price = best_ask if target_side == Side.ASK else best_bid
            # When SELLING as taker, we hit the bid; when BUYING as taker, we lift the ask.
            taker_price = best_bid if target_side == Side.ASK else best_ask
            h = HedgeOrder(self._gen_id(), self.venue, target_side, target_size,
                           taker_price, is_maker=False, placed_ts=ts)
            actions.append(("taker", h))
            self.active = h
            return actions

        return actions

    def on_fill(self, _side: Side) -> None:
        self.active = None
