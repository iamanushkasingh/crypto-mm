"""Queue-position-aware fill probability and EV-of-waiting model.

Each resting order tracks the volume of contracts in front of it in its
price level. Public trades on that side deplete the queue (FIFO matching).
Cancels at the level also deplete the queue, *but* we can't tell from L2
deltas whether a size reduction came from a cancel ahead of us or behind
us — we conservatively assume half is ahead.

Fill probability over horizon Δt at queue position q with arrival rate λ:
    p(Δt) ≈ 1 - exp( -λ * Δt / (q + s) )

where s is our own order size. λ is estimated from a rolling exponential
average of recent trade volume on that side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Optional


@dataclass
class LambdaTracker:
    """Exponential decay of trade volume per second on each side of a venue."""
    half_life_sec: float = 5.0
    _last_ts: int = 0
    _value: float = 0.0   # contracts / second

    def update(self, ts: int, traded_volume: float) -> None:
        if self._last_ts == 0:
            self._last_ts = ts
        dt = max(0.0, (ts - self._last_ts) * 1e-9)
        decay = 0.5 ** (dt / self.half_life_sec)
        # rough conversion: traded_volume is in this dt, so add it as a rate
        rate = traded_volume / max(dt, 1e-3)
        self._value = decay * self._value + (1 - decay) * rate
        self._last_ts = ts

    @property
    def lam(self) -> float:
        return max(self._value, 1e-6)


@dataclass
class QueuePosition:
    """Tracks our position in the queue at a single price level."""
    venue: str
    side: str
    price: float
    own_size: float
    queue_ahead: float      # contracts ahead of us
    placed_ts: int

    def on_trade(self, traded_volume: float) -> float:
        """A trade on our side depletes the queue. Returns volume hitting us."""
        if traded_volume <= 0:
            return 0.0
        eat_queue = min(self.queue_ahead, traded_volume)
        self.queue_ahead -= eat_queue
        remainder = traded_volume - eat_queue
        if remainder <= 0:
            return 0.0
        # remainder hits us — clamp to our size
        hit = min(self.own_size, remainder)
        self.own_size -= hit
        return hit

    def on_level_reduction(self, delta: float) -> None:
        """A cancel at our level: conservatively assume half is ahead of us."""
        if delta <= 0:
            return
        ahead = 0.5 * delta
        self.queue_ahead = max(0.0, self.queue_ahead - ahead)

    def fill_prob(self, dt_sec: float, lam: float) -> float:
        eff_q = self.queue_ahead + self.own_size
        if eff_q <= 0:
            return 1.0
        return 1.0 - math.exp(- lam * dt_sec / eff_q)


def expected_value_of_waiting(
    edge_now_bps: float,
    fill_prob_in_window: float,
    inventory_risk_bps_per_sec: float,
    window_sec: float,
) -> float:
    """EV of leaving a maker order in book for another `window_sec`.

    edge_now_bps      = (quote_price - hedge_price) in bps (positive = profitable)
    fill_prob_in_window
    inventory_risk_bps_per_sec = σ * sqrt(...) * |q| approximation

    Returns EV in bps. If <= 0, the model says cancel / convert to taker.
    """
    expected_gain = fill_prob_in_window * edge_now_bps
    expected_inventory_cost = (1 - fill_prob_in_window) * inventory_risk_bps_per_sec * window_sec
    return expected_gain - expected_inventory_cost
