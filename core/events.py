"""Tick-event types.

All events carry an event-time `ts` in nanoseconds since epoch. Both
historical replay and live WS clients normalize to this same shape so the
engine doesn't care where the event came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"


class Venue(str, Enum):
    BINANCE = "binance"
    BYBIT = "bybit"


@dataclass
class BookUpdate:
    """Incremental L2 update: list of (price, size) — size=0 means delete."""
    ts: int
    venue: Venue
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    is_snapshot: bool = False


@dataclass
class Trade:
    """Public trade print on a venue."""
    ts: int
    venue: Venue
    price: float
    size: float
    aggressor: Side  # which side hit (BID == buyer was the aggressor)


# Strategy-internal events (not from the wire)

@dataclass
class Fill:
    """A simulated fill of one of our orders."""
    ts: int
    venue: Venue
    side: Side          # which side of our book filled
    price: float
    size: float
    is_maker: bool
    order_id: int


@dataclass
class Cancel:
    ts: int
    venue: Venue
    order_id: int
    reason: str
