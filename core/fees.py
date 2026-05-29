"""Fee / rebate schedule.

Fees are stored in bps and applied to notional (price * size). Positive
means we pay; negative means we receive a rebate.
"""
from __future__ import annotations

from dataclasses import dataclass
from .events import Venue


@dataclass(frozen=True)
class FeeSchedule:
    maker_bps: float
    taker_bps: float

    def maker_cost(self, notional: float) -> float:
        return notional * self.maker_bps * 1e-4

    def taker_cost(self, notional: float) -> float:
        return notional * self.taker_bps * 1e-4


def schedule_from_config(cfg: dict) -> dict[Venue, FeeSchedule]:
    out = {}
    for v_str, v in (("binance", Venue.BINANCE), ("bybit", Venue.BYBIT)):
        block = cfg["fees"][v_str]
        out[v] = FeeSchedule(maker_bps=float(block["maker_bps"]),
                             taker_bps=float(block["taker_bps"]))
    return out
