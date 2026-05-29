"""Microstructure signals consumed by the smart quoter.

Three features, all O(1)-amortized per tick:

  microprice(venue): size-weighted touch price. Better predictor of the
      next trade price than the mid because it leans toward the thicker side.

  ofi(venue, window_ms): Order-Flow Imbalance — signed trade volume in the
      recent window. Positive ⇒ buyers have been aggressing (next mid move
      likely UP). Negative ⇒ sellers aggressing (mid likely DOWN).

  cross_basis_bps(): (quote_venue_mid − hedge_venue_mid) in bps.
      If the quote venue is RICHER than the hedge venue, arbitrageurs will
      sell on the quote venue (hitting OUR bid). So a positive basis is a
      warning to widen the bid. Conversely a negative basis warns the ask.

Also tracks a rolling realized-vol estimator so the quoter can widen
spreads when the market is moving.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

from core.events import Trade, Venue, Side
from core.orderbook import OrderBook


@dataclass
class SignalEngine:
    ofi_window_ms: float = 500.0          # rolling OFI window
    vol_halflife_sec: float = 30.0        # realized-vol EWMA halflife
    ofi_normalizer: float = 1.0           # divide raw OFI to roughly +/-1
    # Mean-reversion: rolling EWMA of mid; current_mid - ewma_mid expressed
    # in bps gives the "deviation" signal. Positive = price ABOVE its
    # recent average → expect mean-reversion DOWN.
    mean_revert_halflife_sec: float = 8.0
    # state
    _trades: Dict[Venue, Deque[Trade]] = field(default_factory=lambda: {
        Venue.BINANCE: deque(), Venue.BYBIT: deque()})
    _last_mid: Dict[Venue, float] = field(default_factory=lambda: {
        Venue.BINANCE: 0.0, Venue.BYBIT: 0.0})
    _ewma_mid: Dict[Venue, float] = field(default_factory=lambda: {
        Venue.BINANCE: 0.0, Venue.BYBIT: 0.0})
    _last_ts: int = 0
    _vol_var: float = 0.0                 # EWMA variance of mid returns

    # ---- ingest ----

    def on_trade(self, t: Trade) -> None:
        dq = self._trades[t.venue]
        dq.append(t)
        cutoff = t.ts - int(self.ofi_window_ms * 1e6)
        while dq and dq[0].ts < cutoff:
            dq.popleft()

    def on_book(self, venue: Venue, mid: float, ts: int) -> None:
        if mid <= 0:
            return
        prev = self._last_mid[venue]
        if prev > 0 and ts > self._last_ts:
            ret = math.log(mid / prev)
            dt = max((ts - self._last_ts) * 1e-9, 1e-6)
            # convert to per-second return^2, then EWMA
            inst = (ret * ret) / dt
            alpha_v = 1 - 0.5 ** (dt / self.vol_halflife_sec)
            self._vol_var = (1 - alpha_v) * self._vol_var + alpha_v * inst
            # EWMA mid for mean-reversion
            alpha_m = 1 - 0.5 ** (dt / self.mean_revert_halflife_sec)
            prev_ewma = self._ewma_mid[venue]
            if prev_ewma <= 0:
                self._ewma_mid[venue] = mid
            else:
                self._ewma_mid[venue] = (1 - alpha_m) * prev_ewma + alpha_m * mid
        elif prev <= 0:
            self._ewma_mid[venue] = mid
        self._last_mid[venue] = mid
        self._last_ts = ts

    # ---- features ----

    def ofi(self, venue: Venue) -> float:
        total = 0.0
        for t in self._trades[venue]:
            sign = +1 if t.aggressor == Side.BID else -1
            total += sign * t.size
        return total

    def ofi_norm(self, venue: Venue) -> float:
        """Normalised OFI in roughly [-1, +1]."""
        return max(-3.0, min(3.0, self.ofi(venue) / max(self.ofi_normalizer, 1e-9)))

    def cross_basis_bps(self, qb: OrderBook, hb: OrderBook) -> float:
        qm = qb.microprice() or qb.mid()
        hm = hb.microprice() or hb.mid()
        if not qm or not hm:
            return 0.0
        return (qm - hm) / qm * 1e4

    def cross_basis_bps_from_mids(self, qm: float, hm: float) -> float:
        if not qm or not hm:
            return 0.0
        return (qm - hm) / qm * 1e4

    def realized_vol_bps_per_sec(self, mid: float) -> float:
        """sqrt of EWMA variance, in bps per √second."""
        if mid <= 0:
            return 0.0
        return math.sqrt(max(self._vol_var, 0.0)) * 1e4

    def mean_reversion_bps(self, venue: Venue) -> float:
        """(current_mid - ewma_mid) / mid in bps.

        Positive ⇒ price is above its short-term average ⇒ expect a pullback
        DOWN ⇒ the ASK is the side likely to "catch" the reversion (a buyer
        crosses up and gets us short into the pullback). Step back ASK.

        Negative ⇒ price is below the EWMA ⇒ expect bounce UP ⇒ step back
        BID (sellers crossing down would fill us right before the bounce).
        """
        mid = self._last_mid.get(venue, 0.0)
        ewma = self._ewma_mid.get(venue, 0.0)
        if mid <= 0 or ewma <= 0:
            return 0.0
        return (mid - ewma) / mid * 1e4
