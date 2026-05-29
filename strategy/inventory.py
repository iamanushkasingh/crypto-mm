"""Inventory and PnL accounting.

Every fill flows through `apply_fill`. Cash, position, and the per-source
PnL ledger are all updated atomically. `mtm` is recomputed on every tick
from the current mid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from core.events import Fill, Side, Venue
from core.fees import FeeSchedule


@dataclass
class Inventory:
    fee_schedules: Dict[Venue, FeeSchedule]
    # signed position per venue (contracts)
    pos: Dict[Venue, float] = field(default_factory=lambda: {Venue.BINANCE: 0.0,
                                                              Venue.BYBIT: 0.0})
    cash_usd: float = 0.0
    # Realized cash from buys/sells (excluding fees) — together with the
    # inventory_mtm of any open position this is the "gross spread capture."
    pnl: Dict[str, float] = field(default_factory=lambda: {
        "trading_cash": 0.0,    # signed cash flow from fills (ex-fees)
        "maker_rebate": 0.0,    # signed; positive if exchange pays us
        "taker_fee": 0.0,       # signed; always non-positive
        "inventory_mtm": 0.0,   # pos * current_mid (open-position MtM)
        "funding": 0.0,
    })
    # Per-venue split — helps separate "intra-venue spread capture" from
    # "cross-venue basis capture".
    trading_cash_by_venue: Dict[Venue, float] = field(default_factory=lambda: {
        Venue.BINANCE: 0.0, Venue.BYBIT: 0.0})
    # Last seen mid per venue (so we can MtM each venue separately)
    last_mid_by_venue: Dict[Venue, float] = field(default_factory=lambda: {
        Venue.BINANCE: 0.0, Venue.BYBIT: 0.0})
    # Cross-venue basis tracker: sum of (favorable basis at fill time) * fill_notional
    # When positive ⇒ on average we filled when basis was in our favour ⇒ profit from
    # the cross-exchange spread itself.
    basis_profit_bps_weighted_notional: float = 0.0
    last_mid: float = 0.0
    n_maker_fills: int = 0
    n_taker_fills: int = 0
    notional_traded: float = 0.0

    @property
    def net_delta(self) -> float:
        return self.pos[Venue.BINANCE] + self.pos[Venue.BYBIT]

    def apply_fill(self, f: Fill) -> None:
        sign = +1 if f.side == Side.BID else -1   # bid fill = we bought
        notional = f.price * f.size
        self.notional_traded += notional
        # update position
        self.pos[f.venue] += sign * f.size
        # trading cash: pay for buys (negative), receive for sells (positive)
        self.pnl["trading_cash"] -= sign * notional
        self.trading_cash_by_venue[f.venue] -= sign * notional
        self.cash_usd -= sign * notional
        # fees
        sched = self.fee_schedules[f.venue]
        if f.is_maker:
            fee = sched.maker_cost(notional)
            self.pnl["maker_rebate"] -= fee   # if maker_bps < 0 fee is negative → rebate is positive
            self.cash_usd -= fee
            self.n_maker_fills += 1
        else:
            fee = sched.taker_cost(notional)
            self.pnl["taker_fee"] -= fee
            self.cash_usd -= fee
            self.n_taker_fills += 1
        self._recompute_spread_capture()

    def mark(self, mid: float, venue: Venue = None) -> None:
        self.last_mid = mid
        if venue is not None:
            self.last_mid_by_venue[venue] = mid
        self._recompute_spread_capture()

    def record_basis_at_fill(self, basis_bps: float, fill_side: Side,
                             notional: float, venue: Venue) -> None:
        """Track whether each fill happened when the cross-exchange basis was
        in our favour.

        Convention:
          - On quote venue (Binance), basis = binance_mid - bybit_mid.
          - We BUY on quote venue when basis < 0 (Binance cheap) → favourable.
          - We SELL on quote venue when basis > 0 (Binance rich) → favourable.
          - Sign convention: favourable = positive contribution.
        """
        from core.events import Venue as _V
        if venue != _V.BINANCE:
            return   # only measure for the quote venue
        sign = +1 if fill_side == Side.ASK else -1   # ASK fill = we sold
        self.basis_profit_bps_weighted_notional += sign * basis_bps * notional

    def _recompute_spread_capture(self) -> None:
        # Per-venue MtM (use each venue's own last mid)
        b_mtm = self.pos[Venue.BINANCE] * self.last_mid_by_venue[Venue.BINANCE]
        y_mtm = self.pos[Venue.BYBIT]   * self.last_mid_by_venue[Venue.BYBIT]
        self.pnl["inventory_mtm"] = b_mtm + y_mtm

    def total_pnl(self) -> float:
        return sum(self.pnl.values())

    def per_venue_pnl(self) -> dict:
        """Decompose trading_cash + inventory_mtm by venue.

        venue_pnl = trading_cash_on_venue + position_on_venue * mid_on_venue
        Sum of both venues = total spread-capture component of PnL.
        """
        out = {}
        for v in (Venue.BINANCE, Venue.BYBIT):
            tc = self.trading_cash_by_venue[v]
            mtm = self.pos[v] * self.last_mid_by_venue[v]
            out[v] = tc + mtm
        return out

    def summary(self) -> dict:
        pv = self.per_venue_pnl()
        # Average basis (in bps) we were filling at, weighted by notional.
        nb = self.notional_traded
        avg_basis = (self.basis_profit_bps_weighted_notional / nb) if nb > 0 else 0.0
        return {
            "net_delta": self.net_delta,
            "pos_binance": self.pos[Venue.BINANCE],
            "pos_bybit": self.pos[Venue.BYBIT],
            "cash_usd": round(self.cash_usd, 4),
            "pnl_total": round(self.total_pnl(), 4),
            **{f"pnl_{k}": round(v, 4) for k, v in self.pnl.items()},
            "pnl_venue_binance": round(pv[Venue.BINANCE], 4),
            "pnl_venue_bybit":   round(pv[Venue.BYBIT], 4),
            "avg_fill_basis_bps_favourable": round(avg_basis, 4),
            "cross_exch_basis_dollar_proxy": round(
                self.basis_profit_bps_weighted_notional * 1e-4, 4),
            "n_maker_fills": self.n_maker_fills,
            "n_taker_fills": self.n_taker_fills,
            "notional_traded": round(self.notional_traded, 2),
        }
