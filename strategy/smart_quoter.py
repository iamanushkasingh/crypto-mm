"""Signal-driven maker quoter.

Strategy:
  - Anchor at the TOUCH (best_bid / best_ask) by default. This is what
    actually gets fills on a live perp.
  - Step back N ticks from the touch on the side predicted to be adverse-
    selected. N = floor(α_signal * |signal|).
  - Pause a side entirely when signal strength exceeds a threshold.
  - Inventory: linear skew + quadratic edge penalty (panic widening at
    >80% of max inv).

Signals used (from strategy.signals.SignalEngine):

  OFI_norm > 0  → buyer aggression incoming → expect mid to rise →
                  selling at touch sells us LOW → step back the ASK,
                  keep the bid (won't fill anyway, and if it does it's good).

  basis = quote_mid − hedge_mid:
    basis > 0  → quote venue is rich → arbitrageurs will sell on quote
                 venue, hitting OUR bid before basis collapses → step
                 back the BID.
    basis < 0  → quote venue is cheap → arbitrageurs lift quote-venue
                 asks → step back the ASK.

  realized_vol_bps: when vol is high, widen both sides modestly.

The result is "join BBO when benign, pull when toxic". That's the only
way a maker MM on a competitive venue can be positive-EV after fees.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.events import Side, Venue
from strategy.quoter import Quote
from strategy.signals import SignalEngine


class SmartQuoter:
    def __init__(self, cfg: dict, venue: Venue) -> None:
        self.venue = venue
        self.size = float(cfg["quote_size_contracts"])
        self.tick_size = float(cfg.get("tick_size", 0.1))
        self.base_ticks_from_touch = int(cfg.get("ticks_from_touch", 0))   # 0 = join BBO

        self.max_inv = float(cfg["max_inventory_contracts"])
        self.skew_ticks_per_contract = float(cfg.get("skew_ticks_per_contract", 5.0))

        # Signal-driven step-back weights (ticks per unit signal)
        self.ofi_ticks_per_unit = float(cfg.get("ofi_ticks_per_unit", 12.0))
        self.basis_ticks_per_bp = float(cfg.get("basis_ticks_per_bp", 6.0))
        self.vol_ticks_per_bps = float(cfg.get("vol_ticks_per_bps", 0.5))
        # Mean-reversion: extra step-back on the side that catches the reversion
        self.mr_ticks_per_bp = float(cfg.get("mr_ticks_per_bp", 6.0))

        # Caps and pauses
        self.max_stepback_ticks = int(cfg.get("max_stepback_ticks", 30))
        self.pause_signal_threshold = float(cfg.get("pause_signal_threshold", 1.5))
        self.pause_ms = float(cfg.get("pause_ms_after_toxic", 400.0))
        self.vol_pause_bps = float(cfg.get("vol_pause_bps", 0.0))   # 0 disables
        self.basis_hard_pause_bps = float(cfg.get("basis_hard_pause_bps", 0.0))

        # --- Asymmetric "favourable side" sizing ---
        # When the basis (and optionally MR) makes one side favourable, boost
        # that side's size and tighten its depth. When the other side is
        # un-favourable past `unfavourable_skip_bps`, skip it entirely.
        self.size_boost_per_bp = float(cfg.get("size_boost_per_favourable_bp", 0.0))
        self.max_size_mult = float(cfg.get("max_size_mult", 1.0))
        self.depth_tighten_ticks_per_bp = float(cfg.get("depth_tighten_ticks_per_favourable_bp", 0.0))
        self.unfavourable_skip_bps = float(cfg.get("unfavourable_skip_bps", 1e9))
        self.favourability_use_mr = bool(cfg.get("favourability_use_mr", True))
        self.mr_weight = float(cfg.get("favourability_mr_weight", 0.5))

        self.bid_quote: Optional[Quote] = None
        self.ask_quote: Optional[Quote] = None
        self._next_id = 1
        self._bid_paused_until = 0
        self._ask_paused_until = 0
        # Drawdown circuit breaker: if recent rolling PnL drops by more than
        # `dd_halt_usd`, pause both sides for `dd_halt_ms`.
        self.dd_window_sec = float(cfg.get("dd_window_sec", 120.0))
        self.dd_halt_usd = float(cfg.get("dd_halt_usd", 10.0))
        self.dd_halt_ms = float(cfg.get("dd_halt_ms", 60000.0))
        self._pnl_history: list[tuple[int, float]] = []  # (ts, total_pnl)
        self._dd_paused_until = 0

        # Telemetry counters
        self.n_bid_paused = 0
        self.n_ask_paused = 0
        self.n_bid_stepback = 0
        self.n_ask_stepback = 0

    def _gen_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def desired_quotes(self, ts: int, best_bid: float, best_ask: float,
                       inv: float, signals: SignalEngine,
                       hedge_mid: float, quote_mid: float
                       ) -> tuple[Optional[Quote], Optional[Quote]]:
        if best_bid <= 0 or best_ask <= 0:
            return None, None

        ofi = signals.ofi_norm(self.venue)
        basis = signals.cross_basis_bps_from_mids(quote_mid, hedge_mid)
        vol_bps = signals.realized_vol_bps_per_sec(quote_mid)
        # Mean-reversion deviation in bps. Positive = price above EWMA →
        # expect mean reversion DOWN → ASK is the side that catches it.
        mr_bps = signals.mean_reversion_bps(self.venue)

        # --- per-side stepback in ticks ---
        # ASK side toxic when buyers aggress (OFI > 0), quote venue is CHEAP
        # (basis < 0), or price has rallied above its EWMA (mr_bps > 0).
        ask_stepback = (max(0.0, +ofi) * self.ofi_ticks_per_unit
                        + max(0.0, -basis) * self.basis_ticks_per_bp
                        + max(0.0, +mr_bps) * self.mr_ticks_per_bp
                        + vol_bps * self.vol_ticks_per_bps)
        bid_stepback = (max(0.0, -ofi) * self.ofi_ticks_per_unit
                        + max(0.0, +basis) * self.basis_ticks_per_bp
                        + max(0.0, -mr_bps) * self.mr_ticks_per_bp
                        + vol_bps * self.vol_ticks_per_bps)
        ask_stepback = min(self.max_stepback_ticks, ask_stepback)
        bid_stepback = min(self.max_stepback_ticks, bid_stepback)

        # --- pause if signal too strong --- (emit event only on transition
        # from "not paused" → "paused", not on every tick the pause extends)
        ask_signal_strength = max(+ofi, -basis * 0.1)
        bid_signal_strength = max(-ofi, +basis * 0.1)
        if ask_signal_strength > self.pause_signal_threshold:
            was_paused = ts < self._ask_paused_until
            self._ask_paused_until = max(self._ask_paused_until,
                                          ts + int(self.pause_ms * 1e6))
            if not was_paused:
                self.n_ask_paused += 1
                self._emit("PAUSE_ASK", ts, reason="signal",
                           strength=round(ask_signal_strength, 3),
                           ofi=round(ofi, 3), basis_bps=round(basis, 2))
        if bid_signal_strength > self.pause_signal_threshold:
            was_paused = ts < self._bid_paused_until
            self._bid_paused_until = max(self._bid_paused_until,
                                          ts + int(self.pause_ms * 1e6))
            if not was_paused:
                self.n_bid_paused += 1
                self._emit("PAUSE_BID", ts, reason="signal",
                           strength=round(bid_signal_strength, 3),
                           ofi=round(ofi, 3), basis_bps=round(basis, 2))

        # --- HARD vol cutoff: pause BOTH sides when realized vol blows out ---
        vol_pause_bps = getattr(self, "vol_pause_bps", 0.0)
        if vol_pause_bps > 0 and vol_bps > vol_pause_bps:
            was_paused = ts < self._ask_paused_until and ts < self._bid_paused_until
            new_until = ts + int(self.pause_ms * 1e6)
            self._ask_paused_until = max(self._ask_paused_until, new_until)
            self._bid_paused_until = max(self._bid_paused_until, new_until)
            if not was_paused:
                self._emit("PAUSE_BOTH_VOL", ts, vol_bps=round(vol_bps, 3))

        # --- HARD basis cutoff: when cross-venue basis is large,
        # the disadvantaged side is almost certainly going to be hit by
        # an arbitrageur. Skip it entirely (don't quote).
        basis_hard_bps = getattr(self, "basis_hard_pause_bps", 0.0)
        if basis_hard_bps > 0:
            if basis > basis_hard_bps:
                was_paused = ts < self._bid_paused_until
                self._bid_paused_until = max(self._bid_paused_until,
                                              ts + int(self.pause_ms * 1e6))
                if not was_paused:
                    self._emit("PAUSE_BID_BASIS", ts, basis_bps=round(basis, 2))
            if basis < -basis_hard_bps:
                was_paused = ts < self._ask_paused_until
                self._ask_paused_until = max(self._ask_paused_until,
                                              ts + int(self.pause_ms * 1e6))
                if not was_paused:
                    self._emit("PAUSE_ASK_BASIS", ts, basis_bps=round(basis, 2))

        # --- inventory: skew in ticks ---
        # Long → push both quotes DOWN (offer cheaper, bid less).
        skew_ticks = int(round(self.skew_ticks_per_contract * inv))

        # Apply inventory cap (don't quote the side that worsens position)
        # PLUS drawdown circuit breaker (halts both sides)
        in_dd = ts < self._dd_paused_until
        suppress_bid = inv >= self.max_inv or ts < self._bid_paused_until or in_dd
        suppress_ask = inv <= -self.max_inv or ts < self._ask_paused_until or in_dd

        # --- Per-side favourability (positive = favourable for that side) ---
        # ASK favourable when Binance is RICH (basis>0): we sell into the
        # premium, hedge by buying cheaper Bybit.
        # BID favourable when Binance is CHEAP (basis<0): we buy at the
        # discount, hedge by selling richer Bybit.
        ask_fav_bps = +basis + (self.mr_weight * mr_bps if self.favourability_use_mr else 0.0)
        bid_fav_bps = -basis - (self.mr_weight * mr_bps if self.favourability_use_mr else 0.0)

        # Skip side if favourability is BELOW a threshold (i.e. quite unfavourable)
        if ask_fav_bps < -self.unfavourable_skip_bps:
            suppress_ask = True
        if bid_fav_bps < -self.unfavourable_skip_bps:
            suppress_bid = True

        # Size scaling: boost when favourable, shrink (toward 0) when unfavourable.
        # Clamped to [0, max_size_mult].
        def _size_mult(fav_bps: float) -> float:
            m = 1.0 + self.size_boost_per_bp * fav_bps
            return max(0.0, min(self.max_size_mult, m))

        ask_size_mult = _size_mult(ask_fav_bps)
        bid_size_mult = _size_mult(bid_fav_bps)

        # Depth-tightening: when favourable, reduce stepback (move closer to BBO).
        ask_tighten = max(0.0, self.depth_tighten_ticks_per_bp * ask_fav_bps)
        bid_tighten = max(0.0, self.depth_tighten_ticks_per_bp * bid_fav_bps)

        if suppress_bid or bid_size_mult <= 1e-6:
            bid_q = None
        else:
            ticks_from_touch_bid = (self.base_ticks_from_touch
                                    - int(bid_stepback)
                                    - skew_ticks
                                    + int(bid_tighten))
            ticks_from_touch_bid = min(ticks_from_touch_bid, 0)
            bid_price = best_bid + ticks_from_touch_bid * self.tick_size
            bid_q = Quote(self._gen_id(), self.venue, Side.BID, bid_price,
                          self.size * bid_size_mult, ts)
            if bid_stepback > 0.5:
                self.n_bid_stepback += 1

        if suppress_ask or ask_size_mult <= 1e-6:
            ask_q = None
        else:
            ticks_from_touch_ask = (self.base_ticks_from_touch
                                    - int(ask_stepback)
                                    + skew_ticks
                                    + int(ask_tighten))
            ticks_from_touch_ask = min(ticks_from_touch_ask, 0)
            ask_price = best_ask - ticks_from_touch_ask * self.tick_size
            ask_q = Quote(self._gen_id(), self.venue, Side.ASK, ask_price,
                          self.size * ask_size_mult, ts)
            if ask_stepback > 0.5:
                self.n_ask_stepback += 1

        return bid_q, ask_q

    def step(self, ts: int, mid: float, inv: float,
             microprice: float, signals: SignalEngine, hb_book
             ) -> list[tuple[str, Quote]]:
        # The smart quoter needs the touch and the hedge mid.
        from core.orderbook import OrderBook
        # mid here is the quote-venue mid; we also need touch:
        qb_bb = signals_quote_bb = None
        # We can't fetch the quote book directly; instead the engine will
        # pass the touch via a dedicated method. To keep the existing
        # engine call signature, fall back to deriving touch from mid.
        # In practice the engine sets self.last_touch via set_touch().
        bb = getattr(self, "_last_bb", 0.0)
        ba = getattr(self, "_last_ba", 0.0)
        hm = hb_book.microprice() or hb_book.mid() or 0.0
        target_bid, target_ask = self.desired_quotes(
            ts, bb, ba, inv, signals, hm, microprice)

        actions: list[tuple[str, Quote]] = []
        # Bid side
        if target_bid is None:
            if self.bid_quote is not None:
                actions.append(("cancel", self.bid_quote))
                self.bid_quote = None
        else:
            if self.bid_quote is None:
                actions.append(("place", target_bid))
                self.bid_quote = target_bid
            elif abs(target_bid.price - self.bid_quote.price) > self.tick_size * 0.5:
                actions.append(("replace", target_bid))
                self.bid_quote = target_bid

        # Ask side — symmetric
        if target_ask is None:
            if self.ask_quote is not None:
                actions.append(("cancel", self.ask_quote))
                self.ask_quote = None
        else:
            if self.ask_quote is None:
                actions.append(("place", target_ask))
                self.ask_quote = target_ask
            elif abs(target_ask.price - self.ask_quote.price) > self.tick_size * 0.5:
                actions.append(("replace", target_ask))
                self.ask_quote = target_ask
        return actions

    def set_touch(self, best_bid: float, best_ask: float) -> None:
        self._last_bb = best_bid
        self._last_ba = best_ask

    def attach_logger(self, run_logger) -> None:
        """Engine calls this once at startup so we can emit events."""
        self._run_logger = run_logger

    def _emit(self, event_type: str, ts: int, **kw) -> None:
        rl = getattr(self, "_run_logger", None)
        if rl is None:
            return
        rl.log_event(event_type, ts, **kw)

    def observe_pnl(self, ts: int, total_pnl: float) -> None:
        """Engine calls this every step — used by the drawdown halt."""
        self._pnl_history.append((ts, total_pnl))
        # Evict samples older than dd_window_sec
        cutoff = ts - int(self.dd_window_sec * 1e9)
        while self._pnl_history and self._pnl_history[0][0] < cutoff:
            self._pnl_history.pop(0)
        if len(self._pnl_history) >= 2:
            window_pnl = self._pnl_history[-1][1] - self._pnl_history[0][1]
            if window_pnl < -self.dd_halt_usd:
                self._dd_paused_until = max(self._dd_paused_until,
                                             ts + int(self.dd_halt_ms * 1e6))

    def on_fill(self, side: Side) -> None:
        if side == Side.BID:
            self.bid_quote = None
        else:
            self.ask_quote = None
