"""Strategy engine — the per-tick main loop.

Subscribes to BookUpdate and Trade events from both venues. For each event:
  1. update the relevant venue order book
  2. recompute consolidated mid
  3. let the simulator process any fills the trade caused
  4. mark inventory MtM
  5. let the quoter and hedger emit new actions
  6. apply those actions to the simulator
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from core.events import BookUpdate, Trade, Venue, Side, Fill
from core.orderbook import OrderBook
from core.fees import schedule_from_config
from exec.simulator import ExecSimulator
from strategy.quoter import Quoter
from strategy.smart_quoter import SmartQuoter
from strategy.signals import SignalEngine
from strategy.hedger import Hedger
from strategy.inventory import Inventory
from strategy.logger import RunLogger


class Engine:
    def close(self) -> None:
        """Flush + close the structured logger. Safe to call multiple times."""
        try:
            self.run_logger.close()
        except Exception:
            pass

    def __init__(self, cfg: dict, log_fn=None) -> None:
        self.cfg = cfg
        self.quote_venue = Venue(cfg["quote_venue"])
        self.hedge_venue = Venue(cfg["hedge_venue"])

        self.books = {
            Venue.BINANCE: OrderBook("binance"),
            Venue.BYBIT: OrderBook("bybit"),
        }
        self.sim = ExecSimulator(
            queue_position_fraction=float(cfg.get("queue_position_fraction", 1.0)))
        self.quoter_mode = str(cfg.get("quoter_mode", "dumb")).lower()
        if self.quoter_mode == "smart":
            self.quoter = SmartQuoter(cfg, self.quote_venue)
            self.signals = SignalEngine(
                ofi_window_ms=float(cfg.get("ofi_window_ms", 500.0)),
                ofi_normalizer=float(cfg.get("ofi_normalizer", 1.0)),
                vol_halflife_sec=float(cfg.get("vol_halflife_sec", 30.0)),
                mean_revert_halflife_sec=float(cfg.get("mean_revert_halflife_sec", 8.0)),
            )
        else:
            self.quoter = Quoter(cfg, self.quote_venue)
            self.signals = None
        self.hedger = Hedger(cfg, self.hedge_venue)
        self.inv = Inventory(schedule_from_config(cfg))
        self.log_fn = log_fn or (lambda *_a, **_k: None)
        # Structured logger (JSONL). Config keys:
        #   logging_enabled (bool, default True)
        #   logging_run_id (str, optional)
        #   logging_dir (str, default "logs")
        #   logging_snapshot_sec (float, default 10.0)
        self.run_logger = RunLogger(
            run_id=cfg.get("logging_run_id"),
            log_dir=cfg.get("logging_dir", "logs"),
            snapshot_interval_sec=float(cfg.get("logging_snapshot_sec", 10.0)),
            enabled=bool(cfg.get("logging_enabled", True)),
        )
        self.run_logger.write_meta({"config": cfg})
        # Let the smart quoter emit pause/halt events to the structured log.
        if self.quoter_mode == "smart":
            self.quoter.attach_logger(self.run_logger)
        self.last_ts = 0
        self.event_count = 0
        self.equity_curve: list[tuple[int, float]] = []
        self._sample_interval_ns = int(1e8)   # 100 ms

    # ----- event handlers -----

    def on_book(self, u: BookUpdate) -> List[Fill]:
        self.books[u.venue].apply(u)
        self.sim.on_book_update(self.books[u.venue])
        if self.signals is not None:
            m = self.books[u.venue].microprice() or self.books[u.venue].mid()
            if m:
                self.signals.on_book(u.venue, m, u.ts)
        # Per-venue mid mark
        m = self.books[u.venue].mid()
        if m:
            self.inv.mark(m, venue=u.venue)
        self.last_ts = u.ts
        self._step(u.ts)
        return []

    def on_trade(self, t: Trade) -> List[Fill]:
        if self.signals is not None:
            self.signals.on_trade(t)
        fills = self.sim.on_trade(t)
        for f in fills:
            # Record cross-venue basis at fill time (only for quote-venue fills)
            qb = self.books[self.quote_venue]
            hb = self.books[self.hedge_venue]
            qb_mid = qb.mid() or 0
            hb_mid = hb.mid() or 0
            basis_bps = ((qb_mid - hb_mid) / qb_mid * 1e4) if qb_mid and hb_mid else 0.0
            if qb_mid and hb_mid:
                self.inv.record_basis_at_fill(basis_bps, f.side,
                                              f.price * f.size, f.venue)
            # Capture pre-fill inventory snapshot for the log
            pos_b_before = self.inv.pos[Venue.BINANCE]
            pos_y_before = self.inv.pos[Venue.BYBIT]
            net_before = self.inv.net_delta
            self.inv.apply_fill(f)
            # Structured fill log with full signal context
            self.run_logger.log_fill({
                "ts": f.ts,
                "venue": f.venue.value, "side": f.side.value,
                "price": f.price, "size": f.size,
                "is_maker": f.is_maker, "order_id": f.order_id,
                "qb_bid": qb.best_bid()[0] if qb.best_bid() else None,
                "qb_ask": qb.best_ask()[0] if qb.best_ask() else None,
                "qb_mid": qb_mid, "hb_mid": hb_mid,
                "basis_bps": basis_bps,
                "ofi_norm": self.signals.ofi_norm(f.venue) if self.signals else None,
                "vol_bps_sec": self.signals.realized_vol_bps_per_sec(qb_mid)
                                if self.signals and qb_mid else None,
                "mr_bps": self.signals.mean_reversion_bps(f.venue)
                            if self.signals else None,
                "pos_binance_before": pos_b_before,
                "pos_bybit_before": pos_y_before,
                "net_delta_before": net_before,
                "pos_binance_after": self.inv.pos[Venue.BINANCE],
                "pos_bybit_after":   self.inv.pos[Venue.BYBIT],
                "pnl_total_after":   self.inv.total_pnl(),
            })
            # If the filled order was one of the quoter's quotes, notify it.
            if f.venue == self.quote_venue:
                self.quoter.on_fill(f.side)
            else:
                self.hedger.on_fill(f.side)
            self.log_fn("FILL", f)
        self.last_ts = t.ts
        self._step(t.ts)
        return fills

    # ----- strategy step -----

    def _step(self, ts: int) -> None:
        qb = self.books[self.quote_venue]
        hb = self.books[self.hedge_venue]
        if not qb.ready or not hb.ready:
            return
        qmid = qb.mid()
        if qmid is None:
            return

        # MtM
        self.inv.mark(qmid)
        self.event_count += 1
        if ts - (self.equity_curve[-1][0] if self.equity_curve else 0) > self._sample_interval_ns:
            self.equity_curve.append((ts, self.inv.total_pnl()))

        # Periodic structured snapshot (rate-limited inside the logger)
        self.run_logger.maybe_snapshot(ts, {
            "pnl_total": self.inv.total_pnl(),
            "pnl_trading_cash": self.inv.pnl["trading_cash"],
            "pnl_inventory_mtm": self.inv.pnl["inventory_mtm"],
            "pnl_maker_rebate": self.inv.pnl["maker_rebate"],
            "pnl_taker_fee": self.inv.pnl["taker_fee"],
            "pos_binance": self.inv.pos[Venue.BINANCE],
            "pos_bybit":   self.inv.pos[Venue.BYBIT],
            "net_delta":   self.inv.net_delta,
            "n_maker_fills": self.inv.n_maker_fills,
            "n_taker_fills": self.inv.n_taker_fills,
            "n_quotes_posted": getattr(self, "_n_quotes_posted", 0),
            "notional_traded": self.inv.notional_traded,
            "qb_mid": qmid,
            "hb_mid": hb.mid() or 0,
            "ofi_norm": self.signals.ofi_norm(self.quote_venue) if self.signals else None,
            "vol_bps_sec": self.signals.realized_vol_bps_per_sec(qmid) if self.signals else None,
            "mr_bps": self.signals.mean_reversion_bps(self.quote_venue) if self.signals else None,
        })

        net_delta = self.inv.net_delta

        # 1) Quoter — pass current touch so it can anchor to BBO if configured.
        qb_bb = qb.best_bid()
        qb_ba = qb.best_ask()
        quote_bb = qb_bb[0] if qb_bb else 0.0
        quote_ba = qb_ba[0] if qb_ba else 0.0
        if self.quoter_mode == "smart":
            microprice = qb.microprice() or qmid
            # Smart quoter needs the touch; pass via set_touch().
            self.quoter.set_touch(quote_bb, quote_ba)
            self.quoter.observe_pnl(ts, self.inv.total_pnl())
            actions = self.quoter.step(ts, qmid, net_delta, microprice,
                                        self.signals, hb)
        else:
            actions = self.quoter.step(ts, qmid, net_delta, quote_bb, quote_ba)
        for kind, q in actions:
            if kind in ("place", "replace"):
                # place_maker enforces "one quoter order per venue+side":
                # it auto-cancels any stale quoter order at the same venue+side,
                # which protects against partial-fill phantom orders.
                self.sim.place_maker(q, qb, ts, role="quoter")
                self.log_fn(kind.upper(), q)
                self._n_quotes_posted = getattr(self, "_n_quotes_posted", 0) + 1
            elif kind == "cancel":
                self.sim.cancel(q.order_id)
                self.log_fn("CANCEL", q)

        # 2) Hedger
        hb_best = (hb.best_bid(), hb.best_ask())
        if hb_best[0] is None or hb_best[1] is None:
            return
        best_bid = hb_best[0][0]
        best_ask = hb_best[1][0]
        hactions = self.hedger.step(ts, net_delta, best_bid, best_ask)
        for kind, h in hactions:
            if kind == "place":
                self.sim.place_maker(h, hb, ts, role="hedger")
                self.log_fn("HEDGE_PLACE", h)
            elif kind == "cancel":
                self.sim.cancel(h.order_id)
                self.log_fn("HEDGE_CANCEL", h)
            elif kind == "taker":
                fill = self.sim.fill_taker(h, hb, ts)
                if fill:
                    self.inv.apply_fill(fill)
                    self.hedger.on_fill(fill.side)
                    self.log_fn("HEDGE_TAKER_FILL", fill)
