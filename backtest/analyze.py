"""Post-run analyzer: pretty-prints the PnL ledger and plots equity curve."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from strategy.engine import Engine


def print_summary(engine: Engine) -> None:
    s = engine.inv.summary()
    print()
    print("=" * 56)
    print("  Run summary")
    print("=" * 56)
    print(f"  events processed     : {engine.event_count:>12,}")
    print(f"  notional traded (USD): {s['notional_traded']:>12,.2f}")
    print(f"  maker fills          : {s['n_maker_fills']:>12,}")
    print(f"  taker fills          : {s['n_taker_fills']:>12,}")
    print(f"  net delta (contracts): {s['net_delta']:>12.6f}")
    print(f"  pos binance          : {s['pos_binance']:>12.6f}")
    print(f"  pos bybit            : {s['pos_bybit']:>12.6f}")
    print()
    print("  PnL breakdown (USD)")
    print(f"    trading_cash       : {s['pnl_trading_cash']:>12.4f}    (realized buys/sells, ex-fees)")
    print(f"    inventory_mtm      : {s['pnl_inventory_mtm']:>12.4f}    (open position * mid, per-venue)")
    print(f"    maker_rebate       : {s['pnl_maker_rebate']:>12.4f}    (negative ⇒ we paid maker fee)")
    print(f"    taker_fee          : {s['pnl_taker_fee']:>12.4f}    (always ≤ 0)")
    print(f"    funding            : {s['pnl_funding']:>12.4f}")
    print("    " + "-" * 30)
    print(f"    TOTAL              : {s['pnl_total']:>12.4f}")
    print()
    print("  Where the trading edge actually came from:")
    print(f"    binance leg PnL    : {s['pnl_venue_binance']:>12.4f}    (trading_cash_binance + pos_binance*mid_binance)")
    print(f"    bybit leg PnL      : {s['pnl_venue_bybit']:>12.4f}    (trading_cash_bybit + pos_bybit*mid_bybit)")
    print(f"    sum (= trading_cash + inventory_mtm) : {s['pnl_venue_binance'] + s['pnl_venue_bybit']:>8.4f}")
    print()
    print(f"  Cross-exchange basis at fill time:")
    print(f"    avg basis bps fav.  : {s['avg_fill_basis_bps_favourable']:>12.4f}    (positive ⇒ we filled when basis was in our favour)")
    print(f"    basis $ proxy       : {s['cross_exch_basis_dollar_proxy']:>12.4f}    (~bps * notional / 1e4, how much $ that basis contributed)")
    print("=" * 56)


def plot_equity(engine: Engine, out_path: Optional[str] = None) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[analyze] matplotlib unavailable, skipping plot: {e}")
        return None
    if not engine.equity_curve:
        print("[analyze] no equity samples to plot")
        return None
    xs = [(t - engine.equity_curve[0][0]) * 1e-9 for t, _ in engine.equity_curve]
    ys = [v for _, v in engine.equity_curve]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=110)
    ax.plot(xs, ys, color="#3354dd")
    ax.set_title("Strategy equity curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PnL (USD)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    out = out_path or "plots/equity.png"
    Path(out).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[analyze] plot saved -> {out}")
    return out
