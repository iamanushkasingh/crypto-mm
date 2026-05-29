"""Comprehensive performance metrics for a backtest run.

Usage:
    python backtest/performance.py logs/<run_id> [--capital 50000] [--rf 0.05]

Computes and prints (with explanations):

  RETURN-BASED
    CAGR, Volatility, Sharpe, Sortino, Calmar, Maximum Drawdown,
    Drawdown duration, Annualised P&L per dollar of capital

  TRADE-BASED
    Win rate (fills with positive 5-second forward edge), Profit Factor,
    Avg Win, Avg Loss, Expectancy (per fill)

  RISK-ADJUSTED
    Alpha and Beta vs BTC buy-and-hold over the same window,
    Information Ratio, Tracking Error

  MM-SPECIFIC
    Slippage (mid - fill price, per side), Implementation Shortfall,
    Fill Ratio (fills / quotes posted), Turnover (notional / capital),
    Average Exposure (|position| × mid / capital), Peak Exposure

  RISK
    VaR (95%, 99%) and CVaR (95%, 99%) on N-second PnL changes,
    Inventory Risk (max |pos|, RMS |pos|, position vol)

  CAPACITY (proxy)
    "Queue volume traversed" — how much of the per-level book size we
    consumed across all fills. A signal for how large the strategy can
    realistically scale before fills become depth-limited.

All annualisations assume 365 trading days × 86,400 seconds = ~31.5M s/yr.

Inputs (auto-detected from run_dir):
    snapshots.jsonl   — periodic PnL & inventory & mid (every snapshot_sec)
    fills.jsonl       — every fill with signal context
    meta.json         — config used for the run
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any


SEC_PER_YEAR = 365.0 * 86400.0


def _load(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _mean(xs):
    return st.fmean(xs) if xs else 0.0


def _std(xs):
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def _percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(p * (len(s) - 1))
    return s[k]


def compute(run_dir: str, capital_usd: float = 50_000.0,
            rf_annual: float = 0.05, snapshot_sec: float = 10.0
            ) -> dict[str, Any]:
    d = Path(run_dir)
    snaps = _load(d / "snapshots.jsonl")
    fills = _load(d / "fills.jsonl")
    if not snaps:
        raise SystemExit(f"no snapshots found in {run_dir}")

    # --- Time window ---
    t0_ns = snaps[0]["ts"]
    tN_ns = snaps[-1]["ts"]
    duration_sec = (tN_ns - t0_ns) * 1e-9 if tN_ns > t0_ns else 1.0
    duration_yrs = duration_sec / SEC_PER_YEAR

    # --- PnL series ---
    pnl_series = [s["pnl_total"] for s in snaps]
    final_pnl = pnl_series[-1]
    # Returns in USD per snapshot interval — convert to "return on capital"
    returns_pct = [(pnl_series[i] - pnl_series[i-1]) / capital_usd
                   for i in range(1, len(pnl_series))]

    # --- Sharpe / Sortino / Vol ---
    mean_ret_per_period = _mean(returns_pct)
    std_ret_per_period = _std(returns_pct)
    periods_per_year = SEC_PER_YEAR / snapshot_sec
    rf_per_period = rf_annual / periods_per_year
    excess = [r - rf_per_period for r in returns_pct]
    sharpe = ((_mean(excess) / _std(excess)) * math.sqrt(periods_per_year)
              if _std(excess) > 0 else float("nan"))
    downside = [min(0.0, e) for e in excess]
    # Population variance of downside (semideviation)
    downside_std = math.sqrt(_mean([d * d for d in downside]))
    sortino = ((_mean(excess) / downside_std) * math.sqrt(periods_per_year)
               if downside_std > 0 else float("nan"))
    vol_annual = std_ret_per_period * math.sqrt(periods_per_year)

    # --- CAGR (on capital) ---
    final_equity_ratio = 1 + final_pnl / capital_usd
    cagr = (final_equity_ratio ** (1 / duration_yrs) - 1) if duration_yrs > 0 else 0.0

    # --- Max Drawdown ---
    peak = pnl_series[0]
    max_dd = 0.0
    max_dd_start_idx = 0
    max_dd_end_idx = 0
    cur_peak_idx = 0
    for i, p in enumerate(pnl_series):
        if p > peak:
            peak = p
            cur_peak_idx = i
        dd = peak - p
        if dd > max_dd:
            max_dd = dd
            max_dd_start_idx = cur_peak_idx
            max_dd_end_idx = i
    max_dd_pct = max_dd / capital_usd
    dd_duration_sec = ((snaps[max_dd_end_idx]["ts"] - snaps[max_dd_start_idx]["ts"])
                       * 1e-9)
    calmar = (cagr / max_dd_pct) if max_dd_pct > 0 else float("inf")

    # --- Alpha / Beta / IR vs BTC buy-and-hold ---
    btc = [s.get("qb_mid", 0.0) for s in snaps]
    btc_returns = [(btc[i] / btc[i-1] - 1) if btc[i-1] > 0 else 0.0
                   for i in range(1, len(btc))]
    if len(btc_returns) == len(returns_pct) and len(btc_returns) > 2:
        # OLS: returns = alpha + beta * btc_returns
        x_mean, y_mean = _mean(btc_returns), _mean(returns_pct)
        x_var = _mean([(x - x_mean) ** 2 for x in btc_returns])
        x_y_cov = _mean([(x - x_mean) * (y - y_mean)
                         for x, y in zip(btc_returns, returns_pct)])
        beta = x_y_cov / x_var if x_var > 0 else 0.0
        alpha_per_period = y_mean - beta * x_mean
        alpha_annual = alpha_per_period * periods_per_year
        tracking_diff = [r - bb for r, bb in zip(returns_pct, btc_returns)]
        tracking_error = _std(tracking_diff) * math.sqrt(periods_per_year)
        info_ratio = (_mean(tracking_diff) * periods_per_year / tracking_error
                      if tracking_error > 0 else float("nan"))
    else:
        beta = alpha_annual = tracking_error = info_ratio = float("nan")

    btc_return_total = (btc[-1] / btc[0] - 1) if btc and btc[0] > 0 else 0.0

    # --- Trade-based metrics (per fill) ---
    # Define each fill's PnL as: (forward_mid_5s - fill_price) * sign
    # where sign = +1 for sells, -1 for buys (forward-mid moving WITH our trade is good)
    snap_idx_by_ts = sorted([(s["ts"], s.get("qb_mid", 0.0)) for s in snaps])
    WINDOW_NS = 5_000_000_000
    fill_edges = []
    fill_sizes = []
    slippage_bid = []   # for buy fills: fill_price - mid_at_fill (we paid, so positive = paid above mid)
    slippage_ask = []   # for sell fills: mid_at_fill - fill_price (we received, positive = below mid)
    queue_volume_traversed = 0.0
    for f in fills:
        if f["venue"] != "binance" or not f.get("qb_mid"):
            continue
        target_ts = f["ts"] + WINDOW_NS
        future_mid = None
        for ts, mid in snap_idx_by_ts:
            if ts >= target_ts and mid > 0:
                future_mid = mid
                break
        if future_mid is None:
            continue
        sign = -1 if f["side"] == "bid" else +1
        edge = (future_mid - f["qb_mid"]) * sign  # positive = move was in our favour
        fill_edges.append(edge)
        fill_sizes.append(f["size"])
        if f["side"] == "bid":
            slippage_bid.append(f["qb_mid"] - f["price"])   # positive = we bought below mid (good)
        else:
            slippage_ask.append(f["price"] - f["qb_mid"])   # positive = we sold above mid (good)
        queue_volume_traversed += f["size"]

    wins = [e for e in fill_edges if e > 0]
    losses = [e for e in fill_edges if e < 0]
    win_rate = len(wins) / len(fill_edges) if fill_edges else 0.0
    avg_win = _mean(wins) if wins else 0.0
    avg_loss = _mean(losses) if losses else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    expectancy = _mean(fill_edges) if fill_edges else 0.0  # per fill, in $ of mid move

    # --- Implementation Shortfall ---
    # Compare actual realised PnL to a "frictionless" benchmark: if every fill
    # had been at the exact mid with zero fees, what would PnL be?
    # Frictionless PnL = sum_over_fills( sign(sell - buy) * mid_at_fill * size )
    # Roughly: with maker fills, you generally trade *at* your quote price,
    # which is offset from mid by the depth. So shortfall ≈ depth × notional.
    impl_shortfall_usd = 0.0
    for f in fills:
        if not f.get("qb_mid"):
            continue
        # We OWED to trade at mid; we actually traded at fill price.
        # For a buy (bid) at price P < mid: P − mid is negative → we got it cheaper than mid → POSITIVE shortfall (we benefited).
        # For a sell (ask) at price P > mid: P − mid is positive → POSITIVE shortfall (we benefited).
        sign = -1 if f["side"] == "bid" else +1
        impl_shortfall_usd += sign * (f["price"] - f["qb_mid"]) * f["size"]

    # --- VaR / CVaR on N-second PnL changes (already in returns_pct, in pct of capital) ---
    var_95 = -_percentile(returns_pct, 0.05) * capital_usd
    var_99 = -_percentile(returns_pct, 0.01) * capital_usd
    tail_95 = [r for r in returns_pct if r <= _percentile(returns_pct, 0.05)]
    tail_99 = [r for r in returns_pct if r <= _percentile(returns_pct, 0.01)]
    cvar_95 = -_mean(tail_95) * capital_usd if tail_95 else 0.0
    cvar_99 = -_mean(tail_99) * capital_usd if tail_99 else 0.0

    # --- Inventory Risk ---
    net_pos = [s["net_delta"] for s in snaps]
    abs_pos = [abs(p) for p in net_pos]
    max_inv = max(abs_pos)
    rms_inv = math.sqrt(_mean([p * p for p in abs_pos]))
    avg_mid = _mean(btc)
    max_inv_usd = max_inv * avg_mid
    rms_inv_usd = rms_inv * avg_mid

    # --- Fill ratio ---
    n_quotes_posted = snaps[-1].get("n_quotes_posted", 0) or 0
    n_maker_fills = snaps[-1].get("n_maker_fills", 0) or 0
    n_taker_fills = snaps[-1].get("n_taker_fills", 0) or 0
    fill_ratio = (n_maker_fills / n_quotes_posted) if n_quotes_posted > 0 else 0.0

    # --- Turnover ---
    notional_traded = snaps[-1].get("notional_traded", 0.0)
    turnover_ratio = notional_traded / capital_usd

    # --- Exposure ---
    pos_usd = [abs(p) * m for p, m in zip(net_pos, btc)]
    avg_exposure_usd = _mean(pos_usd)
    peak_exposure_usd = max(pos_usd) if pos_usd else 0.0
    avg_exposure_ratio = avg_exposure_usd / capital_usd
    peak_exposure_ratio = peak_exposure_usd / capital_usd

    # --- Capacity proxy ---
    # If our average fill size is `s_f`, the typical level size is `L`, and we
    # consume `qpf * L` queue ahead per fill, the strategy effectively absorbs
    # ~ s_f + qpf*L of book size each fill. Doubling the size would double both
    # the queue we sit behind AND the queue we consume — fill rate roughly
    # halves. So "capacity" before degradation is on the order of L / s_f * size.
    avg_fill_size = _mean(fill_sizes) if fill_sizes else 0.0
    # We can also report the total volume of size traversed (queue we ate)
    capacity_proxy_notional = avg_fill_size * avg_mid * 10  # 10x heuristic — could quote bigger before degrading by ~50%

    return {
        "duration_sec": duration_sec,
        "duration_min": duration_sec / 60,
        "capital_usd": capital_usd,
        "final_pnl_usd": final_pnl,
        "pnl_pct": final_pnl / capital_usd * 100,
        # return-based
        "cagr_pct": cagr * 100,
        "vol_annual_pct": vol_annual * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd_usd": max_dd,
        "max_dd_pct": max_dd_pct * 100,
        "max_dd_duration_sec": dd_duration_sec,
        # benchmark
        "btc_return_pct": btc_return_total * 100,
        "alpha_annual_pct": alpha_annual * 100,
        "beta": beta,
        "tracking_error_pct": tracking_error * 100,
        "info_ratio": info_ratio,
        # trade-based
        "n_fills": len(fill_edges),
        "win_rate_pct": win_rate * 100,
        "avg_win_usd_per_btc": avg_win,
        "avg_loss_usd_per_btc": avg_loss,
        "profit_factor": profit_factor,
        "expectancy_usd_per_btc_per_fill": expectancy,
        # MM-specific
        "slippage_bid_avg_usd": _mean(slippage_bid) if slippage_bid else 0.0,
        "slippage_ask_avg_usd": _mean(slippage_ask) if slippage_ask else 0.0,
        "implementation_shortfall_usd": impl_shortfall_usd,
        "fill_ratio_pct": fill_ratio * 100,
        "n_quotes_posted": n_quotes_posted,
        "n_maker_fills": n_maker_fills,
        "n_taker_fills": n_taker_fills,
        "turnover_ratio": turnover_ratio,
        "notional_traded_usd": notional_traded,
        "avg_exposure_usd": avg_exposure_usd,
        "peak_exposure_usd": peak_exposure_usd,
        "avg_exposure_ratio": avg_exposure_ratio,
        "peak_exposure_ratio": peak_exposure_ratio,
        # risk
        "var_95_usd": var_95,
        "var_99_usd": var_99,
        "cvar_95_usd": cvar_95,
        "cvar_99_usd": cvar_99,
        "max_inv_btc": max_inv,
        "rms_inv_btc": rms_inv,
        "max_inv_usd": max_inv_usd,
        "rms_inv_usd": rms_inv_usd,
        # capacity
        "avg_fill_size_btc": avg_fill_size,
        "capacity_proxy_notional_usd": capacity_proxy_notional,
    }


def _fmt(x: float, suffix: str = "", precision: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x:>{12 + precision}.{precision}f}{suffix}"


def print_report(m: dict) -> None:
    print()
    print("=" * 78)
    print("  PERFORMANCE METRICS")
    print("=" * 78)
    print(f"  Window: {m['duration_min']:.1f} minutes "
          f"({m['duration_sec']:.0f} s)   Capital: ${m['capital_usd']:,.0f}")
    print(f"  Final PnL: ${m['final_pnl_usd']:+.4f}  "
          f"({m['pnl_pct']:+.4f}% on capital)")
    print()
    print("  ── RETURN-BASED ─────────────────────────────────────────────")
    print(f"    CAGR (annualised)            : {_fmt(m['cagr_pct'], '%')}")
    print(f"    Volatility (annualised)      : {_fmt(m['vol_annual_pct'], '%')}")
    print(f"    Sharpe Ratio                 : {_fmt(m['sharpe'])}")
    print(f"    Sortino Ratio                : {_fmt(m['sortino'])}")
    print(f"    Calmar Ratio                 : {_fmt(m['calmar'])}")
    print(f"    Maximum Drawdown             : ${m['max_dd_usd']:>10.4f}  "
          f"({m['max_dd_pct']:+.4f}% of capital)")
    print(f"    Max-DD duration              : {m['max_dd_duration_sec']:>10.1f} s")
    print()
    print("  ── BENCHMARK (BTC buy-and-hold over the same window) ────────")
    print(f"    BTC return over window       : {_fmt(m['btc_return_pct'], '%')}")
    print(f"    Alpha (annualised, vs BTC)   : {_fmt(m['alpha_annual_pct'], '%')}")
    print(f"    Beta (vs BTC)                : {_fmt(m['beta'])}")
    print(f"    Tracking error (annualised)  : {_fmt(m['tracking_error_pct'], '%')}")
    print(f"    Information Ratio            : {_fmt(m['info_ratio'])}")
    print()
    print("  ── TRADE-BASED ──────────────────────────────────────────────")
    print(f"    Fills analysed (5s forward)  : {m['n_fills']:>11}")
    print(f"    Win rate                     : {_fmt(m['win_rate_pct'], '%')}")
    print(f"    Avg WIN  (USD per BTC)       : {_fmt(m['avg_win_usd_per_btc'])}")
    print(f"    Avg LOSS (USD per BTC)       : {_fmt(m['avg_loss_usd_per_btc'])}")
    print(f"    Profit Factor                : {_fmt(m['profit_factor'])}")
    print(f"    Expectancy per fill (USD/BTC): {_fmt(m['expectancy_usd_per_btc_per_fill'])}")
    print()
    print("  ── MM-SPECIFIC ──────────────────────────────────────────────")
    print(f"    Slippage on BID fills (USD)  : {_fmt(m['slippage_bid_avg_usd'])}  "
          "(positive = bought below mid; good)")
    print(f"    Slippage on ASK fills (USD)  : {_fmt(m['slippage_ask_avg_usd'])}  "
          "(positive = sold above mid; good)")
    print(f"    Implementation Shortfall (USD): {_fmt(m['implementation_shortfall_usd'])}  "
          "(positive = better than mid)")
    print(f"    Quotes posted                : {m['n_quotes_posted']:>11,}")
    print(f"    Maker fills                  : {m['n_maker_fills']:>11,}")
    print(f"    Taker fills                  : {m['n_taker_fills']:>11,}")
    print(f"    Fill ratio (maker/quotes)    : {_fmt(m['fill_ratio_pct'], '%')}")
    print(f"    Notional traded              : ${m['notional_traded_usd']:>14,.2f}")
    print(f"    Turnover (notional / capital): {_fmt(m['turnover_ratio'], 'x')}")
    print(f"    Average exposure (USD)       : ${m['avg_exposure_usd']:>12.2f}  "
          f"({m['avg_exposure_ratio']*100:.4f}% of capital)")
    print(f"    Peak exposure (USD)          : ${m['peak_exposure_usd']:>12.2f}  "
          f"({m['peak_exposure_ratio']*100:.4f}% of capital)")
    print()
    print("  ── RISK ─────────────────────────────────────────────────────")
    print(f"    VaR  95% ({m['duration_min']:.0f}min, snapshot Δ): ${m['var_95_usd']:>10.4f}")
    print(f"    VaR  99% ({m['duration_min']:.0f}min, snapshot Δ): ${m['var_99_usd']:>10.4f}")
    print(f"    CVaR 95%                     : ${m['cvar_95_usd']:>10.4f}")
    print(f"    CVaR 99%                     : ${m['cvar_99_usd']:>10.4f}")
    print(f"    Max inventory (BTC)          : {m['max_inv_btc']:>11.6f}  "
          f"(${m['max_inv_usd']:.2f})")
    print(f"    RMS inventory (BTC)          : {m['rms_inv_btc']:>11.6f}  "
          f"(${m['rms_inv_usd']:.2f})")
    print()
    print("  ── CAPACITY (proxy) ─────────────────────────────────────────")
    print(f"    Avg fill size (BTC)          : {m['avg_fill_size_btc']:>11.6f}")
    print(f"    Capacity proxy notional ≈    : ${m['capacity_proxy_notional_usd']:>14,.2f}  "
          "(rough estimate of size before fill-rate halves)")
    print("=" * 78)


def write_report(m: dict, run_dir: str) -> None:
    """Persist metrics to disk so the user has them alongside the other logs."""
    import io, contextlib, json as _json
    out = Path(run_dir)
    # Pretty text
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(m)
    (out / "performance.txt").write_text(buf.getvalue())
    # Machine-readable
    (out / "performance.json").write_text(_json.dumps(m, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="path to logs/<run_id>")
    ap.add_argument("--capital", type=float, default=50_000.0,
                    help="assumed capital base in USD (default 50000)")
    ap.add_argument("--rf", type=float, default=0.05,
                    help="risk-free rate per year (default 5%)")
    ap.add_argument("--snapshot-sec", type=float, default=10.0,
                    help="snapshot interval used during the run (default 10)")
    ap.add_argument("--no-save", action="store_true",
                    help="only print, don't write performance.txt/.json")
    args = ap.parse_args()
    m = compute(args.run_dir, capital_usd=args.capital,
                rf_annual=args.rf, snapshot_sec=args.snapshot_sec)
    print_report(m)
    if not args.no_save:
        write_report(m, args.run_dir)
        print(f"\n[performance] wrote performance.txt and performance.json to {args.run_dir}")


if __name__ == "__main__":
    main()
