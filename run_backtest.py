"""Backtest entry point.

Two modes:
  --synthetic                run on the in-memory tick generator
  --date YYYY-MM-DD          run on cached real ticks (must run
                             data/fetch_historical.py first)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# repo-root-relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy.engine import Engine
from backtest.replay import replay
from backtest.analyze import print_summary, plot_equity


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic ticks instead of cached real data")
    ap.add_argument("--minutes", type=float, default=5.0,
                    help="(synthetic only) minutes of ticks to simulate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sigma", type=float, default=25.0,
                    help="(synthetic only) USD/minute mid-price volatility")
    ap.add_argument("--spread-ticks", type=int, default=4,
                    help="(synthetic only) synthetic top-of-book spread in ticks")
    ap.add_argument("--date", default=None,
                    help="(real data) YYYY-MM-DD; reads data/cache/*")
    ap.add_argument("--max-events", type=int, default=None,
                    help="stop replay after N events (useful on full-day historical data)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--plot", default="plots/equity.png")
    ap.add_argument("--capital", type=float, default=50_000.0,
                    help="assumed capital base for performance metrics (default $50k)")
    ap.add_argument("--rf", type=float, default=0.05,
                    help="risk-free rate per year for Sharpe / Sortino (default 5%)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).resolve().parent / cfg_path
    cfg = _load_cfg(str(cfg_path))

    engine = Engine(cfg)

    if args.synthetic:
        from data.synth import generate
        events = generate(minutes=args.minutes, seed=args.seed,
                          sigma_per_min=args.sigma,
                          spread_ticks=args.spread_ticks)
        t0 = time.time()
        replay(events, engine, verbose=args.verbose, max_events=args.max_events)
        dt = time.time() - t0
        print(f"[run] synthetic backtest done in {dt:.2f}s "
              f"({engine.event_count / max(dt, 1e-6):.0f} events/s)")
    elif args.date:
        from data.historical_reader import merged_stream
        cache_dir = Path(__file__).resolve().parent / "data" / "cache"
        events = merged_stream(args.date, cache_dir)
        t0 = time.time()
        replay(events, engine, verbose=args.verbose, max_events=args.max_events)
        dt = time.time() - t0
        print(f"[run] historical replay done in {dt:.2f}s "
              f"({engine.event_count / max(dt, 1e-6):.0f} events/s)")
    else:
        ap.error("specify --synthetic or --date YYYY-MM-DD")

    print_summary(engine)
    plot_path = Path(__file__).resolve().parent / args.plot
    plot_equity(engine, str(plot_path))

    # Close the structured logger BEFORE computing performance, so the JSONL
    # files are flushed and complete.
    engine.close()

    # Compute and persist performance metrics into the same logs/<run_id>/
    # directory the engine just wrote to. Saves performance.txt (human-readable)
    # and performance.json (machine-readable).
    if engine.run_logger.enabled:
        try:
            from backtest.performance import compute as _perf_compute
            from backtest.performance import print_report, write_report
            run_dir = str(engine.run_logger.dir)
            metrics = _perf_compute(run_dir, capital_usd=args.capital,
                                     rf_annual=args.rf,
                                     snapshot_sec=float(cfg.get("logging_snapshot_sec", 10.0)))
            print_report(metrics)
            write_report(metrics, run_dir)
            print(f"[performance] wrote performance.txt and performance.json to {run_dir}")
        except Exception as e:
            print(f"[performance] skipped: {e}")


if __name__ == "__main__":
    main()
