"""Load a structured log directory and print fill-quality statistics.

Usage:
    python backtest/analyze_logs.py logs/<run_id>

What it shows:
  - PnL evolution from snapshots.jsonl
  - Fill distribution: side, venue, basis distribution at fill time
  - Adverse-selection proxy: for each fill, look at mid N seconds later;
    if it moved AGAINST our fill direction, the fill was adverse.
  - Signal vs. adverse-cost correlation table: helps you decide whether OFI,
    basis, vol, MR are actually predicting toxicity.

Output is all printed; nothing is written. Extend as needed for plotting.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def _load(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def analyze(run_dir: str) -> None:
    d = Path(run_dir)
    fills = _load(d / "fills.jsonl")
    snaps = _load(d / "snapshots.jsonl")
    print(f"Run: {d.name}    fills={len(fills)}    snapshots={len(snaps)}")
    print()

    # --- PnL trajectory from snapshots ---
    if snaps:
        first, last = snaps[0], snaps[-1]
        print("PnL trajectory:")
        print(f"  start ts {first['ts']:>22d}  pnl={first['pnl_total']:>+8.4f}")
        print(f"  end   ts {last['ts']:>22d}  pnl={last['pnl_total']:>+8.4f}")
        pmax = max(s["pnl_total"] for s in snaps)
        pmin = min(s["pnl_total"] for s in snaps)
        print(f"  peak  pnl: {pmax:>+8.4f}   trough pnl: {pmin:>+8.4f}")
        print()

    # --- Fill distribution ---
    by_venue_side = defaultdict(int)
    basis_at_fill = []
    ofi_at_fill = []
    for f in fills:
        by_venue_side[(f["venue"], f["side"])] += 1
        if f.get("basis_bps") is not None:
            basis_at_fill.append(f["basis_bps"])
        if f.get("ofi_norm") is not None:
            ofi_at_fill.append(f["ofi_norm"])

    print("Fill counts by venue/side:")
    for k, n in sorted(by_venue_side.items()):
        print(f"  {k[0]:>8}  {k[1]:>4}  : {n}")
    print()

    if basis_at_fill:
        b = sorted(basis_at_fill)
        print(f"Basis at fill (bps):  min {b[0]:+.2f}  median {b[len(b)//2]:+.2f}  "
              f"mean {st.fmean(b):+.2f}  max {b[-1]:+.2f}")
    if ofi_at_fill:
        o = sorted(ofi_at_fill)
        print(f"OFI_norm at fill   :  min {o[0]:+.2f}  median {o[len(o)//2]:+.2f}  "
              f"mean {st.fmean(o):+.2f}  max {o[-1]:+.2f}")
    print()

    # --- Adverse-selection proxy ---
    # For each fill at ts0, find the snapshot ~5s after ts0 and compare mid.
    # If we BOUGHT and mid went DOWN, that's adverse. Vice versa for SELL.
    snap_by_ts = sorted([(s["ts"], s.get("qb_mid", 0.0)) for s in snaps])
    if snap_by_ts and fills:
        WINDOW_NS = 5_000_000_000   # 5 seconds
        adverse_bps_list = []
        for f in fills:
            if f["venue"] != "binance" or not f.get("qb_mid"):
                continue
            target_ts = f["ts"] + WINDOW_NS
            # binary search-ish: linear is fine for prototype
            future_mid = None
            for ts, mid in snap_by_ts:
                if ts >= target_ts and mid > 0:
                    future_mid = mid
                    break
            if future_mid is None:
                continue
            sign = -1 if f["side"] == "bid" else +1   # we bought (bid) → adverse if mid falls
            # adverse_bps = (-Δmid * sign) / mid * 1e4
            # if we bought and mid fell, Δmid<0, sign=-1, adverse_bps positive (bad).
            delta = future_mid - f["qb_mid"]
            adverse_bps = (-delta * sign) / f["qb_mid"] * 1e4
            adverse_bps_list.append((adverse_bps, f.get("basis_bps", 0.0),
                                     f.get("ofi_norm", 0.0), f.get("mr_bps", 0.0)))

        if adverse_bps_list:
            avg = st.fmean(a for a, *_ in adverse_bps_list)
            adverse_count = sum(1 for a, *_ in adverse_bps_list if a > 0)
            print(f"Adverse-selection (5s post-fill mid move):")
            print(f"  fills analysed: {len(adverse_bps_list)}")
            print(f"  avg adverse bps: {avg:+.3f}  (positive = bad)")
            print(f"  fraction adverse: {adverse_count}/{len(adverse_bps_list)} "
                  f"= {100 * adverse_count / len(adverse_bps_list):.1f}%")
            print()

            # Bucket by signal sign to see if signals were predictive
            print("Adverse cost bucketed by signal sign (mean bps; negative = good):")
            for name, idx in (("basis>0", 1), ("basis<0", 1),
                              ("ofi>0", 2), ("ofi<0", 2),
                              ("mr>0", 3), ("mr<0", 3)):
                if name.endswith(">0"):
                    bucket = [a for a, *xs in adverse_bps_list if xs[idx-1] > 0]
                else:
                    bucket = [a for a, *xs in adverse_bps_list if xs[idx-1] < 0]
                if bucket:
                    print(f"  {name:>10}: n={len(bucket):>4}  "
                          f"avg adverse bps = {st.fmean(bucket):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="path to logs/<run_id>")
    args = ap.parse_args()
    analyze(args.run_dir)


if __name__ == "__main__":
    main()
