"""Structured run logger.

Writes three JSONL files per run, under `logs/<run_id>/`:

  fills.jsonl       — one line per fill, with full signal context at fill time.
                      Use this for adverse-selection analysis: load into pandas,
                      compute mid moves N ms after each fill, see which signals
                      correlate with toxic vs benign fills.

  events.jsonl      — significant strategy events: pause/unpause, drawdown halt,
                      side suppression, hedge timeout escalation. NOT every quote
                      replace — would explode the log.

  snapshots.jsonl   — periodic PnL + inventory + signal snapshot (every
                      `snapshot_interval_sec`). Use this to plot equity curve,
                      inventory trajectory, signal evolution.

  meta.json         — config + start/end timestamps for the run.

All timestamps are integer nanoseconds since epoch (matching the engine).
Each line is a complete JSON object so the files are trivial to ingest:

    import pandas as pd
    fills = pd.read_json("logs/<run_id>/fills.jsonl", lines=True)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def _default(o: Any):
    """JSON serializer for things json.dump can't handle by default."""
    if hasattr(o, "value"):     # enums
        return o.value
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


class RunLogger:
    def __init__(self, run_id: Optional[str] = None,
                 log_dir: str = "logs",
                 snapshot_interval_sec: float = 10.0,
                 enabled: bool = True) -> None:
        self.enabled = enabled
        if not enabled:
            return
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%S")
        self.dir = Path(log_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fills = open(self.dir / "fills.jsonl", "w", buffering=64 * 1024)
        self._events = open(self.dir / "events.jsonl", "w", buffering=64 * 1024)
        self._snaps = open(self.dir / "snapshots.jsonl", "w", buffering=64 * 1024)
        self._snap_interval_ns = int(snapshot_interval_sec * 1e9)
        self._last_snap_ts = 0
        self.n_fills = 0
        self.n_events = 0
        self.n_snaps = 0

    def write_meta(self, meta: dict) -> None:
        if not self.enabled:
            return
        with open(self.dir / "meta.json", "w") as f:
            json.dump(meta, f, default=_default, indent=2)

    def log_fill(self, payload: dict) -> None:
        if not self.enabled:
            return
        self._fills.write(json.dumps(payload, default=_default) + "\n")
        self.n_fills += 1

    def log_event(self, event_type: str, ts: int, **payload) -> None:
        if not self.enabled:
            return
        payload["type"] = event_type
        payload["ts"] = ts
        self._events.write(json.dumps(payload, default=_default) + "\n")
        self.n_events += 1

    def maybe_snapshot(self, ts: int, payload: dict) -> None:
        if not self.enabled:
            return
        if ts - self._last_snap_ts < self._snap_interval_ns:
            return
        payload["ts"] = ts
        self._snaps.write(json.dumps(payload, default=_default) + "\n")
        self._last_snap_ts = ts
        self.n_snaps += 1

    def close(self) -> None:
        if not self.enabled:
            return
        for f in (self._fills, self._events, self._snaps):
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        print(f"[logger] wrote {self.n_fills} fills, "
              f"{self.n_events} events, {self.n_snaps} snapshots to {self.dir}")
