"""Read cached historical archives and yield BookUpdate / Trade events,
merged across Binance + Bybit and sorted by event time.

Binance aggTrades CSV columns (USDⓂ-perp):
    agg_id, price, qty, first_id, last_id, ts_ms, buyer_is_maker, is_best_match

Binance bookTicker CSV columns:
    update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty,
    transaction_time, event_time

Bybit trading archive CSV columns (linear):
    timestamp, symbol, side, size, price, tickDirection, trdMatchID,
    grossValue, homeNotional, foreignNotional

Bybit doesn't publish bookTicker in its public archive, so we synthesize
best-bid/ask updates from each trade by nudging the prior best on the
trade's side. This is a pragmatic approximation — for true L2 backtest
you'd buy LOBSTER-style data or record it yourself live.
"""
from __future__ import annotations

import csv
import gzip
import io
import zipfile
from pathlib import Path
from typing import Iterator, List, Tuple, Union

from core.events import BookUpdate, Trade, Venue, Side


def _open_zip_member(path: Path) -> Iterator[List[str]]:
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                # CSVs are utf-8 ASCII; skip header if present
                text = io.TextIOWrapper(f, encoding="utf-8", newline="")
                reader = csv.reader(text)
                for row in reader:
                    yield row


def _open_gz(path: Path) -> Iterator[List[str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            yield row


def binance_book_events(book_ticker_zip: Path) -> Iterator[BookUpdate]:
    first = True
    for row in _open_zip_member(book_ticker_zip):
        if first:
            first = False
            # Skip header if non-numeric in first cell
            try:
                int(row[0])
            except (ValueError, IndexError):
                continue
        try:
            bb = float(row[1]); bbq = float(row[2])
            ba = float(row[3]); baq = float(row[4])
            ts_ms = int(row[5])
        except (ValueError, IndexError):
            continue
        yield BookUpdate(
            ts=ts_ms * 1_000_000,
            venue=Venue.BINANCE,
            bids=[(bb, bbq)],
            asks=[(ba, baq)],
            is_snapshot=True,    # bookTicker is always a fresh top-of-book
        )


def binance_trade_events(agg_trades_zip: Path) -> Iterator[Trade]:
    first = True
    for row in _open_zip_member(agg_trades_zip):
        if first:
            first = False
            try:
                int(row[0])
            except (ValueError, IndexError):
                continue
        try:
            price = float(row[1])
            qty = float(row[2])
            ts_ms = int(row[5])
            buyer_is_maker = row[6].lower() in ("true", "1")
        except (ValueError, IndexError):
            continue
        # buyer_is_maker=True → seller was aggressor → ASK aggressor
        aggressor = Side.ASK if buyer_is_maker else Side.BID
        yield Trade(ts=ts_ms * 1_000_000, venue=Venue.BINANCE,
                    price=price, size=qty, aggressor=aggressor)


def bybit_trade_events(trades_gz: Path) -> Iterator[Trade]:
    first = True
    for row in _open_gz(trades_gz):
        if first:
            first = False
            try:
                float(row[0])
            except (ValueError, IndexError):
                continue
        try:
            ts_s = float(row[0])
            side = row[2].lower()
            size = float(row[3])
            price = float(row[4])
        except (ValueError, IndexError):
            continue
        # side "Buy" = taker bought = BID aggressor
        aggressor = Side.BID if side.startswith("buy") else Side.ASK
        yield Trade(ts=int(ts_s * 1e9), venue=Venue.BYBIT,
                    price=price, size=size, aggressor=aggressor)


def _trade_to_synth_book(t: Trade, tick: float = 0.1) -> BookUpdate:
    """Produce a synthetic top-of-book update from a single trade."""
    if t.aggressor == Side.BID:
        last_ask = t.price
        last_bid = t.price - tick
    else:
        last_bid = t.price
        last_ask = t.price + tick
    return BookUpdate(ts=t.ts, venue=t.venue,
                      bids=[(last_bid, 1.0)],
                      asks=[(last_ask, 1.0)],
                      is_snapshot=True)


def trades_with_synth_book(trade_iter: Iterator[Trade], tick: float = 0.1
                           ) -> Iterator[Union[Trade, BookUpdate]]:
    """Single-pass generator: per trade, yield (Trade, synth BookUpdate)."""
    for t in trade_iter:
        yield t
        yield _trade_to_synth_book(t, tick)


def merged_stream(date: str, cache_dir: Path
                  ) -> Iterator[Union[BookUpdate, Trade]]:
    """Yield all events from both venues for `date`, ordered by ts.

    For each venue, real book data is used if present in the cache;
    otherwise top-of-book is synthesized from the trade stream.
    """
    ym = date[:7]
    bnc_book_d = cache_dir / f"binance_bookTicker_{date}.zip"
    bnc_book_m = cache_dir / f"binance_bookTicker_{ym}.zip"
    bnc_trd_d = cache_dir / f"binance_aggTrades_{date}.zip"
    bnc_trd_m = cache_dir / f"binance_aggTrades_{ym}.zip"
    bbt_trd = cache_dir / f"bybit_trading_{date}.csv.gz"

    streams: list[Iterator] = []

    # Binance — single-pass streaming; never materialize the whole day.
    bnc_book = bnc_book_d if bnc_book_d.exists() else (bnc_book_m if bnc_book_m.exists() else None)
    bnc_trd = bnc_trd_d if bnc_trd_d.exists() else (bnc_trd_m if bnc_trd_m.exists() else None)
    if bnc_trd:
        if bnc_book:
            streams.append(binance_trade_events(bnc_trd))
            streams.append(binance_book_events(bnc_book))
        else:
            print("[reader] binance bookTicker missing — synthesizing top-of-book from aggTrades (streaming)")
            streams.append(trades_with_synth_book(binance_trade_events(bnc_trd)))

    # Bybit — no L2 archive published, always stream-synth.
    if bbt_trd.exists():
        streams.append(trades_with_synth_book(bybit_trade_events(bbt_trd)))

    # K-way merge by ts
    import heapq
    heads = []
    for i, s in enumerate(streams):
        try:
            ev = next(s)
            heads.append((ev.ts, i, ev, s))
        except StopIteration:
            pass
    heapq.heapify(heads)
    while heads:
        ts, i, ev, s = heapq.heappop(heads)
        yield ev
        try:
            nxt = next(s)
            heapq.heappush(heads, (nxt.ts, i, nxt, s))
        except StopIteration:
            pass
