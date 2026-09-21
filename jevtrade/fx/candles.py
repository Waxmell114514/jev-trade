"""Dukascopy 1-minute candles: a judge cheap enough to grade twenty thousand posts.

The tick judge in ``ticks.py`` is the right one and it is too expensive here. A
year of wire posts is twenty-one thousand events on seven pairs; each one needs
a quarter of an hour before and an hour after, which on ticks is dozens of
hour-files per event and tens of gigabytes across the study. The same feed
publishes **one file per pair per day per side** of 1-minute candles: 1,440
records, 12 kB compressed, 3,600 files for a year of seven pairs both sides.
That is three orders of magnitude less network for a judge that is coarser by
one minute.

What the format is (probed on EURUSD 2025-04-02 and USDJPY, not assumed):

* ``datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/BID_candles_min_1.bi5``
  and ``ASK_candles_min_1.bi5``, with **MM zero-based** exactly like the tick
  files -- January is ``00``, so the April 2025 file is ``2025/03/02``.
* The body is LZMA. Decompressed it is consecutive 24-byte big-endian records,
  ``struct.unpack(">IIIIIf")`` = *(seconds since 00:00 UTC, open, close, low,
  high, volume)*, 1,440 of them on a trading day. Prices are integers scaled by
  **1e5**, or **1e3** when the quote currency is JPY: the first EURUSD record of
  2025-04-02 is ``(0, 107947, 107931, 107931, 107950, 90.6)`` -> open 1.07947.
* A date the feed does not have answers 404, and some answer 200 with an empty
  body. Both mean "no bars" and both are cached as such.
* **The weekend is padded, which the tick files are not, and this is the one
  place the format will silently corrupt a study.** A Saturday file is not
  empty: it is 1,440 records at Friday's closing price with **volume zero**.
  Checked on 2025-04-05, EURUSD, NZDUSD and USDCHF: 1,440 bars, 0 with volume.
  Friday's own file carries volume to 20:59 UTC and pads 21:00-23:59; Sunday
  pads until 21:00 and then has 180 real minutes. Taken at face value those
  bars give a weekend "trade" a guaranteed 0 bp return and turn a Friday-
  evening +60m into a 51-hour return in disguise. So **a zero-volume minute is
  treated as no minute**: a bar the aggregated feed saw no trade in is not a
  price anybody could have transacted at, and dropping it reproduces the tick
  tape's weekend behaviour exactly.

**The entry rule, stated so the cost is visible.** A signal enters at the
**open of the first bar whose start is at or after ``when + latency``** -- so
the latency always rounds *up* to the next minute boundary, and a one-second
reader and a fifty-nine-second one get the same fill. It pays the **ASK open to
go long and the BID open to go short**, and exits at the **mid close** of the
bar ``h`` minutes later. A round trip therefore pays half the spread, which is
the friendly end of the honest range.

What that claims and what it does not:

* **Measured** -- the half spread at entry, from the two files' opens at the
  same minute, and the forward return net of it.
* **Assumed** -- that the exit is free, as in the tick study.
* **Not measured** -- anything inside a minute. On these bars a latency of 0 s
  and one of 59 s are the same trade. The latency sweep says so where it prints.
"""

from __future__ import annotations

import base64
import bisect
import concurrent.futures
import lzma
import struct
import sys
import threading
import time
import urllib.error
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable, NamedTuple, Sequence

from ..listing.store import Store
from .ticks import DEFAULT_SCALE, JPY_SCALE, TickFeedError, fetch_bi5, log_bps, scale_for

CANDLE_URL = (
    "https://datafeed.dukascopy.com/datafeed/{symbol}"
    "/{year:04d}/{month:02d}/{day:02d}/{side}_candles_min_1.bi5"
)

RECORD = struct.Struct(">IIIIIf")
RECORD_SIZE = RECORD.size  # 24
MINUTE = 60
DAY = 86400
BARS_PER_DAY = 1440

BID, ASK = "BID", "ASK"
SIDES = (BID, ASK)

# The seven the wire actually talks about. USDCNH exists on the feed and is not
# here: the wire's China headlines move the yuan on a fix nobody trades in the
# same window, and an eighth pair would be a hypothesis, not a control.
SYMBOLS = ("EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD")

DEFAULT_LATENCY_S = 1.0
# On minute bars 0 s and 60 s are the same fill unless the post lands exactly on
# a boundary, so the sweep is 0 / 60 / 300 and the table says why.
LATENCIES = (0.0, 60.0, 300.0)
HORIZONS = (1, 5, 15, 30, 60)
PRE_MIN = 15
MAX_DAYS = 512  # decoded days held in memory; the bytes stay on disk


class Bar(NamedTuple):
    ts: float  # the bar's start, epoch seconds UTC
    open: float
    close: float
    low: float
    high: float
    volume: float


class Minute(NamedTuple):
    """One minute with both sides of the book, which is the point of two files."""

    ts: float
    bid: Bar
    ask: Bar

    @property
    def mid_open(self) -> float:
        return (self.bid.open + self.ask.open) / 2.0

    @property
    def mid_close(self) -> float:
        return (self.bid.close + self.ask.close) / 2.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid_open
        return (self.ask.open - self.bid.open) / mid * 1e4 if mid > 0 else 0.0

    def entry(self, sign: int) -> float:
        """The price actually paid: the ask to go long, the bid to go short."""
        return self.ask.open if sign > 0 else self.bid.open


def _ts(bar: Minute) -> float:
    return bar.ts


# ------------------------------------------------------------------- the feed


def day_url(symbol: str, when: date, side: str) -> str:
    """One pair, one UTC day, one side. The month is zero-based; nothing else is."""
    return CANDLE_URL.format(symbol=symbol.upper(), year=when.year, month=when.month - 1,
                             day=when.day, side=side.upper())


def day_start(when: float) -> int:
    return int(when // DAY) * DAY


def to_date(when: float) -> date:
    return datetime.fromtimestamp(when, timezone.utc).date()


def days_covering(start: float, end: float) -> list[date]:
    if end < start:
        start, end = end, start
    first, last = to_date(start), to_date(end)
    out: list[date] = []
    day = first
    while day <= last:
        out.append(day)
        day += timedelta(days=1)
    return out


def decode_candles(raw: bytes, scale: float, day_start_ts: float = 0.0) -> list[Bar]:
    """LZMA'd 24-byte records -> bars. ``day_start_ts`` is 00:00 UTC in epoch seconds.

    An empty body is a day with no bars -- every weekend, most holidays -- which
    is a normal answer and not an error.
    """
    if not raw:
        return []
    body = lzma.decompress(raw)
    return [
        Bar(day_start_ts + seconds, open_ / scale, close / scale,
            low / scale, high / scale, volume)
        for seconds, open_, close, low, high, volume in RECORD.iter_unpack(
            body[: len(body) - len(body) % RECORD_SIZE]
        )
    ]


def fetch_day(
    store: Store,
    symbol: str,
    when: date,
    side: str,
    *,
    fetcher: Callable[[str], bytes] | None = None,
) -> bytes:
    """One day-file, cached as the **compressed** bytes it arrived as (base64).

    Compressed is a tenth of decoded and is what makes a year resumable: an
    interrupted warm-up re-reads what it has and fetches only what it never
    reached. A transient failure is reported and *not* cached, because caching
    one would turn a bad minute of network into a permanent hole in the tape.
    """
    key = f"dukas1m:{symbol.upper()}:{side.upper()}:{when.isoformat()}"
    found, encoded = store.get(key)
    if not found:
        fetch = fetcher or fetch_bi5
        try:
            raw = fetch(day_url(symbol, when, side))
        except TickFeedError as exc:
            print(f"  {exc}", file=sys.stderr)
            raise
        encoded = base64.b64encode(raw or b"").decode("ascii")
        store.put(key, encoded)
    try:
        return base64.b64decode(encoded or "")
    except ValueError:
        return b""


# ------------------------------------------------------------------- the tape


class MinuteTape:
    """Every minute of one pair, fetched a day at a time and only where asked.

    Both sides of one day are one unit: a minute with a bid and no ask has no
    spread and cannot price an entry, so a day whose two files disagree about
    which minutes exist keeps only the minutes both have -- **and only the ones
    with volume on both**, which is what keeps the feed's zero-volume weekend
    padding out of the study. Decoded days are held in a bounded LRU; evicting
    one costs an LZMA decompress, not a request.
    """

    minutes = True

    def __init__(
        self,
        store: Store,
        symbol: str,
        *,
        fetcher: Callable[[str], bytes] | None = None,
        max_days: int = MAX_DAYS,
    ) -> None:
        self.store = store
        self.symbol = symbol.upper()
        self.pair = self.symbol
        self.scale = scale_for(self.symbol)
        self.fetcher = fetcher
        self.max_days = max(1, max_days)
        self.fetched = 0
        self._days: "OrderedDict[date, list[Minute]]" = OrderedDict()
        self._locks: dict[date, threading.Lock] = {}
        self._guard = threading.Lock()

    def _cached(self, when: date) -> list[Minute] | None:
        with self._guard:
            bars = self._days.get(when)
            if bars is not None:
                self._days.move_to_end(when)
            return bars

    def bars(self, when: date | float) -> list[Minute]:
        """One UTC day's minutes, fetched once however many threads ask.

        An unreachable file is an empty day for this call and is *not* cached,
        so the next run asks again; a weekend is an empty day that is.
        """
        if not isinstance(when, date):
            when = to_date(when)
        bars = self._cached(when)
        if bars is not None:
            return bars
        with self._guard:
            lock = self._locks.setdefault(when, threading.Lock())
        # One fetch per day at a time: the on-disk cache writes through a
        # temporary file named after the key, so two threads would race for it.
        with lock:
            bars = self._cached(when)
            if bars is not None:
                return bars
            start = float(datetime(when.year, when.month, when.day,
                                   tzinfo=timezone.utc).timestamp())
            try:
                bid = decode_candles(
                    fetch_day(self.store, self.symbol, when, BID, fetcher=self.fetcher),
                    self.scale, start)
                ask = decode_candles(
                    fetch_day(self.store, self.symbol, when, ASK, fetcher=self.fetcher),
                    self.scale, start)
            except (TickFeedError, lzma.LZMAError, urllib.error.URLError) as exc:
                print(f"  {self.symbol} {when}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                # Remembered as empty for this run only -- never written to
                # disk -- so one unreachable file costs one round of retries
                # instead of one per query, and the next run asks again.
                with self._guard:
                    self._days[when] = []
                return []
            asks = {bar.ts: bar for bar in ask}
            # Both sides, and both with volume: a minute the feed padded (the
            # whole weekend, the hours around the Friday close) carries the last
            # real price at volume zero on both sides, and is not a minute
            # anything could have been done in. See the module docstring.
            merged = [
                Minute(bar.ts, bar, asks[bar.ts]) for bar in bid
                if bar.ts in asks and bar.volume > 0 and asks[bar.ts].volume > 0
            ]
            with self._guard:
                self.fetched += 1
                self._days[when] = merged
                self._days.move_to_end(when)
                while len(self._days) > self.max_days:
                    self._days.popitem(last=False)
            return merged

    def load(self, days: Sequence[date], *, workers: int = 2) -> None:
        missing = [d for d in dict.fromkeys(days) if self._cached(d) is None]
        if len(missing) > 1 and workers > 1:
            with concurrent.futures.ThreadPoolExecutor(min(workers, len(missing))) as pool:
                list(pool.map(self.bars, missing))
        else:
            for day in missing:
                self.bars(day)

    def bar_at(self, when: float) -> Minute | None:
        """The bar that *contains* ``when`` -- the minute it fell in, or None."""
        bars = self.bars(to_date(when))
        if not bars:
            return None
        index = bisect.bisect_right(bars, when, key=_ts) - 1
        if index < 0:
            return None
        bar = bars[index]
        return bar if bar.ts <= when < bar.ts + MINUTE else None

    def next_bar_open(self, when: float) -> Minute | None:
        """The first bar whose start is at or after ``when`` -- where an entry fills.

        It looks into the next day when the moment is in the last minute of one,
        and no further: a Friday-evening post whose next bar is on Sunday has no
        honest entry and gets None, exactly as the tick tape drops it.
        """
        for day in (to_date(when), to_date(when + DAY)):
            bars = self.bars(day)
            if not bars:
                continue
            index = bisect.bisect_left(bars, when, key=_ts)
            if index < len(bars):
                return bars[index]
        return None


class MinuteTapes:
    """One ``MinuteTape`` per symbol, shared by every arm and every null."""

    minutes = True

    def __init__(self, store: Store, *, fetcher: Callable[[str], bytes] | None = None,
                 factory: Callable[..., MinuteTape] | None = None) -> None:
        self.store = store
        self.fetcher = fetcher
        self.factory = factory or MinuteTape
        self._cache: dict[str, MinuteTape] = {}
        self._guard = threading.Lock()

    def get(self, symbol: str) -> MinuteTape:
        with self._guard:
            tape = self._cache.get(symbol.upper())
            if tape is None:
                tape = self.factory(self.store, symbol.upper(), fetcher=self.fetcher)
                self._cache[symbol.upper()] = tape
            return tape

    @property
    def fetched(self) -> int:
        with self._guard:
            return sum(t.fetched for t in self._cache.values())


# --------------------------------------------------------------- measurement


@dataclass
class Outcome:
    """One trade on minute bars, priced on both sides of the book.

    ``fwd_bps`` is the number reported: it pays the spread going in and exits at
    the mid close. ``fwd_mid_bps`` is the same trade with no spread at all, and
    the gap between the two is what the half spread cost at that minute.
    """

    entry_ts: float
    entry_px: float
    entry_mid: float
    pre_bps: float  # unsigned by the side; the caller signs it
    spread_bps: float
    latency_s: float
    fwd_bps: dict[int, float] = field(default_factory=dict)
    fwd_mid_bps: dict[int, float] = field(default_factory=dict)


def forward_returns_minutes(
    tape: MinuteTape,
    when: float,
    sign: int,
    *,
    latency_s: float = DEFAULT_LATENCY_S,
    horizons_min: Iterable[int] = HORIZONS,
    pre_min: int = PRE_MIN,
) -> Outcome | None:
    """What a reader that answered in ``latency_s`` would have got, or None.

    The entry is the **open of the first bar whose start is at or after
    ``when + latency_s``**, so the latency rounds *up* to the next minute
    boundary: on these bars a one-second reader and a fifty-nine-second one get
    the same fill, and the sweep prints that rather than implying a precision
    the judge does not have. A long pays the ASK open and a short is filled at
    the BID open; the exit is the **mid close** of the bar ``h`` minutes after
    the entry bar.

    None means the tape could not answer honestly: a missing day (the weekend,
    a holiday), a horizon that crosses into one, or a post with no bar fifteen
    minutes behind it to measure ``pre`` against. A "+60m" return computed
    across a Friday close would be a 51-hour return in disguise, so it is
    dropped whole instead.
    """
    horizons = tuple(horizons_min)
    before = tape.bar_at(when - pre_min * MINUTE)
    at_post = tape.bar_at(when)
    if before is None or at_post is None:
        return None
    pre = log_bps(before.mid_close, at_post.mid_close)
    if pre is None:
        return None

    entry = tape.next_bar_open(when + latency_s)
    if entry is None:
        return None
    entry_px = entry.entry(sign)
    entry_mid = entry.mid_open
    if not (entry_px > 0 and entry_mid > 0 and entry.ask.open >= entry.bid.open > 0):
        return None

    forward: dict[int, float] = {}
    forward_mid: dict[int, float] = {}
    for minutes in horizons:
        exit_bar = tape.bar_at(entry.ts + minutes * MINUTE)
        if exit_bar is None:
            return None
        net = log_bps(entry_px, exit_bar.mid_close)
        gross = log_bps(entry_mid, exit_bar.mid_close)
        if net is None or gross is None:
            return None
        forward[minutes] = sign * net
        forward_mid[minutes] = sign * gross

    return Outcome(
        entry_ts=entry.ts, entry_px=entry_px, entry_mid=entry_mid, pre_bps=pre,
        spread_bps=entry.spread_bps, latency_s=latency_s,
        fwd_bps=forward, fwd_mid_bps=forward_mid,
    )


def has_bars(
    tape: MinuteTape,
    when: float,
    *,
    latency_s: float = DEFAULT_LATENCY_S,
    horizons_min: Iterable[int] = HORIZONS,
    pre_min: int = PRE_MIN,
) -> bool:
    """Whether a moment could be measured at all -- used to place the nulls.

    The cheap half first: a random moment within five days of a post lands in
    the weekend about a third of the time, and one ``bar_at`` rejects it for one
    empty day file instead of the several a full measurement would ask for.
    """
    horizons = tuple(horizons_min)
    if tape.bar_at(when) is None:
        return False
    if horizons and tape.bar_at(when + latency_s + max(horizons) * MINUTE) is None:
        return False
    return forward_returns_minutes(
        tape, when, 1, latency_s=latency_s, horizons_min=horizons, pre_min=pre_min
    ) is not None


# --------------------------------------------------------------- warming up


@dataclass
class Warmth:
    """How much of a date range is on disk. Reported, because it bounds the study."""

    wanted: int = 0
    fetched: int = 0
    # Files the feed answered with nothing at all. It is *not* the weekend
    # count: weekend files come back full of zero-volume padding, which the
    # tape drops on decode rather than the fetch refusing.
    empty: int = 0
    failures: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.fetched}/{self.wanted} day-files on disk "
            f"({self.empty} empty -- weekends and holidays), "
            f"{self.failures} unreachable, {self.seconds:.0f}s"
        )


def warm(
    store: Store,
    symbols: Sequence[str],
    since: float,
    until: float,
    *,
    workers: int = 2,
    fetcher: Callable[[str], bytes] | None = None,
    progress_every: int = 200,
) -> Warmth:
    """Pre-fetch every pair/side/day file in a range, resumably, with progress.

    Two workers, not eight: the feed answers 503 under a storm of new
    connections and the keep-alive fetch is already fast once a connection is
    up. A file already on disk costs no request, so an interrupted run is
    resumed by running the same command again.
    """
    days = days_covering(since, until)
    jobs = [(symbol.upper(), day, side)
            for symbol in symbols for day in days for side in SIDES]
    out = Warmth(wanted=len(jobs))
    started = time.perf_counter()
    counter = {"done": 0}
    guard = threading.Lock()

    def one(job: tuple[str, date, str]) -> None:
        symbol, day, side = job
        try:
            raw = fetch_day(store, symbol, day, side, fetcher=fetcher)
        except (TickFeedError, urllib.error.URLError) as exc:
            with guard:
                out.failures += 1
            print(f"  {symbol} {day} {side}: {exc}", file=sys.stderr)
            return
        with guard:
            out.fetched += 1
            if not raw:
                out.empty += 1
            counter["done"] += 1
            done = counter["done"] + out.failures
            if progress_every and (done % progress_every == 0 or done == len(jobs)):
                out.seconds = time.perf_counter() - started
                print(f"  candles {done}/{len(jobs)} files, {out.empty} empty, "
                      f"{out.failures} failures, {out.seconds:.0f}s",
                      file=sys.stderr, flush=True)

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, jobs))
    out.seconds = time.perf_counter() - started
    return out


def coverage(store: Store, symbols: Sequence[str], since: float, until: float
             ) -> dict[str, tuple[int, int]]:
    """``{symbol: (files on disk, files the range wants)}`` -- no network at all."""
    days = days_covering(since, until)
    out: dict[str, tuple[int, int]] = {}
    for symbol in symbols:
        have = 0
        for day in days:
            for side in SIDES:
                key = f"dukas1m:{symbol.upper()}:{side}:{day.isoformat()}"
                have += 1 if store.get(key)[0] else 0
        out[symbol.upper()] = (have, len(days) * len(SIDES))
    return out


__all__ = [
    "ASK", "BARS_PER_DAY", "BID", "Bar", "CANDLE_URL", "DEFAULT_LATENCY_S",
    "DEFAULT_SCALE", "HORIZONS", "JPY_SCALE", "LATENCIES", "MAX_DAYS", "MINUTE",
    "Minute", "MinuteTape", "MinuteTapes", "Outcome", "PRE_MIN", "RECORD",
    "RECORD_SIZE", "SIDES", "SYMBOLS", "Warmth", "coverage", "day_start", "day_url",
    "days_covering", "decode_candles", "fetch_day", "forward_returns_minutes",
    "has_bars", "log_bps", "scale_for", "to_date", "warm",
]
