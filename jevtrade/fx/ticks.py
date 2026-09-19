"""Dukascopy tick files, and the entry rule a one-second reader would really get.

Yahoo's 5-minute bars are a judge with two defects. They only go back sixty
days, which caps the whole study at thirty documents, and they have no bid and
no ask, so every arm trades at a mid price that nobody is quoted. Dukascopy
publishes free tick files back to 2003 for the majors, with both sides of the
book, which fixes both.

What the format is (probed, not assumed):

* ``datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5``
  with **MM zero-based** -- January is ``00``. ``DD`` and ``HH`` are two digits
  and ``HH`` is the UTC hour.
* The body is LZMA. Decompressed it is consecutive 20-byte big-endian records,
  ``>IIIff`` = (milliseconds since the start of the hour, ask, bid, ask volume,
  bid volume). Prices are integers scaled by **1e5**, or by **1e3** when the
  quote currency is JPY.
* An hour with no ticks -- every weekend hour, most holidays -- answers 200 with
  a zero-byte body; a date the feed does not have answers 404. Both mean "no
  ticks" and both are cached as such, because a study that re-asks for every
  Saturday of seventeen years spends its afternoon on them. A 5xx or a dropped
  connection is transient, is retried with backoff, and is *not* cached: caching
  it would turn a bad minute of network into a permanent hole in the tape.

What the measurement then claims, and what it does not:

* **Measured** -- the entry. The first tick at or after ``timestamp + latency``,
  paying the **ask** to go long and hitting the **bid** to go short. The half
  spread is a real cost and it is in the number.
* **Measured** -- the rush. How far the mid moved between the publication
  timestamp and that entry tick, signed by the side taken. If a latency of one
  second already gives away the move, the latency sweep says so.
* **Assumed** -- that the exit is free. The forward return is taken to the
  **mid**, so a round trip pays half the spread and not the whole of it. That is
  the friendly end of the honest range; a trade that had to hit the other side
  again would pay ``spread_bps`` more.
* **Assumed** -- that a tick you can see is a tick you could have traded.
  Dukascopy publishes its own aggregated feed; size is not modelled at all.
"""

from __future__ import annotations

import base64
import bisect
import concurrent.futures
import lzma
import math
import random
import struct
import sys
import threading
import time
import http.client
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, NamedTuple, Sequence

from ..listing.store import Store
from .reader import TICK_SYMBOLS

FEED_URL = (
    "https://datafeed.dukascopy.com/datafeed/{symbol}"
    "/{year:04d}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
)
UA = {"User-Agent": "Mozilla/5.0 (compatible; jev-trade research)"}

RECORD = struct.Struct(">IIIff")
RECORD_SIZE = RECORD.size  # 20
HOUR = 3600

# Prices are integers. Everything quoted in JPY is scaled by a thousand and
# everything else by a hundred thousand; getting this backwards moves a price by
# two orders of magnitude and every return with it.
JPY_SCALE = 1e3
DEFAULT_SCALE = 1e5

DEFAULT_LATENCY_S = 1.0
LATENCIES = (0.0, 1.0, 5.0, 30.0, 120.0)
HORIZONS = (1, 5, 15, 30, 60)
PRE_MIN = 15
# A quote older than this is not a quote: the tape is shut, or the hour is a
# hole, and a price from before the weekend is not the price at this moment.
STALE_S = 300.0
# ... and if nothing prints within this long after the entry moment, there was
# nobody to trade with.
WAIT_S = 120.0

MAX_HOURS = 256  # decoded hours kept in memory; the bytes stay on disk
TRIES = 5
BACKOFF_S = 1.0


class TickFeedError(RuntimeError):
    """A transient failure -- retried, never cached as "this hour is empty"."""


class Tick(NamedTuple):
    ts: float  # epoch seconds, milliseconds included
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


class Quote(NamedTuple):
    bid: float
    ask: float
    ts: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def _ts(tick: Tick) -> float:
    return tick.ts


# ------------------------------------------------------------------- the feed


def feed_symbol(pair: str) -> str:
    """``"EURUSD=X"`` -> ``"EURUSD"``; an unknown name is passed through."""
    return TICK_SYMBOLS.get(pair, pair.replace("=X", ""))


def scale_for(symbol: str) -> float:
    return JPY_SCALE if symbol.upper().endswith("JPY") else DEFAULT_SCALE


def hour_url(symbol: str, dt_utc: datetime) -> str:
    """The file for one UTC hour. The month is zero-based; everything else is not."""
    return FEED_URL.format(
        symbol=symbol.upper(), year=dt_utc.year, month=dt_utc.month - 1,
        day=dt_utc.day, hour=dt_utc.hour,
    )


def hour_start(when: float) -> int:
    return int(when // HOUR) * HOUR


def hours_covering(start: float, end: float) -> list[int]:
    """Every hour start whose file could hold a tick in ``[start, end]``."""
    if end < start:
        start, end = end, start
    return list(range(hour_start(start), hour_start(end) + HOUR, HOUR))


def decode_bi5(raw: bytes, scale: float, *, at: float = 0.0) -> list[Tick]:
    """LZMA'd 20-byte records -> ticks. ``at`` is the hour's start in epoch seconds.

    An empty body is an hour with no ticks, which is most of every weekend, so
    it is a normal answer rather than an error.
    """
    if not raw:
        return []
    body = lzma.decompress(raw)
    return [
        Tick(at + ms / 1000.0, bid / scale, ask / scale)
        for ms, ask, bid, _av, _bv in RECORD.iter_unpack(
            body[: len(body) - len(body) % RECORD_SIZE]
        )
    ]


_local = threading.local()


def _ssl_context() -> ssl.SSLContext:
    cafile = (os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
              or os.environ.get("CURL_CA_BUNDLE") or None)
    return ssl.create_default_context(cafile=cafile)


def _connection(host: str, timeout: float) -> http.client.HTTPSConnection:
    """One persistent TLS connection per thread, tunnelled through the proxy if set.

    Measured from this environment: the first request on a connection to the
    feed costs 9-16 s (the handshake), every request after it about 0.2 s. A
    fresh connection per file therefore caps the fetch at ~5 files a minute and
    trips the feed's 503s under concurrency; one connection per worker, kept
    open, does ~300 a minute.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "host", None) == host:
        conn.timeout = timeout
        return conn
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        parsed = urllib.parse.urlparse(proxy)
        conn = http.client.HTTPSConnection(
            parsed.hostname or "", parsed.port, timeout=timeout, context=_ssl_context(),
        )
        headers = {}
        if parsed.username:
            token = base64.b64encode(
                f"{parsed.username}:{parsed.password or ''}".encode()
            ).decode("ascii")
            headers["Proxy-Authorization"] = f"Basic {token}"
        conn.set_tunnel(host, 443, headers=headers)
    else:
        conn = http.client.HTTPSConnection(host, 443, timeout=timeout, context=_ssl_context())
    _local.conn, _local.host = conn, host
    return conn


def _drop_connection() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _local.conn, _local.host = None, None


def _get(url: str, timeout: float = 30.0) -> bytes:
    """GET over the thread's persistent connection; ``HTTPError`` on a non-200."""
    parts = urllib.parse.urlparse(url)
    conn = _connection(parts.hostname or "", timeout)
    try:
        conn.request("GET", parts.path or "/", headers={**UA, "Connection": "keep-alive"})
        response = conn.getresponse()
        body = response.read()
    except (http.client.HTTPException, OSError):
        _drop_connection()
        raise
    if response.status != 200:
        if not response.getheader("Connection", "").lower() == "keep-alive":
            _drop_connection()
        raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
    return body


def fetch_bi5(url: str, *, timeout: float = 30.0, tries: int = TRIES) -> bytes:
    """The raw file, ``b""`` for an hour that has none. Raises on transient failures.

    404 is a settled answer (the feed has no such file) and so is a zero-byte
    200 (the hour is empty). Everything else -- 5xx, a reset, a timeout -- is the
    feed having a bad minute and is retried, then reported, never cached.

    The backoff is jittered because eight workers that all fail at once and all
    sleep the same second come back as one burst and fail again; the feed
    answers 503 under load, which is exactly the condition retrying is for.
    """
    last = ""
    for attempt in range(max(1, tries)):
        try:
            return _get(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return b""
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < tries:
            time.sleep(BACKOFF_S * 2**attempt * (0.5 + random.random()))
    raise TickFeedError(f"{url}: {last}")


def fetch_hour(
    store: Store,
    symbol: str,
    hour: int,
    *,
    scale: float | None = None,
    fetcher: Callable[[str], bytes] | None = None,
) -> list[Tick]:
    """One UTC hour of ticks, cached as the **compressed** bytes it arrived as.

    The compressed form is a tenth of the decoded one and is what makes a
    seventeen-year run resumable on a laptop disk: an interrupted run re-reads
    the files it already has and only fetches the hours it never reached.
    """
    scale = scale_for(symbol) if scale is None else scale
    fetch = fetcher or fetch_bi5
    key = f"dukas:{symbol}:{datetime.fromtimestamp(hour, timezone.utc):%Y-%m-%dT%H}"
    found, encoded = store.get(key)
    if not found:
        url = hour_url(symbol, datetime.fromtimestamp(hour, timezone.utc))
        try:
            raw = fetch(url)
        except TickFeedError as exc:
            print(f"  {exc}", file=sys.stderr)
            return []
        encoded = base64.b64encode(raw or b"").decode("ascii")
        store.put(key, encoded)
    try:
        return decode_bi5(base64.b64decode(encoded or ""), scale, at=hour)
    except (lzma.LZMAError, ValueError) as exc:
        print(f"  {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []


# ------------------------------------------------------------------- the tape


class TickTape:
    """Every tick of one pair, fetched an hour at a time and only where asked.

    Hours are fetched in parallel, cached on disk as bytes and in memory as
    decoded lists, with the decoded cache bounded: a seventeen-year run touches
    tens of thousands of hours and would otherwise hold a hundred million ticks
    in RAM. Evicting one costs an LZMA decompress, not a request.
    """

    def __init__(
        self,
        store: Store,
        symbol: str,
        *,
        workers: int = 8,
        fetcher: Callable[[str], bytes] | None = None,
        max_hours: int = MAX_HOURS,
        stale_s: float = STALE_S,
        wait_s: float = WAIT_S,
    ) -> None:
        self.store = store
        self.pair = symbol
        self.symbol = feed_symbol(symbol)
        self.scale = scale_for(self.symbol)
        self.workers = max(1, workers)
        self.fetcher = fetcher
        self.max_hours = max(1, max_hours)
        self.stale_s = stale_s
        self.wait_s = wait_s
        self.fetched = 0
        self._hours: "OrderedDict[int, list[Tick]]" = OrderedDict()
        self._locks: dict[int, threading.Lock] = {}
        self._guard = threading.Lock()

    # -- hours

    def _cached(self, hour: int) -> list[Tick] | None:
        with self._guard:
            ticks = self._hours.get(hour)
            if ticks is not None:
                self._hours.move_to_end(hour)
            return ticks

    def hour(self, hour: int) -> list[Tick]:
        """The ticks of one UTC hour, fetched once however many threads ask."""
        ticks = self._cached(hour)
        if ticks is not None:
            return ticks
        with self._guard:
            lock = self._locks.setdefault(hour, threading.Lock())
        # One fetch per hour at a time: the on-disk cache writes through a
        # temporary file named after the key, so two threads would race for it.
        with lock:
            ticks = self._cached(hour)
            if ticks is not None:
                return ticks
            ticks = fetch_hour(self.store, self.symbol, hour,
                               scale=self.scale, fetcher=self.fetcher)
            with self._guard:
                self.fetched += 1
                self._hours[hour] = ticks
                self._hours.move_to_end(hour)
                while len(self._hours) > self.max_hours:
                    self._hours.popitem(last=False)
            return ticks

    def load(self, hours: Sequence[int]) -> None:
        """Fetch the hours that are not in memory yet, in parallel."""
        missing = [h for h in dict.fromkeys(hours) if self._cached(h) is None]
        if len(missing) > 1 and self.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                min(self.workers, len(missing))
            ) as pool:
                list(pool.map(self.hour, missing))
        else:
            for h in missing:
                self.hour(h)

    # -- queries

    def window(self, start: float, end: float) -> list[Tick]:
        """Every tick in ``[start, end]``, in order."""
        hours = hours_covering(start, end)
        self.load(hours)
        out: list[Tick] = []
        for h in hours:
            ticks = self.hour(h)
            if not ticks:
                continue
            lo = bisect.bisect_left(ticks, start, key=_ts)
            hi = bisect.bisect_right(ticks, end, key=_ts)
            out.extend(ticks[lo:hi])
        return out

    def quote_at(self, when: float, *, max_age_s: float | None = None) -> Quote | None:
        """The last tick at or before ``when``, or None if the tape is shut."""
        max_age = self.stale_s if max_age_s is None else max_age_s
        hours = hours_covering(when - max_age, when)
        self.load(hours)
        for h in reversed(hours):
            ticks = self.hour(h)
            if not ticks:
                continue
            i = bisect.bisect_right(ticks, when, key=_ts) - 1
            if i >= 0 and ticks[i].ts >= when - max_age:
                return Quote(ticks[i].bid, ticks[i].ask, ticks[i].ts)
        return None

    def first_tick_after(self, when: float, *, max_wait_s: float | None = None) -> Tick | None:
        """The first tick at or after ``when`` -- the entry, once latency is added."""
        wait = self.wait_s if max_wait_s is None else max_wait_s
        hours = hours_covering(when, when + wait)
        self.load(hours)
        for h in hours:
            ticks = self.hour(h)
            if not ticks:
                continue
            i = bisect.bisect_left(ticks, when, key=_ts)
            if i < len(ticks) and ticks[i].ts <= when + wait:
                return ticks[i]
        return None

    def mid_at(self, when: float, *, max_age_s: float | None = None) -> float | None:
        quote = self.quote_at(when, max_age_s=max_age_s)
        return None if quote is None else quote.mid


# --------------------------------------------------------------- measurement


@dataclass
class TickOutcome:
    """One trade, priced on both sides of the book.

    ``fwd_bps`` is the number the study reports: it pays the spread going in and
    exits at the mid. ``fwd_mid_bps`` is the same trade with no spread at all --
    the bar study's number -- and the gap between the two is what the half
    spread costs at that latency.
    """

    entry_ts: float
    entry_px: float
    entry_mid: float
    pre_bps: float  # unsigned, like the bar tape
    rush_bps: float  # signed by the side taken
    spread_bps: float
    latency_s: float
    fwd_bps: dict[int, float] = field(default_factory=dict)
    fwd_mid_bps: dict[int, float] = field(default_factory=dict)


def log_bps(a: float, b: float) -> float | None:
    if not (a > 0 and b > 0) or math.isnan(a) or math.isnan(b):
        return None
    return math.log(b / a) * 1e4


def forward_returns_ticks(
    tape: TickTape,
    when: float,
    sign: int,
    *,
    latency_s: float = DEFAULT_LATENCY_S,
    horizons_min: Iterable[int] = HORIZONS,
    pre_min: int = PRE_MIN,
) -> TickOutcome | None:
    """What a reader that answered in ``latency_s`` would have got, or None.

    None means the tape could not answer honestly: no quote within five minutes
    of a point the measurement needs, or no tick within two minutes of the entry
    moment. A Friday-evening event whose sixtieth minute is after the close is
    therefore dropped whole, exactly as it is on bars -- a "+60m" return measured
    across a weekend is a 51-hour return in disguise.
    """
    horizons = tuple(horizons_min)
    # Everything this measurement can ask for, fetched in one parallel batch:
    # asking hour by hour as the questions come up turns three files into eight
    # round trips, which over 2,300 documents is the difference between an
    # afternoon and a week.
    tape.load(hours_covering(
        when - pre_min * 60 - tape.stale_s,
        when + latency_s + max(horizons, default=0) * 60 + tape.wait_s,
    ))
    at_release = tape.quote_at(when)
    before = tape.quote_at(when - pre_min * 60)
    if at_release is None or before is None:
        return None
    pre = log_bps(before.mid, at_release.mid)
    if pre is None:
        return None

    entry = tape.first_tick_after(when + latency_s)
    if entry is None:
        return None
    entry_px = entry.ask if sign > 0 else entry.bid
    entry_mid = entry.mid
    if not (entry_px > 0 and entry_mid > 0 and entry.ask >= entry.bid > 0):
        return None
    rush = log_bps(at_release.mid, entry_mid)
    if rush is None:
        return None
    spread_bps = (entry.ask - entry.bid) / entry_mid * 1e4

    forward: dict[int, float] = {}
    forward_mid: dict[int, float] = {}
    for minutes in horizons:
        exit_quote = tape.quote_at(when + latency_s + minutes * 60)
        if exit_quote is None:
            return None
        net = log_bps(entry_px, exit_quote.mid)
        gross = log_bps(entry_mid, exit_quote.mid)
        if net is None or gross is None:
            return None
        forward[minutes] = sign * net
        forward_mid[minutes] = sign * gross

    return TickOutcome(
        entry_ts=entry.ts, entry_px=entry_px, entry_mid=entry_mid,
        pre_bps=pre, rush_bps=sign * rush, spread_bps=spread_bps,
        latency_s=latency_s, fwd_bps=forward, fwd_mid_bps=forward_mid,
    )


def has_ticks(
    tape: TickTape,
    when: float,
    *,
    latency_s: float = DEFAULT_LATENCY_S,
    horizons_min: Iterable[int] = HORIZONS,
    pre_min: int = PRE_MIN,
) -> bool:
    """Whether a moment could be measured at all -- used to place the nulls.

    The cheap half first: a random moment within five days of an event lands in
    the weekend about a third of the time, and one ``quote_at`` rejects it for
    one empty hour file instead of the dozen a full measurement would ask for.
    """
    horizons = tuple(horizons_min)
    if tape.quote_at(when) is None:
        return False
    if horizons and tape.quote_at(when + latency_s + max(horizons) * 60) is None:
        return False
    return forward_returns_ticks(
        tape, when, 1, latency_s=latency_s, horizons_min=horizons, pre_min=pre_min
    ) is not None


__all__ = [
    "BACKOFF_S", "DEFAULT_LATENCY_S", "DEFAULT_SCALE", "FEED_URL", "HORIZONS",
    "JPY_SCALE",
    "LATENCIES", "MAX_HOURS", "PRE_MIN", "Quote", "STALE_S", "Tick",
    "TRIES", "TickFeedError", "TickOutcome", "TickTape", "WAIT_S", "decode_bi5",
    "feed_symbol", "fetch_bi5", "fetch_hour", "forward_returns_ticks",
    "has_ticks", "hour_start", "hour_url", "hours_covering", "log_bps",
    "scale_for",
]
