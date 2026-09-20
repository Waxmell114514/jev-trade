"""Score every arm on the same documents, with the spot tape as the judge.

An arm turns a document into signals: (pair, long or short). Each signal enters
after the published timestamp and is measured by signed log return at fixed
horizons. The null for each arm is the same pair and the same side at random
moments within five days, placed only where the tape actually has prices --
which on spot FX is most of the work, since a third of the random draws land in
a weekend.

There are two judges and the arms do not know which one they are being graded
by. The **bar** judge is Yahoo's 5-minute series: entry at the open of the first
bar after the release, mid prices, sixty days deep. The **tick** judge is
Dukascopy: entry on the first tick at or after the release plus a stated
latency, paying the ask to go long and the bid to go short, back to 2003. The
tick judge is the one that can answer the question the study is actually about,
because it is the only one that knows what a second is worth -- see
``latency_sweep``, which runs the same signals at 0, 1, 5, 30 and 120 seconds.

Four arms sit next to the reader:

* ``keyword-bot``  the incumbent: hawkish words minus dovish words
* ``surprise-bot`` the other incumbent: printed rate minus snapshotted forecast,
                   live only for weeks with a calendar snapshot
* ``all text``     every document, taking the keyword bot's side (or long the
                   issuer's currency when the words tie). This arm answers the
                   question the whole study rests on -- do central-bank *text*
                   events move the spot tape at all, relative to nothing
                   happening -- and if it does not, nothing downstream matters.
* ``reader``       the two-round tree, at a strength threshold

The context run adds two more, both of them numbers rather than readings, so
the relative reader has something to be relative *against*:

* ``bill-surprise`` the decision the statement took against the one the
                    six-month bill implied. It is a crude proxy for fed funds
                    futures, which are not free, and it is labelled as one.
* ``dots-surprise`` the change in next year's median dot against the previous
                    SEP, on projection meetings only.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import random
import re
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..listing.store import Store
from . import tape as _tape
from .baseline import BotSignal, announced_rate, keyword_bot, surprise_bot
from .context import MEETINGS_PER_SIX_MONTHS, Context, quarter_steps
from .diff import Editions, diff_for
from .documents import Document, match_calendar
from .reader import (
    ABSOLUTE,
    CUT,
    DOVISH,
    HAWKISH,
    HIKE,
    HOLD,
    DiffVerdict,
    Reader,
    Reading,
    Verdict,
    signed_pair,
)
from .tape import HORIZONS, Bars, forward_returns
from .ticks import (
    DEFAULT_LATENCY_S,
    LATENCIES,
    TickTape,
    forward_returns_ticks,
    has_ticks,
)

INPUT_USD_PER_MTOK = 0.042


@dataclass(frozen=True)
class Signal:
    code: str
    ts: float
    title: str
    pair: str
    sign: int
    strength: float = 1.0


@dataclass
class Outcome:
    """One graded signal.

    The first five fields are what a bar tape can say. The rest only a tick tape
    can: what the mid did between publication and entry (``rush_bps``, signed by
    the side taken), what the book cost at entry (``spread_bps``), how late the
    entry was (``latency_s``), and the same forward return with no spread paid
    (``fwd_mid_bps``), which is the bar study's number and the honest comparison.
    """

    signal: Signal
    symbol: str
    pre_bps: float
    release_bar_bps: float
    fwd_bps: dict[int, float]  # signed by the signal's side, net of the half spread
    spread_bps: float = 0.0
    rush_bps: float = 0.0
    latency_s: float = 0.0
    entry_ts: float = 0.0
    fwd_mid_bps: dict[int, float] = field(default_factory=dict)


class Tape:
    """One fetch per symbol, shared by every arm and every null."""

    def __init__(self, store: Store, *, bar_min: int = 5, days: int = 60,
                 loader: Callable[..., Bars | None] | None = None) -> None:
        self.store = store
        self.bar_min = bar_min
        self.days = days
        self.loader = loader or _tape.bars
        self._cache: dict[str, Bars | None] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, symbol: str) -> Bars | None:
        with self._guard:
            if symbol in self._cache:
                return self._cache[symbol]
            lock = self._locks.setdefault(symbol, threading.Lock())
        # One loader per symbol at a time: the on-disk cache writes through a
        # temporary file named after the key, so two threads fetching the same
        # symbol would race each other for it.
        with lock:
            if symbol in self._cache:
                return self._cache[symbol]
            series = self.loader(self.store, symbol, bar_min=self.bar_min, days=self.days)
            with self._guard:
                self._cache[symbol] = series
            return series


class TickTapes:
    """One ``TickTape`` per symbol, shared by every arm, every null and every latency.

    The bar ``Tape`` above fetches one series per symbol and answers from memory.
    This one cannot: seventeen years of EURUSD ticks is a hundred million rows,
    so it fetches by the hour, on demand, and keeps a bounded window decoded.
    What both share is that a symbol is loaded once per run and every arm asks
    the same object, so the arms are graded on identical prices.
    """

    ticks = True

    def __init__(self, store: Store, *, workers: int = 8,
                 factory: Callable[..., TickTape] | None = None, **options: Any) -> None:
        self.store = store
        self.workers = workers
        self.factory = factory or TickTape
        self.options = options
        self._cache: dict[str, TickTape] = {}
        self._guard = threading.Lock()

    def get(self, symbol: str) -> TickTape | None:
        with self._guard:
            tape = self._cache.get(symbol)
            if tape is None:
                tape = self.factory(self.store, symbol, workers=self.workers, **self.options)
                self._cache[symbol] = tape
            return tape

    @property
    def fetched(self) -> int:
        with self._guard:
            return sum(t.fetched for t in self._cache.values())


def measure(
    tape: Tape | TickTapes,
    signals: Sequence[Signal],
    *,
    horizons: Iterable[int] = HORIZONS,
    workers: int = 8,
    latency_s: float = DEFAULT_LATENCY_S,
    pre_min: int = _tape.PRE_MIN,
) -> list[Outcome]:
    """Outcomes for the signals the tape can cover, on bars or on ticks.

    ``latency_s`` and ``pre_min`` are read only by the tick path; a bar tape has
    no notion of a one-second entry, which is the whole reason for the other one.
    """
    horizons = tuple(horizons)
    if getattr(tape, "ticks", False):
        return measure_ticks(tape, signals, horizons=horizons, workers=workers,
                             latency_s=latency_s, pre_min=pre_min)

    def one(signal: Signal) -> Outcome | None:
        series = tape.get(signal.pair)
        if series is None:
            return None
        result = forward_returns(series, signal.ts, signal.sign, horizons)
        if result is None:
            return None
        pre, bar, fwd = result
        return Outcome(signal=signal, symbol=series.symbol, pre_bps=pre,
                       release_bar_bps=bar, fwd_bps=fwd)

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        return [o for o in pool.map(one, signals) if o is not None]


def measure_ticks(
    tapes: TickTapes,
    signals: Sequence[Signal],
    *,
    horizons: Iterable[int] = HORIZONS,
    workers: int = 8,
    latency_s: float = DEFAULT_LATENCY_S,
    pre_min: int = _tape.PRE_MIN,
) -> list[Outcome]:
    """The same grading on ticks: pay the book at entry, exit at the mid."""
    horizons = tuple(horizons)

    def one(signal: Signal) -> Outcome | None:
        tape = tapes.get(signal.pair)
        if tape is None:
            return None
        result = forward_returns_ticks(
            tape, signal.ts, signal.sign, latency_s=latency_s,
            horizons_min=horizons, pre_min=pre_min,
        )
        if result is None:
            return None
        return Outcome(
            signal=signal, symbol=tape.symbol, pre_bps=result.pre_bps,
            release_bar_bps=0.0, fwd_bps=result.fwd_bps,
            spread_bps=result.spread_bps, rush_bps=result.rush_bps,
            latency_s=result.latency_s, entry_ts=result.entry_ts,
            fwd_mid_bps=result.fwd_mid_bps,
        )

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        return [o for o in pool.map(one, signals) if o is not None]


def null_signals(
    tape: Tape | TickTapes,
    outcomes: Sequence[Outcome],
    *,
    per: int = 2,
    span_days: float = 5.0,
    exclude_h: float = 3.0,
    horizons: Iterable[int] = HORIZONS,
    seed: int = 0,
    tries: int = 40,
    latency_s: float = DEFAULT_LATENCY_S,
    pre_min: int = _tape.PRE_MIN,
    workers: int = 8,
) -> list[Signal]:
    """The same pairs and sides at random nearby moments the tape covers.

    Unlike the crypto study, a random moment near an FX event is very often the
    weekend, so the draw is repeated until it lands somewhere measurable. The
    controls are therefore matched on *session* as well as on pair and side, and
    on ticks they are measured at the same latency and pay the same spread.
    """
    horizons = tuple(horizons)
    if getattr(tape, "ticks", False):
        return _tick_nulls(tape, outcomes, per=per, span_days=span_days,
                           exclude_h=exclude_h, horizons=horizons, seed=seed,
                           tries=tries, latency_s=latency_s, pre_min=pre_min,
                           workers=workers)
    rng = random.Random(seed)
    out: list[Signal] = []
    for outcome in outcomes:
        source = outcome.signal
        series = tape.get(source.pair)
        if series is None:
            continue
        made = 0
        for _ in range(per * tries):
            if made >= per:
                break
            offset = rng.uniform(-span_days * 86400, span_days * 86400)
            if abs(offset) < exclude_h * 3600:
                continue
            when = source.ts + offset
            if forward_returns(series, when, 1, horizons) is None:
                continue
            out.append(Signal(code=f"null:{source.code}", ts=when, title="",
                              pair=source.pair, sign=source.sign, strength=source.strength))
            made += 1
    return out


def _tick_nulls(
    tapes: TickTapes,
    outcomes: Sequence[Outcome],
    *,
    per: int,
    span_days: float,
    exclude_h: float,
    horizons: tuple[int, ...],
    seed: int,
    tries: int,
    latency_s: float,
    pre_min: int,
    workers: int,
) -> list[Signal]:
    """The tick null, drawn per outcome so the draws can run in parallel.

    Each outcome gets its own seeded stream rather than sharing one, because a
    shared stream makes the controls depend on the order the threads ran in --
    and a null that is not reproducible is not a control. Probing a random
    moment costs network, and a third of the draws land in a closed weekend, so
    the outer loop is where the parallelism has to be.
    """

    def draws(item: tuple[int, Outcome]) -> list[Signal]:
        index, outcome = item
        source = outcome.signal
        tape = tapes.get(source.pair)
        if tape is None:
            return []
        rng = random.Random(f"{seed}:{index}:{source.code}:{source.ts}")
        found: list[Signal] = []
        for _ in range(per * tries):
            if len(found) >= per:
                break
            offset = rng.uniform(-span_days * 86400, span_days * 86400)
            if abs(offset) < exclude_h * 3600:
                continue
            when = source.ts + offset
            if not has_ticks(tape, when, latency_s=latency_s,
                             horizons_min=horizons, pre_min=pre_min):
                continue
            found.append(Signal(code=f"null:{source.code}", ts=when, title="",
                                pair=source.pair, sign=source.sign,
                                strength=source.strength))
        return found

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        return [s for group in pool.map(draws, enumerate(outcomes)) for s in group]


# ------------------------------------------------------------------ arms


def bot_signals(
    documents: Sequence[Document],
    bot: Callable[[Document], list[BotSignal]],
) -> list[Signal]:
    out: list[Signal] = []
    for document in documents:
        for hit in bot(document):
            out.append(Signal(document.id, document.ts, document.title, hit.pair, hit.sign))
    return out


def surprise_signals(
    documents: Sequence[Document], rows: Sequence[dict[str, Any]]
) -> list[Signal]:
    out: list[Signal] = []
    for document in documents:
        for hit in surprise_bot(document, rows):
            out.append(Signal(document.id, document.ts, document.title, hit.pair, hit.sign))
    return out


def all_text_signals(documents: Sequence[Document]) -> list[Signal]:
    """Every document, one signal, so the base rate for "text happened" is visible."""
    out: list[Signal] = []
    for document in documents:
        hits = keyword_bot(document)
        if hits:
            out.append(Signal(document.id, document.ts, document.title,
                              hits[0].pair, hits[0].sign))
            continue
        entry = signed_pair(document.currency, "hawkish")
        if entry is not None:  # the words tied: take the long side, by convention
            out.append(Signal(document.id, document.ts, document.title, entry[0], entry[1]))
    return out


FOMC_STATEMENT = re.compile(r"fomc statement", re.I)


def statements(documents: Sequence[Document]) -> list[Document]:
    """The FOMC statements in a collection, by title, oldest first.

    This is the same filter the 2009-2026 run used to split its results, so the
    count (150 over the archive) is comparable. It keeps a handful of documents
    that are not rate decisions -- swap-line announcements, the longer-run goals
    statement -- because the title is what the filter has and inventing a
    second, unstated rule would make the two runs incomparable.
    """
    return sorted(
        [d for d in documents if FOMC_STATEMENT.search(d.title or "")], key=lambda d: d.ts
    )


def decision_from_rates(document: Document, previous: Document | None) -> str | None:
    """hike / hold / cut, from the rate the code parsed out of each statement.

    ``None`` when either statement's target range could not be parsed, which is
    an answer and not a zero: a statement whose rate is unreadable must not be
    counted as a hold.
    """
    if previous is None:
        return None
    now = announced_rate(f"{document.title}\n{document.body}")
    before = announced_rate(f"{previous.title}\n{previous.body}")
    if now is None or before is None:
        return None
    if abs(now - before) < 1e-9:
        return HOLD
    return HIKE if now > before else CUT


def bill_surprise_signals(
    documents: Sequence[Document],
    contexts: dict[str, Context],
    *,
    previous_of_id: dict[str, Document] | None = None,
) -> list[Signal]:
    """The decision against the one the six-month bill implied.

    **This is a crude proxy.** The instrument that actually prices a meeting is
    the fed funds future, and that is not free. What is free is the six-month
    bill, which prices the average funds rate over six months; the spread over
    the effective rate is therefore the whole *path*, not this meeting, and the
    path is divided by the four meetings six months holds to get a per-meeting
    expectation. That division is an assumption and the arm inherits its error.
    A surprise inside one basis point is no signal.
    """
    previous_of_id = previous_of_id or {}
    out: list[Signal] = []
    for document in documents:
        context = contexts.get(document.id)
        if context is None or context.rates is None:
            continue
        bill = context.rates.values.get("6m")
        effective = context.rates.values.get("ff")
        previous = previous_of_id.get(document.id) or context.previous
        if bill is None or effective is None or previous is None:
            continue
        now = announced_rate(f"{document.title}\n{document.body}")
        before = announced_rate(f"{previous.title}\n{previous.body}")
        if now is None or before is None:
            continue
        expected = (bill - effective) / MEETINGS_PER_SIX_MONTHS
        gap = (now - before) - expected
        if abs(gap) < 0.01:
            continue
        signed = signed_pair(document.currency, HAWKISH if gap > 0 else DOVISH)
        if signed is None:
            continue
        out.append(Signal(document.id, document.ts, document.title, signed[0], signed[1],
                          min(1.0, abs(gap) / 0.25)))
    return out


def dots_surprise_signals(
    documents: Sequence[Document], contexts: dict[str, Context]
) -> list[Signal]:
    """The change in next year's median dot against the previous SEP.

    Projection meetings only, and only the ones whose table carries a funds-rate
    *median*: the SEP has published one since September 2015 and printed ranges
    and a histogram before that, so the arm is dark for the first third of the
    archive. No signal when the median did not move, or when the previous SEP
    had no cell for that year. A higher median is fewer cuts, which is hawkish
    for the dollar.
    """
    out: list[Signal] = []
    for document in documents:
        context = contexts.get(document.id)
        if context is None or context.projections is None:
            continue
        dots = context.projections
        # The current year's column is nearly over by December and the longer
        # run is not a path; next year is the one the market trades.
        index = 1 if len(dots.years) > 2 else 0
        if index >= len(dots.medians) or index >= len(dots.previous):
            continue
        was = dots.previous[index]
        if was is None:
            continue
        gap = dots.medians[index] - was
        if abs(gap) < 1e-9:
            continue
        signed = signed_pair(document.currency, HAWKISH if gap > 0 else DOVISH)
        if signed is None:
            continue
        out.append(Signal(document.id, document.ts, document.title, signed[0], signed[1],
                          min(1.0, abs(quarter_steps(gap)) / 4.0)))
    return out


def reader_signals(
    readings: Sequence[Reading], threshold: float, *, min_confidence: float = 0.5
) -> list[Signal]:
    out: list[Signal] = []
    for reading in readings:
        for verdict in reading.signals(threshold, min_confidence=min_confidence):
            out.append(Signal(reading.document.id, reading.document.ts,
                              reading.document.title, verdict.pair, verdict.sign,
                              verdict.strength))
    return out


# ------------------------------------------------------------------ summary


@dataclass
class Stat:
    n: int
    mean: float
    se: float


def _stat(values: list[float]) -> Stat:
    if not values:
        return Stat(0, 0.0, 0.0)
    if len(values) == 1:
        return Stat(1, values[0], 0.0)
    return Stat(len(values), statistics.fmean(values),
                statistics.stdev(values) / math.sqrt(len(values)))


@dataclass
class ArmSummary:
    """What an arm did. ``fwd`` is net of the half spread wherever there is one.

    ``spread`` and ``rush`` are zero on the bar path, which has neither a book
    nor a latency; on ticks they are the two numbers that decide whether reading
    in a second is worth anything -- what the entry cost, and how much of the
    move had already happened by the time the reader could act.
    """

    name: str
    signals: int
    measured: int
    pre: Stat
    release_bar: Stat
    fwd: dict[int, Stat]
    hit: dict[int, float]
    null: dict[int, Stat]
    z: dict[int, float] = field(default_factory=dict)
    note: str = ""
    spread: Stat = field(default_factory=lambda: Stat(0, 0.0, 0.0))
    rush: Stat = field(default_factory=lambda: Stat(0, 0.0, 0.0))
    fwd_mid: dict[int, Stat] = field(default_factory=dict)
    latency_s: float = 0.0

    @property
    def measurable(self) -> float:
        return self.measured / self.signals if self.signals else 0.0


def summarize(
    name: str,
    signals: Sequence[Signal],
    outcomes: Sequence[Outcome],
    nulls: Sequence[Outcome],
    *,
    horizons: Iterable[int] = HORIZONS,
    note: str = "",
    latency_s: float = 0.0,
) -> ArmSummary:
    horizons = tuple(horizons)
    fwd = {h: _stat([o.fwd_bps[h] for o in outcomes]) for h in horizons}
    null = {h: _stat([o.fwd_bps[h] for o in nulls]) for h in horizons}
    fwd_mid = {
        h: _stat([o.fwd_mid_bps[h] for o in outcomes if h in o.fwd_mid_bps])
        for h in horizons
    }
    hit: dict[int, float] = {}
    for h in horizons:
        signed = [o.fwd_bps[h] for o in outcomes if o.fwd_bps[h] != 0.0]
        hit[h] = sum(v > 0 for v in signed) / len(signed) if signed else 0.0
    z: dict[int, float] = {}
    for h in horizons:
        spread = math.sqrt(fwd[h].se**2 + null[h].se**2)
        z[h] = (fwd[h].mean - null[h].mean) / spread if spread > 0 else 0.0
    return ArmSummary(
        name=name, signals=len(signals), measured=len(outcomes),
        pre=_stat([o.pre_bps * o.signal.sign for o in outcomes]),
        release_bar=_stat([o.release_bar_bps * o.signal.sign for o in outcomes]),
        fwd=fwd, hit=hit, null=null, z=z, note=note,
        spread=_stat([o.spread_bps for o in outcomes]),
        rush=_stat([o.rush_bps for o in outcomes]),
        fwd_mid=fwd_mid, latency_s=latency_s,
    )


def latency_sweep(
    tapes: TickTapes,
    signals: Sequence[Signal],
    *,
    latencies: Iterable[float] = LATENCIES,
    horizons: Iterable[int] = (15, 60),
    per: int = 2,
    workers: int = 8,
    seed: int = 0,
    pre_min: int = _tape.PRE_MIN,
) -> list[ArmSummary]:
    """The same signals entered later and later -- the scalping question, one row each.

    Latency zero is the counterfactual nobody has: the trade is filled on the
    first tick after the timestamp itself. One second is a reader that answered
    and hit the button. Two minutes is a human who read the statement. If the
    rows are flat the speed is not what was being paid for; if they decay, the
    decay rate is the price of being slow, in basis points.
    """
    horizons = tuple(horizons)
    out: list[ArmSummary] = []
    for latency in latencies:
        outcomes = measure(tapes, signals, horizons=horizons, workers=workers,
                           latency_s=latency, pre_min=pre_min)
        nulls = measure(
            tapes,
            null_signals(tapes, outcomes, per=per, horizons=horizons, seed=seed,
                         latency_s=latency, pre_min=pre_min, workers=workers),
            horizons=horizons, workers=workers, latency_s=latency, pre_min=pre_min,
        )
        out.append(summarize(f"{latency:g}s", signals, outcomes, nulls,
                             horizons=horizons, latency_s=latency))
    return out


# ------------------------------------------------------------------ reading


def state_hash(
    document: Document,
    changes: Sequence[tuple[str, str]],
    calendar: dict[str, Any] | None,
    *,
    body_chars: int = 6000,
    context: str = "",
    extra: str = "",
) -> str:
    """A short digest of everything round one will see besides the tree itself.

    Keying a cached reading on the document id alone is wrong the moment the
    *input* changes without the document changing: a statement read in a 60-day
    window has no previous edition, and the same statement read in a 17-year
    window has one, twelve changed sentences, and a different answer. The id and
    the tree version cannot see that, so the state does.
    """
    parts: list[Any] = [
        document.title,
        (document.body or "")[:body_chars],
        [list(pair) for pair in changes],
        None if not calendar else [
            calendar.get(k, "") for k in ("title", "impact", "forecast", "previous")
        ],
    ]
    # Appended only when there is one, so every ``v1`` digest ever written stays
    # exactly what it was and the absolute reader's cache is still valid.
    if context:
        parts.append(hashlib.sha1(context.encode("utf-8")).hexdigest()[:10])
    # Anything else round one will see -- today, the press-conference block.
    # Appended only when there is one, for the same reason as ``context``.
    if extra:
        parts.append(hashlib.sha1(extra.encode("utf-8")).hexdigest()[:10])
    payload = json.dumps(parts, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def read_all(
    documents: Sequence[Document],
    make_reader: Callable[[], Reader],
    *,
    store: Store | None = None,
    cache_tag: str = "",
    calendar: Sequence[dict[str, Any]] = (),
    max_diffs: int = 12,
    workers: int = 1,
    body_chars: int = 6000,
    contexts: dict[str, str] | None = None,
    pressers: dict[str, dict[str, Any]] | None = None,
    progress: Callable[[int, int, int, float], None] | None = None,
    progress_every: int = 100,
) -> list[Reading]:
    """Read every document, one reader per worker (the client mutates its map).

    The diff against the previous edition is computed here, from the same
    document list, so a statement read in isolation and one read inside a run
    see the same changed sentences -- and it is computed *before* the cache is
    consulted, because it is part of what the cache key means.

    ``contexts`` maps a document id to the pre-release context block the reader
    should be handed, and is part of the cache key: two readings of the same
    statement against different context are different readings. ``pressers``
    does the same for the press conference that followed it, and is in the key
    for the same reason.

    ``progress`` is called with ``(done, total, went to round two, dollars so
    far)`` every ``progress_every`` documents, because a 2,300-document run is
    watched from a log and a silent hour is indistinguishable from a hang.
    """
    local = threading.local()
    corpus = list(documents)
    editions = Editions(corpus)
    rows = list(calendar)
    counter = {"done": 0, "round_two": 0, "tokens": 0}
    guard = threading.Lock()

    def reader() -> Reader:
        if not hasattr(local, "reader"):
            local.reader = make_reader()
        return local.reader

    def one(document: Document) -> Reading:
        previous, changes = diff_for(document, editions, limit=max_diffs)
        row = match_calendar(document, rows) if rows else None
        context = (contexts or {}).get(document.id, "")
        presser = (pressers or {}).get(document.id)
        extra = json.dumps(presser, sort_keys=True, ensure_ascii=False) if presser else ""
        digest = state_hash(document, changes, row, body_chars=body_chars,
                            context=context, extra=extra)
        key = f"reading:{cache_tag}:{document.id}:{digest}"
        result: Reading | None = None
        if store is not None:
            found, data = store.get(key)
            if found:
                result = reading_from_dict(document, data)
        if result is None:
            result = reader().read(document, previous=previous, changes=changes,
                                   calendar=row, context=context, presser=presser)
            if store is not None:
                store.put(key, reading_to_dict(result))
        with guard:
            counter["done"] += 1
            counter["round_two"] += 1 if result.rounds == 2 else 0
            counter["tokens"] += result.input_tokens
            done = counter["done"]
            if progress is not None and (done % progress_every == 0 or done == len(corpus)):
                progress(done, len(corpus), counter["round_two"],
                         counter["tokens"] * INPUT_USD_PER_MTOK / 1e6)
        return result

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(one, corpus))


def reading_to_dict(reading: Reading) -> dict[str, Any]:
    verdict = reading.verdict
    return {
        "kind": reading.kind, "p_kind": reading.p_kind,
        "policy_relevant": reading.policy_relevant, "stance": reading.stance,
        "p_stance": reading.p_stance, "confidence": reading.confidence,
        "new_information": reading.new_information, "magnitude": reading.magnitude,
        "guidance_changed": reading.guidance_changed, "surprise": reading.surprise,
        "intervention": reading.intervention, "confirm": reading.confirm,
        "horizon": reading.horizon, "rounds": reading.rounds,
        "latency_ms": reading.latency_ms, "wall_ms": reading.wall_ms,
        "input_tokens": reading.input_tokens, "questions_asked": reading.questions_asked,
        "mode": reading.mode, "expected_action": reading.expected_action,
        "p_expected": reading.p_expected, "actual_action": reading.actual_action,
        "p_actual": reading.p_actual, "relative": reading.relative,
        "p_relative": reading.p_relative, "surprise_channel": reading.surprise_channel,
        "p_channel": reading.p_channel, "surprise_size": reading.surprise_size,
        "versus_minutes": reading.versus_minutes, "context_chars": reading.context_chars,
        "presser_stance": reading.presser_stance, "p_presser": reading.p_presser,
        "remarks_vs_statement": reading.remarks_vs_statement,
        "qa_vs_remarks": reading.qa_vs_remarks, "pushback": reading.pushback,
        "dominant_topic": reading.dominant_topic,
        "presser_chars": reading.presser_chars,
        "diffs": [
            {"index": d.index, "was": d.was, "now": d.now, "stance": d.stance,
             "p_stance": d.p_stance, "material": d.material}
            for d in reading.diffs
        ],
        "verdict": None if verdict is None else {
            "pair": verdict.pair, "sign": verdict.sign, "p_side": verdict.p_side,
            "confidence": verdict.confidence, "strength": verdict.strength,
        },
    }


def reading_from_dict(document: Document, data: dict[str, Any]) -> Reading:
    verdict = data.get("verdict")
    return Reading(
        document=document, kind=data["kind"], p_kind=data["p_kind"],
        policy_relevant=data["policy_relevant"], stance=data["stance"],
        p_stance=data["p_stance"], confidence=data["confidence"],
        new_information=data["new_information"], magnitude=data["magnitude"],
        guidance_changed=data["guidance_changed"], surprise=data["surprise"],
        intervention=data["intervention"], confirm=data["confirm"],
        horizon=data["horizon"],
        diffs=[DiffVerdict(**d) for d in data["diffs"]],
        verdict=None if verdict is None else Verdict(**verdict),
        rounds=data["rounds"], latency_ms=data["latency_ms"], wall_ms=data["wall_ms"],
        input_tokens=data["input_tokens"], questions_asked=data.get("questions_asked", 0),
        # ``.get`` throughout: a reading cached before the context tree existed
        # is still a valid absolute reading and must keep loading.
        mode=data.get("mode", ABSOLUTE),
        expected_action=data.get("expected_action", ""), p_expected=data.get("p_expected", 0.0),
        actual_action=data.get("actual_action", ""), p_actual=data.get("p_actual", 0.0),
        relative=data.get("relative", ""), p_relative=data.get("p_relative", 0.0),
        surprise_channel=data.get("surprise_channel", ""), p_channel=data.get("p_channel", 0.0),
        surprise_size=data.get("surprise_size", 0.0),
        versus_minutes=data.get("versus_minutes", ""),
        context_chars=data.get("context_chars", 0),
        presser_stance=data.get("presser_stance", ""), p_presser=data.get("p_presser", 0.0),
        remarks_vs_statement=data.get("remarks_vs_statement", ""),
        qa_vs_remarks=data.get("qa_vs_remarks", ""), pushback=data.get("pushback", 0.0),
        dominant_topic=data.get("dominant_topic", ""),
        presser_chars=data.get("presser_chars", 0),
    )


def cost_usd(readings: Sequence[Reading]) -> float:
    return sum(r.input_tokens for r in readings) * INPUT_USD_PER_MTOK / 1e6


# ------------------------------------------------------------------ audit


@dataclass
class Disagreement:
    title: str
    when: float
    issuer: str
    bot: list[tuple[str, int]]
    reader: list[tuple[str, int]]
    outcomes: dict[str, float]  # pair -> unsigned forward return at one horizon, bps


@dataclass
class Confusion:
    """How often the model's reading of the decision matches the parsed rate.

    ``actual`` is the interesting one: it is a fact, not a judgment, so a model
    that gets it wrong cannot be trusted on the judgment that sits on top of it.
    ``expected`` is not a fact -- nothing here knows what the market expected --
    so its agreement with the decision is reported as what it is: how often the
    model thought the market had already priced what happened.
    """

    rows: dict[tuple[str, str], int] = field(default_factory=dict)  # (parsed, model) -> n
    unparsed: int = 0

    @property
    def n(self) -> int:
        return sum(self.rows.values())

    @property
    def agreed(self) -> int:
        return sum(v for (parsed, model), v in self.rows.items() if parsed == model)

    @property
    def rate(self) -> float:
        return self.agreed / self.n if self.n else 0.0


def action_confusion(
    readings: Sequence[Reading],
    previous_of_id: dict[str, Document],
    *,
    field_name: str = "actual_action",
) -> Confusion:
    """Cross-check one of the model's action answers against the parsed rate."""
    out = Confusion()
    for reading in readings:
        answer = getattr(reading, field_name, "")
        if not answer:
            continue
        parsed = decision_from_rates(reading.document, previous_of_id.get(reading.document.id))
        if parsed is None:
            out.unparsed += 1
            continue
        key = (parsed, answer)
        out.rows[key] = out.rows.get(key, 0) + 1
    return out


def channel_counts(readings: Sequence[Reading]) -> dict[str, int]:
    """How the reader distributed the surprise across the six channels."""
    out: dict[str, int] = {}
    for reading in readings:
        if reading.surprise_channel:
            out[reading.surprise_channel] = out.get(reading.surprise_channel, 0) + 1
    return out


def signal_disagreements(
    left: Sequence[Signal],
    right: Sequence[Signal],
    outcomes: Sequence[Outcome],
    *,
    horizon: int = 15,
) -> list[Disagreement]:
    """Documents two arms traded differently, with what the tape then did.

    The tape's verdict is unsigned by either arm -- it is the pair's own move at
    the horizon -- so a reader of the list can see which side was right without
    the table having already decided.
    """
    by_key: dict[tuple[str, str], float] = {}
    for outcome in outcomes:
        if horizon in outcome.fwd_bps:
            by_key[(outcome.signal.code, outcome.signal.pair)] = (
                outcome.fwd_bps[horizon] * outcome.signal.sign
            )
    index: dict[str, Signal] = {s.code: s for s in list(left) + list(right)}
    per_side: list[dict[str, list[tuple[str, int]]]] = []
    for group in (left, right):
        rows: dict[str, list[tuple[str, int]]] = {}
        for signal in group:
            rows.setdefault(signal.code, []).append((signal.pair, signal.sign))
        per_side.append(rows)
    out: list[Disagreement] = []
    for code in sorted(index, key=lambda c: index[c].ts):
        a = sorted(per_side[0].get(code, []))
        b = sorted(per_side[1].get(code, []))
        if a == b:
            continue
        signal = index[code]
        pairs = {p for p, _ in a} | {p for p, _ in b}
        seen = {p: by_key[(code, p)] for p in pairs if (code, p) in by_key}
        out.append(Disagreement(signal.title, signal.ts, code.split(":", 1)[0], a, b, seen))
    return out


def disagreements(
    readings: Sequence[Reading],
    threshold: float,
    outcomes: Sequence[Outcome],
    *,
    horizon: int = 15,
) -> list[Disagreement]:
    """Documents where counting words and reading them traded differently."""
    by_key: dict[tuple[str, str], float] = {}
    for outcome in outcomes:
        if horizon in outcome.fwd_bps:
            by_key[(outcome.signal.code, outcome.signal.pair)] = (
                outcome.fwd_bps[horizon] * outcome.signal.sign
            )
    out: list[Disagreement] = []
    for reading in readings:
        document = reading.document
        bot = sorted((s.pair, s.sign) for s in keyword_bot(document))
        mine = sorted((v.pair, v.sign) for v in reading.signals(threshold))
        if bot == mine:
            continue
        pairs = {p for p, _ in bot} | {p for p, _ in mine}
        seen = {p: by_key[(document.id, p)] for p in pairs if (document.id, p) in by_key}
        out.append(Disagreement(document.title, document.ts, document.issuer, bot, mine, seen))
    return out


__all__ = [
    "ArmSummary", "Confusion", "DEFAULT_LATENCY_S", "Disagreement", "LATENCIES",
    "Outcome", "Signal", "Stat", "Tape", "TickTape", "TickTapes", "action_confusion",
    "all_text_signals", "bill_surprise_signals", "bot_signals", "channel_counts",
    "cost_usd", "decision_from_rates", "disagreements", "dots_surprise_signals",
    "keyword_bot", "latency_sweep",
    "measure", "measure_ticks", "null_signals", "read_all", "reader_signals",
    "reading_from_dict", "reading_to_dict", "signal_disagreements", "state_hash",
    "statements", "summarize", "surprise_bot", "surprise_signals",
]
