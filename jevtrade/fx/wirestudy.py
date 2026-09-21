"""Score every arm on the wire, with minute candles as the judge.

An arm turns a wire post into signals: (pair, long or short). Each one enters at
the open of the first minute bar at or after the post plus a latency, pays the
ask to go long and the bid to go short, and is measured by signed log return at
fixed horizons. The null is the same pair and the same side at random moments
within five days, drawn *only where the tape has bars* -- spot FX is shut from
about Friday 21:00 to Sunday 21:00 UTC, so a third of naive draws land in a
hole and a "+60m" return across a Friday close would be a 51-hour return.

Four arms:

* ``keyword-bot``   the incumbent: ``wire.py``'s lexicon names a currency and
                    counts hawkish words against dovish ones. It cannot read.
* ``wire-sample``   a seeded random sample of the corpus, taking the keyword
                    sign where it has one. This is the base rate -- "a post
                    happened" -- without paying to grade all twenty-one thousand.
* ``reader >= thr`` the ``w1`` tree at a strength threshold.
* ``null``          per arm, as above.

**The breakdowns are the deliverable, not the headline number.** One mean over
twenty-one thousand posts of every kind is a number about the wire's mixture,
not about anything tradeable. What the study is for is the split: which
*category* of post moves the tape, whether the scheduled ones behave differently
from the unscheduled, whether a number behaves differently from words, which
currency, which session -- and, in every one of those cells, the ``pre`` column,
which is how far the pair had already moved in the fifteen minutes before the
wire posted. That column is the wire's lateness, measured rather than argued
about, and it is the reason a positive cell is not automatically an edge.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import random
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..listing.store import Store
from . import candles as C
from . import wire as W
from .reader import Verdict, WireReader, WireReading, wire_signed_pair
from .study import INPUT_USD_PER_MTOK, Outcome, Signal, Stat, _stat

HORIZONS = (1, 5, 15, 30, 60)
# The three the breakdown tables print: the first leg, the quarter hour the
# central-bank study found its edge in, and the hour.
REPORT_HORIZONS = (5, 15, 60)
DEFAULT_SAMPLE = 3000


# ------------------------------------------------------------------ measuring


def measure(
    tapes: C.MinuteTapes,
    signals: Sequence[Signal],
    *,
    horizons: Iterable[int] = HORIZONS,
    workers: int = 4,
    latency_s: float = C.DEFAULT_LATENCY_S,
    pre_min: int = C.PRE_MIN,
) -> list[Outcome]:
    """Outcomes for the signals the minute tape can cover.

    ``rush`` is not computed and stays zero: on minute bars the move between the
    post and the entry is one bar's open-to-open at most, which is not the
    tick study's "how much of it went before you could act" and would read as
    though it were. The ``pre`` column answers that question here instead.
    """
    horizons = tuple(horizons)

    def one(signal: Signal) -> Outcome | None:
        tape = tapes.get(signal.pair)
        if tape is None:
            return None
        result = C.forward_returns_minutes(
            tape, signal.ts, signal.sign, latency_s=latency_s,
            horizons_min=horizons, pre_min=pre_min,
        )
        if result is None:
            return None
        return Outcome(
            signal=signal, symbol=tape.symbol, pre_bps=result.pre_bps,
            release_bar_bps=0.0, fwd_bps=result.fwd_bps, spread_bps=result.spread_bps,
            rush_bps=0.0, latency_s=result.latency_s, entry_ts=result.entry_ts,
            fwd_mid_bps=result.fwd_mid_bps,
        )

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        return [o for o in pool.map(one, signals) if o is not None]


def null_signals(
    tapes: C.MinuteTapes,
    outcomes: Sequence[Outcome],
    *,
    per: int = 1,
    span_days: float = 5.0,
    exclude_h: float = 3.0,
    horizons: Iterable[int] = HORIZONS,
    seed: int = 0,
    tries: int = 40,
    latency_s: float = C.DEFAULT_LATENCY_S,
    pre_min: int = C.PRE_MIN,
    workers: int = 4,
) -> list[Signal]:
    """The same pairs and sides at random nearby moments the tape covers.

    One per outcome rather than the two the central-bank study drew: there are
    twenty thousand outcomes here, not two thousand, and a control that costs a
    day-file each is the expensive half of the run. Each outcome gets its own
    seeded stream rather than sharing one, because a shared stream makes the
    controls depend on the order the threads ran in -- and a null that is not
    reproducible is not a control.
    """
    horizons = tuple(horizons)

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
            if not C.has_bars(tape, when, latency_s=latency_s,
                              horizons_min=horizons, pre_min=pre_min):
                continue
            found.append(Signal(code=f"null:{source.code}", ts=when, title="",
                                pair=source.pair, sign=source.sign,
                                strength=source.strength))
        return found

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        return [s for group in pool.map(draws, enumerate(outcomes)) for s in group]


# ---------------------------------------------------------------------- arms


def keyword_signals(articles: Sequence[W.Article]) -> list[Signal]:
    """The incumbent: a currency is named, and the words set the sign.

    No currency, no direction, or a currency with no pair in the judge (CNY,
    anything else) means no trade -- the bot abstains rather than guessing, the
    same as the reader does, so the two are comparable on the same corpus.
    """
    out: list[Signal] = []
    for article in articles:
        text = f"{article.headline}\n{' '.join(article.keywords)}\n{article.body}"
        if not W.about_fx_of(text):
            continue
        signed = wire_signed_pair(W.currency_of(text), W.direction_of(text))
        if signed is None:
            continue
        out.append(Signal(article.id, article.published_ts, article.headline,
                          signed[0], signed[1]))
    return out


def sample_signals(
    articles: Sequence[W.Article], size: int = DEFAULT_SAMPLE, *, seed: int = 0
) -> list[Signal]:
    """A seeded random sample of the corpus, for the base rate.

    "Does a wire post move the tape at all" needs the whole corpus and the
    reader's answers are not needed for it, but grading twenty-one thousand
    posts on seven pairs is the expensive half of the run. A sample of three
    thousand puts a standard error on the base rate that is small next to any
    edge worth having, at a seventh of the cost, and being seeded it is the same
    three thousand on every re-run.

    It takes the keyword sign where the bot has one so the arm is directional
    rather than a coin flip; where the bot abstains, the sample does too, which
    is why its count comes in under ``size``.
    """
    rng = random.Random(f"wire-sample:{seed}:{size}")
    pool = list(articles)
    chosen = pool if len(pool) <= size else rng.sample(pool, size)
    chosen.sort(key=lambda a: a.published_ts)
    return [Signal(f"sample:{s.code}", s.ts, s.title, s.pair, s.sign)
            for s in keyword_signals(chosen)]


def reader_signals(
    readings: Sequence[WireReading], threshold: float, *, min_confidence: float = 0.5
) -> list[Signal]:
    out: list[Signal] = []
    for reading in readings:
        for verdict in reading.signals(threshold, min_confidence=min_confidence):
            out.append(Signal(reading.id, reading.ts, reading.article.headline,
                              verdict.pair, verdict.sign, verdict.strength))
    return out


# ------------------------------------------------------------------- reading


def state_hash(article: W.Article, *, body_chars: int) -> str:
    """A digest of everything round one will see besides the tree itself.

    The body cap is in it: the same post read with 3,000 characters and with
    8,000 is a different question, and a cache keyed on the URL alone would
    hand back the wrong answer when ``--body-chars`` changes.
    """
    payload = json.dumps([
        article.headline, article.section, list(article.keywords)[:20],
        (article.body or "")[:body_chars], body_chars,
    ], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def reading_to_dict(reading: WireReading) -> dict[str, Any]:
    verdict = reading.verdict
    return {
        "about_fx": reading.about_fx, "currency": reading.currency,
        "p_currency": reading.p_currency, "direction": reading.direction,
        "p_direction": reading.p_direction, "confidence": reading.confidence,
        "category": reading.category, "p_category": reading.p_category,
        "magnitude": reading.magnitude, "new_information": reading.new_information,
        "scheduled": reading.scheduled, "already_moved": reading.already_moved,
        "is_number": reading.is_number, "confirm": reading.confirm,
        "horizon": reading.horizon, "rounds": reading.rounds,
        "latency_ms": reading.latency_ms, "wall_ms": reading.wall_ms,
        "input_tokens": reading.input_tokens, "questions_asked": reading.questions_asked,
        "verdict": None if verdict is None else {
            "pair": verdict.pair, "sign": verdict.sign, "p_side": verdict.p_side,
            "confidence": verdict.confidence, "strength": verdict.strength,
        },
    }


def reading_from_dict(article: W.Article, data: dict[str, Any]) -> WireReading:
    verdict = data.get("verdict")
    return WireReading(
        article=article, about_fx=data["about_fx"], currency=data["currency"],
        p_currency=data["p_currency"], direction=data["direction"],
        p_direction=data["p_direction"], confidence=data["confidence"],
        category=data["category"], p_category=data["p_category"],
        magnitude=data["magnitude"], new_information=data["new_information"],
        scheduled=data["scheduled"], already_moved=data["already_moved"],
        is_number=data["is_number"], confirm=data["confirm"],
        horizon=data["horizon"],
        verdict=None if verdict is None else Verdict(**verdict),
        rounds=data["rounds"], latency_ms=data["latency_ms"], wall_ms=data["wall_ms"],
        input_tokens=data["input_tokens"],
        questions_asked=data.get("questions_asked", 0),
    )


def read_all(
    articles: Sequence[W.Article],
    make_reader: Callable[[], WireReader],
    *,
    store: Store | None = None,
    cache_tag: str = "",
    workers: int = 1,
    body_chars: int = W.STATE_BODY_CHARS,
    progress: Callable[[int, int, int, float], None] | None = None,
    progress_every: int = 500,
) -> list[WireReading]:
    """Read every post, one reader per worker (the client mutates its map).

    ``progress`` is called with ``(done, total, went to round two, dollars so
    far)`` every ``progress_every`` posts, because a twenty-thousand-post run is
    watched from a log and a silent hour is indistinguishable from a hang.
    """
    local = threading.local()
    corpus = list(articles)
    counter = {"done": 0, "round_two": 0, "tokens": 0}
    guard = threading.Lock()

    def reader() -> WireReader:
        if not hasattr(local, "reader"):
            local.reader = make_reader()
        return local.reader

    def one(article: W.Article) -> WireReading:
        digest = state_hash(article, body_chars=body_chars)
        key = f"wire-reading:{cache_tag}:{article.id}:{digest}"
        result: WireReading | None = None
        if store is not None:
            found, data = store.get(key)
            if found and isinstance(data, dict):
                result = reading_from_dict(article, data)
        if result is None:
            result = reader().read(article, local=W.local_times(article.published_ts))
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


def cost_usd(readings: Sequence[WireReading]) -> float:
    return sum(r.input_tokens for r in readings) * INPUT_USD_PER_MTOK / 1e6


# ------------------------------------------------------------------ breakdown


@dataclass
class Cell:
    """One row of a breakdown table: n, hit rate, mean and s.e. per horizon, and pre.

    ``pre`` is the mean of the fifteen minutes *before* the post, signed by the
    side the arm took. A large positive ``pre`` next to a small forward return
    is the wire being late: the move the post describes had already happened.
    """

    key: str
    n: int
    fwd: dict[int, Stat] = field(default_factory=dict)
    hit: dict[int, float] = field(default_factory=dict)
    pre: Stat = field(default_factory=lambda: Stat(0, 0.0, 0.0))

    def row(self, horizons: Sequence[int]) -> str:
        cells = "".join(
            f"{self.fwd[h].mean:>+7.0f}{'+-' + format(self.fwd[h].se, '.0f'):>7}"
            for h in horizons if h in self.fwd)
        return (f"{self.key[:26]:<27}{self.n:>7}{self.pre.mean:>+7.0f}"
                f"{self.hit.get(15, 0.0):>7.0%}{cells}")


def cell(key: str, outcomes: Sequence[Outcome], horizons: Sequence[int]) -> Cell:
    fwd = {h: _stat([o.fwd_bps[h] for o in outcomes if h in o.fwd_bps]) for h in horizons}
    hit: dict[int, float] = {}
    for h in horizons:
        signed = [o.fwd_bps[h] for o in outcomes if h in o.fwd_bps and o.fwd_bps[h] != 0.0]
        hit[h] = sum(v > 0 for v in signed) / len(signed) if signed else 0.0
    return Cell(key=key, n=len(outcomes), fwd=fwd, hit=hit,
                pre=_stat([o.pre_bps * o.signal.sign for o in outcomes]))


def breakdown(
    outcomes: Sequence[Outcome],
    key_of: Callable[[Outcome], str],
    *,
    horizons: Sequence[int] = REPORT_HORIZONS,
    min_n: int = 1,
) -> list[Cell]:
    """Group the outcomes by one key and summarise each group, biggest first."""
    groups: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        groups.setdefault(key_of(outcome) or "none", []).append(outcome)
    rows = [cell(key, group, horizons) for key, group in groups.items()
            if len(group) >= min_n]
    rows.sort(key=lambda c: -c.n)
    return rows


def header(horizons: Sequence[int]) -> str:
    cols = "".join(f"{'+' + str(h) + 'm':>7}{'s.e.':>7}" for h in horizons)
    return f"{'':<27}{'n':>7}{'pre':>7}{'hit15':>7}{cols}"


def by_reading(
    outcomes: Sequence[Outcome],
    readings: Sequence[WireReading],
    field_name: str,
    *,
    threshold: float = 0.5,
) -> Callable[[Outcome], str]:
    """A key function that looks the outcome's post up among the readings.

    A float answer (``scheduled``, ``is_number``) is bucketed at ``threshold``
    into ``yes``/``no`` rather than printed as a number, because "the mean
    forward return where p(scheduled) = 0.63" is not a sentence anyone wants.
    """
    index = {r.id: r for r in readings}

    def key(outcome: Outcome) -> str:
        reading = index.get(outcome.signal.code.split("sample:", 1)[-1])
        if reading is None:
            return "unread"
        value = getattr(reading, field_name, "")
        if isinstance(value, float):
            return f"{field_name}=yes" if value >= threshold else f"{field_name}=no"
        return str(value or "none")

    return key


def by_session(outcome: Outcome) -> str:
    return W.sessions(outcome.signal.ts)


def by_pair(outcome: Outcome) -> str:
    return outcome.signal.pair


@dataclass
class Abstention:
    """How often the reader declined to trade, per category. A result in itself.

    On a wire where most posts are recaps and chart levels, a reader that trades
    everything is worse than useless; the share it stays out of is the number
    that says whether the tree is filtering or just answering.
    """

    category: str
    read: int
    traded: int

    @property
    def rate(self) -> float:
        return 1.0 - (self.traded / self.read if self.read else 0.0)


def abstentions(readings: Sequence[WireReading], threshold: float) -> list[Abstention]:
    read: dict[str, int] = {}
    traded: dict[str, int] = {}
    for reading in readings:
        key = reading.category or "none"
        read[key] = read.get(key, 0) + 1
        if reading.signals(threshold):
            traded[key] = traded.get(key, 0) + 1
    rows = [Abstention(k, v, traded.get(k, 0)) for k, v in read.items()]
    rows.sort(key=lambda a: -a.read)
    return rows


def answer_counts(readings: Sequence[WireReading], field_name: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for reading in readings:
        key = str(getattr(reading, field_name, "") or "none")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------- latency sweep


@dataclass
class SweepRow:
    latency_s: float
    measured: int
    spread: Stat
    fwd: dict[int, Stat]
    null: dict[int, Stat]
    z: dict[int, float]


def latency_sweep(
    tapes: C.MinuteTapes,
    signals: Sequence[Signal],
    *,
    latencies: Iterable[float] = C.LATENCIES,
    horizons: Iterable[int] = (15, 60),
    per: int = 1,
    workers: int = 4,
    seed: int = 0,
) -> list[SweepRow]:
    """The same signals entered later and later.

    **On minute bars 0 s and 60 s are usually the same trade**: both round up to
    the next bar boundary, and they differ only for a post stamped exactly on
    one. The sweep is 0 / 60 / 300 rather than the tick study's five rows for
    that reason, and the five-minute row is the only one that can honestly
    differ -- it is the question in the form somebody would ask it: can a person
    who reads the headline and clicks still catch this?
    """
    horizons = tuple(horizons)
    out: list[SweepRow] = []
    for latency in latencies:
        outcomes = measure(tapes, signals, horizons=horizons, workers=workers,
                           latency_s=latency)
        nulls = measure(
            tapes,
            null_signals(tapes, outcomes, per=per, horizons=horizons, seed=seed,
                         latency_s=latency, workers=workers),
            horizons=horizons, workers=workers, latency_s=latency,
        )
        fwd = {h: _stat([o.fwd_bps[h] for o in outcomes if h in o.fwd_bps]) for h in horizons}
        null = {h: _stat([o.fwd_bps[h] for o in nulls if h in o.fwd_bps]) for h in horizons}
        z = {}
        for h in horizons:
            spread = math.sqrt(fwd[h].se**2 + null[h].se**2)
            z[h] = (fwd[h].mean - null[h].mean) / spread if spread > 0 else 0.0
        out.append(SweepRow(
            latency_s=latency, measured=len(outcomes),
            spread=_stat([o.spread_bps for o in outcomes]), fwd=fwd, null=null, z=z,
        ))
    return out


def median_body(articles: Sequence[W.Article]) -> int:
    lengths = [len(a.body) for a in articles]
    return int(statistics.median(lengths)) if lengths else 0


__all__ = [
    "Abstention", "Cell", "DEFAULT_SAMPLE", "HORIZONS", "REPORT_HORIZONS",
    "SweepRow", "abstentions", "answer_counts", "breakdown", "by_pair",
    "by_reading", "by_session", "cell", "cost_usd", "header", "keyword_signals",
    "latency_sweep", "measure", "median_body", "null_signals", "read_all",
    "reader_signals", "reading_from_dict", "reading_to_dict", "sample_signals",
    "state_hash",
]
