"""Score every arm on the same documents, with the spot tape as the judge.

An arm turns a document into signals: (pair, long or short). Each signal enters
at the open of the first bar after the published timestamp and is measured by
signed log return at fixed horizons. The null for each arm is the same pair and
the same side at random moments within five days, placed only where the tape
actually has bars -- which on spot FX is most of the work, since a third of the
random draws land in a weekend.

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
"""

from __future__ import annotations

import concurrent.futures
import math
import random
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..listing.store import Store
from . import tape as _tape
from .baseline import BotSignal, keyword_bot, surprise_bot
from .diff import diff_for
from .documents import Document, match_calendar
from .reader import DiffVerdict, Reader, Reading, Verdict, signed_pair
from .tape import HORIZONS, Bars, forward_returns

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
    signal: Signal
    symbol: str
    pre_bps: float
    release_bar_bps: float
    fwd_bps: dict[int, float]  # signed by the signal's side


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


def measure(
    tape: Tape,
    signals: Sequence[Signal],
    *,
    horizons: Iterable[int] = HORIZONS,
    workers: int = 8,
) -> list[Outcome]:
    """Outcomes for the signals whose pair has contiguous bars around them."""
    horizons = tuple(horizons)

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


def null_signals(
    tape: Tape,
    outcomes: Sequence[Outcome],
    *,
    per: int = 2,
    span_days: float = 5.0,
    exclude_h: float = 3.0,
    horizons: Iterable[int] = HORIZONS,
    seed: int = 0,
    tries: int = 40,
) -> list[Signal]:
    """The same pairs and sides at random nearby moments the tape covers.

    Unlike the crypto study, a random moment near an FX event is very often the
    weekend, so the draw is repeated until it lands on measurable bars. The
    controls are therefore matched on *session* as well as on pair and side.
    """
    rng = random.Random(seed)
    horizons = tuple(horizons)
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
) -> ArmSummary:
    horizons = tuple(horizons)
    fwd = {h: _stat([o.fwd_bps[h] for o in outcomes]) for h in horizons}
    null = {h: _stat([o.fwd_bps[h] for o in nulls]) for h in horizons}
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
    )


# ------------------------------------------------------------------ reading


def read_all(
    documents: Sequence[Document],
    make_reader: Callable[[], Reader],
    *,
    store: Store | None = None,
    cache_tag: str = "",
    calendar: Sequence[dict[str, Any]] = (),
    max_diffs: int = 12,
    workers: int = 1,
) -> list[Reading]:
    """Read every document, one reader per worker (the client mutates its map).

    The diff against the previous edition is computed here, from the same
    document list, so a statement read in isolation and one read inside a run
    see the same changed sentences.
    """
    local = threading.local()
    corpus = list(documents)
    rows = list(calendar)

    def reader() -> Reader:
        if not hasattr(local, "reader"):
            local.reader = make_reader()
        return local.reader

    def one(document: Document) -> Reading:
        key = f"reading:{cache_tag}:{document.id}"
        if store is not None:
            found, data = store.get(key)
            if found:
                return reading_from_dict(document, data)
        previous, changes = diff_for(document, corpus, limit=max_diffs)
        result = reader().read(
            document, previous=previous, changes=changes,
            calendar=match_calendar(document, rows) if rows else None,
        )
        if store is not None:
            store.put(key, reading_to_dict(result))
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
    "ArmSummary", "Disagreement", "Outcome", "Signal", "Stat", "Tape",
    "all_text_signals", "bot_signals", "cost_usd", "disagreements", "keyword_bot",
    "measure", "null_signals", "read_all", "reader_signals", "reading_from_dict",
    "reading_to_dict", "summarize", "surprise_bot", "surprise_signals",
]
