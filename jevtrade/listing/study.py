"""Score every arm on the same announcements, with the tape as the judge.

An arm turns an announcement into signals: (token, long or short). For each
signal we enter at the *open of the minute after* the announcement -- a reader
that answers in a second could fill inside the release minute, so this is a
late fill by up to 59 seconds, on purpose -- and measure the signed log return
at fixed horizons. The null for each arm is the same tokens, same sides, at
random moments within a few days of the event: whatever drift the arm's tokens
had that week, the null has too.

Three controls sit next to the reader:

* ``title-bot``   the incumbent: match the title, trade every ticker in it
* ``body-bot``    the same rules over tickers from the body as well, so that
                  "reading the body" and "judging the body" are separable
* ``all mentions`` every candidate ticker of every announcement, bought: the
                  base rate for tokens that get mentioned at all
"""

from __future__ import annotations

import concurrent.futures
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .announcements import Announcement
from .baseline import body_bot, title_bot
from .reader import Reader, Reading, TokenVerdict
from .store import Store
from .tickers import candidates
from .venues import Candles, window

HORIZONS = (1, 5, 15, 60)
BEFORE_MIN = 30
AFTER_MIN = 90
PRE_MIN = 15
INPUT_USD_PER_MTOK = 0.042


@dataclass(frozen=True)
class Signal:
    code: str
    ts: float
    title: str
    token: str
    sign: int
    strength: float = 1.0


@dataclass
class Outcome:
    signal: Signal
    venue: str
    pre_bps: float
    release_bar_bps: float
    fwd_bps: dict[int, float]  # signed by the signal's side


def forward_returns(
    candles: Candles, when: float, sign: int, horizons: Iterable[int] = HORIZONS
) -> tuple[float, float, dict[int, float]] | None:
    """(pre, release-bar, {horizon: signed forward}) in bps, or None if the tape is missing."""
    release = candles.index_at(when)
    if release is None:
        return None
    entry = release + 1
    if entry >= len(candles.open) or release - PRE_MIN < 0:
        return None
    entry_px = candles.open[entry]
    if not (entry_px > 0):
        return None

    def log_bps(a: float, b: float) -> float | None:
        if not (a > 0 and b > 0) or math.isnan(a) or math.isnan(b):
            return None
        return math.log(b / a) * 1e4

    pre = log_bps(candles.open[release - PRE_MIN], candles.open[release])
    bar = log_bps(candles.open[release], candles.close[release])
    fwd: dict[int, float] = {}
    for h in horizons:
        j = entry + h - 1
        if j >= len(candles.close):
            return None
        value = log_bps(entry_px, candles.close[j])
        if value is None:
            return None
        fwd[h] = sign * value
    if pre is None or bar is None:
        return None
    return pre, bar, fwd


def measure(
    store: Store,
    signals: list[Signal],
    *,
    horizons: Iterable[int] = HORIZONS,
    shift_s: float = 0.0,
    workers: int = 8,
    window_fn: Callable[..., Candles | None] = window,
) -> list[Outcome]:
    """Outcomes for the signals whose token has a market with bars at the time."""
    horizons = tuple(horizons)

    def one(signal: Signal) -> Outcome | None:
        when = signal.ts + shift_s
        candles = window_fn(store, signal.token, when, before_min=BEFORE_MIN, after_min=AFTER_MIN)
        if candles is None:
            return None
        result = forward_returns(candles, when, signal.sign, horizons)
        if result is None:
            return None
        pre, bar, fwd = result
        return Outcome(signal=signal, venue=candles.venue, pre_bps=pre,
                       release_bar_bps=bar, fwd_bps=fwd)

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        results = list(pool.map(one, signals))
    return [r for r in results if r is not None]


def null_signals(
    outcomes: list[Outcome], *, per: int = 2, span_days: float = 5.0,
    exclude_h: float = 3.0, seed: int = 0,
) -> list[Signal]:
    """The same tokens and sides, at random moments near (not at) the event."""
    rng = random.Random(seed)
    out: list[Signal] = []
    for outcome in outcomes:
        s = outcome.signal
        for _ in range(per):
            while True:
                offset = rng.uniform(-span_days * 86400, span_days * 86400)
                if abs(offset) >= exclude_h * 3600:
                    break
            out.append(Signal(code=f"null:{s.code}", ts=s.ts + offset, title="",
                              token=s.token, sign=s.sign, strength=s.strength))
    return out


# ------------------------------------------------------------------ arms

def bot_signals(announcements: list[Announcement], bot: Callable[[Announcement], list]) -> list[Signal]:
    out: list[Signal] = []
    for a in announcements:
        for s in bot(a):
            out.append(Signal(a.code, a.ts, a.title, s.token, s.sign))
    return out


def mention_signals(announcements: list[Announcement]) -> list[Signal]:
    out: list[Signal] = []
    for a in announcements:
        for token in candidates(a.title, a.body):
            out.append(Signal(a.code, a.ts, a.title, token, +1))
    return out


def reader_signals(
    readings: list[Reading], threshold: float, *, min_confidence: float = 0.5
) -> list[Signal]:
    out: list[Signal] = []
    for r in readings:
        for v in r.signals(threshold, min_confidence=min_confidence):
            out.append(Signal(r.announcement.code, r.announcement.ts, r.announcement.title,
                              v.token, v.sign, v.strength))
    return out


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
    return Stat(len(values), statistics.fmean(values), statistics.stdev(values) / math.sqrt(len(values)))


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

    @property
    def measurable(self) -> float:
        return self.measured / self.signals if self.signals else 0.0


def summarize(
    name: str, signals: list[Signal], outcomes: list[Outcome], nulls: list[Outcome],
    *, horizons: Iterable[int] = HORIZONS,
) -> ArmSummary:
    horizons = tuple(horizons)
    fwd = {h: _stat([o.fwd_bps[h] for o in outcomes]) for h in horizons}
    null = {h: _stat([o.fwd_bps[h] for o in nulls]) for h in horizons}
    hit = {}
    for h in horizons:
        signed = [o.fwd_bps[h] for o in outcomes if o.fwd_bps[h] != 0.0]
        hit[h] = sum(v > 0 for v in signed) / len(signed) if signed else 0.0
    z = {}
    for h in horizons:
        spread = math.sqrt(fwd[h].se**2 + null[h].se**2)
        z[h] = (fwd[h].mean - null[h].mean) / spread if spread > 0 else 0.0
    return ArmSummary(
        name=name, signals=len(signals), measured=len(outcomes),
        pre=_stat([o.pre_bps * o.signal.sign for o in outcomes]),
        release_bar=_stat([o.release_bar_bps * o.signal.sign for o in outcomes]),
        fwd=fwd, hit=hit, null=null, z=z,
    )


# ------------------------------------------------------------------ reading

def read_all(
    announcements: list[Announcement],
    make_reader: Callable[[], Reader],
    *,
    store: Store | None = None,
    cache_tag: str = "",
    workers: int = 1,
) -> list[Reading]:
    """Read every announcement, one reader per worker (the client mutates its question map)."""
    import threading

    local = threading.local()

    def reader() -> Reader:
        if not hasattr(local, "reader"):
            local.reader = make_reader()
        return local.reader

    def one(a: Announcement) -> Reading:
        key = f"reading:{cache_tag}:{a.code}"
        if store is not None:
            found, data = store.get(key)
            if found:
                return reading_from_dict(a, data)
        result = reader().read(a)
        if store is not None:
            store.put(key, reading_to_dict(result))
        return result

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(one, announcements))


def reading_to_dict(r: Reading) -> dict[str, Any]:
    return {
        "tokens": r.tokens, "event_type": r.event_type, "p_event": r.p_event,
        "magnitude": r.magnitude, "conditional": r.conditional, "priced_in": r.priced_in,
        "rounds": r.rounds, "latency_ms": r.latency_ms, "wall_ms": r.wall_ms,
        "input_tokens": r.input_tokens,
        "verdicts": [
            {"token": v.token, "subject": v.subject, "confirm": v.confirm, "side": v.side,
             "p_side": v.p_side, "confidence": v.confidence, "strength": v.strength}
            for v in r.verdicts
        ],
    }


def reading_from_dict(a: Announcement, d: dict[str, Any]) -> Reading:
    return Reading(
        announcement=a, tokens=list(d["tokens"]), event_type=d["event_type"],
        p_event=d["p_event"], magnitude=d["magnitude"], conditional=d["conditional"],
        priced_in=d["priced_in"],
        verdicts=[TokenVerdict(**v) for v in d["verdicts"]],
        rounds=d["rounds"], latency_ms=d["latency_ms"], wall_ms=d["wall_ms"],
        input_tokens=d["input_tokens"],
    )


def cost_usd(readings: list[Reading]) -> float:
    return sum(r.input_tokens for r in readings) * INPUT_USD_PER_MTOK / 1e6


# ------------------------------------------------------------------ audit

@dataclass
class Disagreement:
    title: str
    when: float
    bot: list[tuple[str, int]]
    reader: list[tuple[str, int]]
    outcomes: dict[str, float]  # token -> unsigned 15m return, bps


def disagreements(
    readings: list[Reading], threshold: float, outcomes: list[Outcome], *, horizon: int = 15
) -> list[Disagreement]:
    """Announcements where matching the title and reading it traded differently."""
    by_key: dict[tuple[str, str], float] = {}
    for o in outcomes:
        by_key[(o.signal.code, o.signal.token)] = o.fwd_bps[horizon] * o.signal.sign
    out: list[Disagreement] = []
    for r in readings:
        bot = sorted((s.token, s.sign) for s in title_bot(r.announcement))
        mine = sorted((v.token, v.sign) for v in r.signals(threshold))
        if bot == mine:
            continue
        tokens = {t for t, _ in bot} | {t for t, _ in mine}
        seen = {t: by_key[(r.announcement.code, t)] for t in tokens if (r.announcement.code, t) in by_key}
        out.append(Disagreement(r.announcement.title, r.announcement.ts, bot, mine, seen))
    return out


__all__ = [
    "Signal", "Outcome", "ArmSummary", "Disagreement", "forward_returns", "measure",
    "null_signals", "bot_signals", "mention_signals", "reader_signals", "summarize",
    "read_all", "cost_usd", "disagreements", "title_bot", "body_bot",
]
