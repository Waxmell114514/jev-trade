"""Does reading the headline help pick the ones that matter?

The aggregate question -- "do crypto headlines move the price" -- is the wrong
one, because most of a feed is altcoin funding rounds and market recaps that
nobody expects to move BTC. The question that decides the thesis is narrower:

    among headlines a reader flags as material, is the move rate higher than
    chance?

So every arm is scored the same way: it selects a subset of headlines, and we
compare that subset's move rate against the rate for randomly timed fake
headlines over the same window. An arm that adds nothing selects a subset that
moves at the null rate.
"""

from __future__ import annotations

import random
import re
import statistics
from dataclasses import dataclass, field

from ..mm.questions import build_state, headline_questions
from ..mm.strategies import headline_risk
from .feeds import Headline
from .label import Bars, Labelled, label

# A plausible desk keyword list for crypto, written from domain knowledge
# before looking at which headlines actually moved. Broad on purpose: this is
# what a rule-based news trigger has to work with.
KEYWORDS = (
    "hack", "hacked", "hackers", "exploit", "exploited", "breach", "stolen",
    "drain", "drained", "halt", "halts", "halted", "suspend", "suspends",
    "suspended", "freeze", "frozen", "outage", "insolvent", "bankruptcy",
    "liquidation", "liquidated", "delist", "delisted",
    "sec", "cftc", "doj", "lawsuit", "sues", "sued", "subpoena", "indicted",
    "enforcement", "investigation", "probe", "ban", "banned", "crackdown",
    "regulation", "regulator", "ruling", "court", "settlement", "fine",
    "approval", "approves", "approved", "etf", "listing", "greenlight",
    "fed", "fomc", "rate", "rates", "inflation", "cpi", "central bank",
    "treasury", "tariff", "recession",
    "whale", "dump", "crash", "plunge", "surge", "soar", "rally", "record",
    "all-time high", "ath", "unlock", "halving", "fork", "upgrade",
)


def keyword_hit(title: str) -> bool:
    low = title.lower()
    return any(re.search(rf"\b{re.escape(w)}\b", low) for w in KEYWORDS)


@dataclass
class ArmResult:
    name: str
    selected: int
    total: int
    movers: int
    rate: float = 0.0
    null_rate: float = 0.0
    null_sd: float = 0.0
    z: float = 0.0
    examples: list[str] = field(default_factory=list)

    @property
    def fire_rate(self) -> float:
        return self.selected / self.total if self.total else 0.0


def null_rate(
    rows: list[Labelled],
    bars: Bars,
    *,
    threshold: float,
    horizon_bars: int,
    n_select: int,
    trials: int = 400,
    seed: int = 0,
    clean: bool = True,
) -> tuple[float, float]:
    """Move rate for ``n_select`` randomly timed fake headlines."""
    if not rows or n_select <= 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    lo, hi = rows[0].headline.ts, rows[-1].headline.ts
    rates: list[float] = []
    for _ in range(trials):
        fake = [
            Headline(ts=rng.uniform(lo, hi), title="", source="null")
            for _ in range(n_select)
        ]
        labelled = label(fake, bars, horizon_bars=horizon_bars)
        if not labelled:
            continue
        hits = sum(
            (r.clean_mover(threshold) if clean else r.moved(threshold))
            for r in labelled
        )
        rates.append(hits / len(labelled))
    if not rates:
        return 0.0, 0.0
    return statistics.fmean(rates), statistics.pstdev(rates)


def score_arm(
    name: str,
    selected: list[Labelled],
    rows: list[Labelled],
    bars: Bars,
    *,
    threshold: float,
    horizon_bars: int,
    clean: bool = True,
    seed: int = 0,
) -> ArmResult:
    hits = [
        r for r in selected
        if (r.clean_mover(threshold) if clean else r.moved(threshold))
    ]
    mean, sd = null_rate(
        rows, bars, threshold=threshold, horizon_bars=horizon_bars,
        n_select=len(selected), trials=400, seed=seed, clean=clean,
    )
    rate = len(hits) / len(selected) if selected else 0.0
    return ArmResult(
        name=name,
        selected=len(selected),
        total=len(rows),
        movers=len(hits),
        rate=rate,
        null_rate=mean,
        null_sd=sd,
        z=(rate - mean) / sd if sd > 0 else 0.0,
        examples=[r.headline.title for r in hits[:3]],
    )


def jev_scores(
    rows: list[Labelled], client, *, instrument: str = "BTC-USD spot"
) -> dict[int, float]:
    """One risk score per headline, using the market-making question set."""
    if hasattr(client, "questions"):
        client.questions = headline_questions()
    out: dict[int, float] = {}
    for index, row in enumerate(rows):
        state = build_state(
            headline=row.headline.title,
            instrument=instrument,
            inventory="flat",
            recent_trading="buying and selling have been roughly balanced",
            trigger="a news headline just arrived",
        )
        out[index] = headline_risk(client.evaluate(state))
    return out


# --------------------------------------------------------------- the crux test
#
# Everything above compares a selected subset against randomly timed controls.
# That is enough to show whether an arm picks better than chance, but not
# whether the *headline* carries the information: news clusters around busy
# moments, and a busy moment stays busy for a while on its own.
#
# So the test that settles it holds the recent volatility fixed. For each
# flagged headline we take a control drawn at random from moments with a
# comparable move behind them, and ask whether the headline still predicts
# more movement than the control does.


def excursion(bars: Bars, ts: float, horizon: int, forward: bool) -> float | None:
    """|log return| over exactly ``horizon`` bars, in sigmas.

    Symmetric by construction, so the value before a headline and the value
    after it are directly comparable -- a maximum-over-horizons measure is not,
    because the maximum is mechanically larger.
    """
    import math

    i = bars.index_at(ts)
    if i is None or i - horizon < 0 or i + horizon >= len(bars.close):
        return None
    a, b = (
        (bars.close[i], bars.close[i + horizon])
        if forward
        else (bars.close[i - horizon], bars.close[i])
    )
    if a <= 0 or b <= 0:
        return None
    return abs(math.log(b / a)) / (bars.sigma * (horizon**0.5))


def matched_null(
    bars: Bars,
    span: tuple[float, float],
    *,
    pre_low: float,
    pre_high: float,
    n: int,
    horizon: int = 1,
    pre_horizon: int = 3,
    trials: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """Forward move at moments whose *recent* move resembles the flagged ones.

    This is the control that removes volatility clustering. Coarse: the pre-move
    bucket is wide, so a group whose pre-moves sit at the top of the bucket will
    be compared against controls nearer its floor. Treat a positive result here
    as unproven rather than demonstrated; a negative one is solid.
    """
    rng = random.Random(seed)
    low, high = span
    means: list[float] = []
    for _ in range(trials):
        picked: list[float] = []
        attempts = 0
        while len(picked) < n and attempts < 4000:
            attempts += 1
            ts = rng.uniform(low, high)
            pre = excursion(bars, ts, pre_horizon, forward=False)
            if pre is None or not (pre_low <= pre < pre_high):
                continue
            post = excursion(bars, ts, horizon, forward=True)
            if post is not None:
                picked.append(post)
        if len(picked) == n:
            means.append(statistics.fmean(picked))
    if not means:
        return 0.0, 0.0
    return statistics.fmean(means), statistics.pstdev(means)


@dataclass
class CruxResult:
    group: str
    condition: str
    n: int
    after: float
    null: float
    null_sd: float
    z: float


def crux(
    selected: list[Labelled],
    bars: Bars,
    span: tuple[float, float],
    *,
    quiet_below: float = 0.7,
    horizon: int = 1,
    pre_horizon: int = 3,
    group: str = "",
) -> list[CruxResult]:
    """Split by how busy the tape already was, then compare against matched controls."""
    out: list[CruxResult] = []
    buckets = (
        ("quiet before", 0.0, quiet_below),
        ("already moving", quiet_below, 99.0),
    )
    for condition, low, high in buckets:
        rows = []
        for item in selected:
            pre = excursion(bars, item.headline.ts, pre_horizon, forward=False)
            if pre is not None and low <= pre < high:
                after = excursion(bars, item.headline.ts, horizon, forward=True)
                if after is not None:
                    rows.append(after)
        if len(rows) < 5:
            continue
        mean, sd = matched_null(
            bars, span, pre_low=low, pre_high=high, n=len(rows),
            horizon=horizon, pre_horizon=pre_horizon,
        )
        out.append(
            CruxResult(
                group=group, condition=condition, n=len(rows),
                after=statistics.fmean(rows), null=mean, null_sd=sd,
                z=(statistics.fmean(rows) - mean) / sd if sd else 0.0,
            )
        )
    return out
