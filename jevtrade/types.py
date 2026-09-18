"""Core data types shared by the feed, the Jev client and the trading engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Union

# ---------------------------------------------------------------- market data


@dataclass(frozen=True, slots=True)
class Tick:
    """One top-of-book snapshot for one instrument.

    ``volume`` is the base-asset volume traded during the interval that ended
    at ``ts`` -- not a cumulative total.
    """

    seq: int
    ts: float
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    bid_size: float
    ask_size: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return (self.ask - self.bid) / mid * 1e4 if mid > 0 else 0.0


# ------------------------------------------------------------- Jev primitives
#
# These mirror the answer shapes documented at https://docs.typesafe.ai/api
# exactly. Jev cannot return anything outside them, which is the whole point of
# a System One model: there is no parsing step and no schema-repair retry.


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    """A yes/no probability. Carries no confidence -- see docs.typesafe.ai/confidence."""

    noul: float
    type: str = "noul"


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """One option out of a set we defined, plus the full distribution."""

    choice: str
    probabilities: Mapping[str, float]
    confidence: float
    type: str = "choice"

    def p(self, option: str) -> float:
        return float(self.probabilities.get(option, 0.0))


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """A probability-weighted position on an ordered rubric we defined."""

    score: float
    legend: Mapping[str, str]
    probabilities: Mapping[str, float]
    confidence: float
    type: str = "score"

    @property
    def normalized(self) -> float:
        """Score rescaled to 0..1 across the rubric levels.

        Safe because it only ever asks "how far along the rubric", never
        "what number does this interpolate to" -- jev-1.13 is explicitly weak
        at the latter (docs.typesafe.ai/model-jaggedness/jev-1.13).
        """
        top = max(len(self.legend) - 1, 1)
        return min(max(self.score / top, 0.0), 1.0)


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass(frozen=True, slots=True)
class JevResponse:
    """One /v1/systemone round trip."""

    model: str
    answers: Mapping[str, Answer]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str  # "jev" for the real API, "mock" for the offline stub

    def noul(self, key: str) -> NoulAnswer:
        answer = self.answers[key]
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"answer {key!r} is {answer.type}, expected noul")
        return answer

    def choice(self, key: str) -> ChoiceAnswer:
        answer = self.answers[key]
        if not isinstance(answer, ChoiceAnswer):
            raise TypeError(f"answer {key!r} is {answer.type}, expected choice")
        return answer

    def score(self, key: str) -> ScoreAnswer:
        answer = self.answers[key]
        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"answer {key!r} is {answer.type}, expected score")
        return answer


# -------------------------------------------------------------- trading types


@dataclass(frozen=True, slots=True)
class Decision:
    """A target position produced by code from Jev's answers.

    Jev never sees a quantity. It answers judgment questions; every number
    below was computed in ``policy.py``.
    """

    seq: int  # tick the decision was formed on
    ts: float
    target_units: float
    p_up: float
    p_down: float
    edge: float  # P(up) - P(down)
    conviction: float  # 0..1, from the Score rubric
    confidence: float  # Jev's own confidence on the direction Choice
    hazard: float
    tradeable: float
    reduce_risk: float
    gate: str  # "" when the decision passed every gate, else why it was clamped
    latency_ms: float
    usage_tokens: int


@dataclass(frozen=True, slots=True)
class Fill:
    seq: int
    ts: float
    side: str  # "buy" | "sell"
    units: float
    price: float
    fee: float


@dataclass
class Position:
    units: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    fills: list[Fill] = field(default_factory=list)

    def notional(self, mid: float) -> float:
        return self.units * mid

    def unrealized_pnl(self, mid: float) -> float:
        return (mid - self.avg_price) * self.units

    def gross_pnl(self, mid: float) -> float:
        """Trading P&L before costs."""
        return self.realized_pnl + self.unrealized_pnl(mid)

    def net_pnl(self, mid: float) -> float:
        """What actually lands in the account: gross P&L less fees."""
        return self.gross_pnl(mid) - self.fees_paid

    def equity(self, mid: float) -> float:
        return self.net_pnl(mid)
