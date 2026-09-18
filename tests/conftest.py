import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevtrade.questions import (  # noqa: E402
    CONTINUATION,
    CUT_POSITION,
    DIRECTION,
    DISORDERLY,
    DOWN,
    FOLLOW_THROUGH,
    LIQUIDITY_OK,
    NO_PATTERN,
    REVERSAL,
    SETUP_LEVELS,
    SETUP_QUALITY,
    UNCLEAR,
    UP,
)
from jevtrade.types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer  # noqa: E402


def make_response(
    *,
    p_up=0.7,
    p_down=0.2,
    confidence=0.8,
    setup=2.4,
    liquidity=0.9,
    disorderly=0.1,
    cut=0.1,
    continuation=0.6,
    reversal=0.2,
    latency_ms=120.0,
):
    """A JevResponse with every answer the policy reads. Overridable per test."""
    probs = {UP: p_up, DOWN: p_down, UNCLEAR: max(0.0, 1 - p_up - p_down)}
    levels = len(SETUP_LEVELS)
    return JevResponse(
        model="jev-1.13.0",
        answers={
            DIRECTION: ChoiceAnswer(
                choice=max(probs, key=probs.__getitem__),
                probabilities=probs,
                confidence=confidence,
            ),
            FOLLOW_THROUGH: ChoiceAnswer(
                choice=CONTINUATION,
                probabilities={
                    CONTINUATION: continuation,
                    REVERSAL: reversal,
                    NO_PATTERN: max(0.0, 1 - continuation - reversal),
                },
                confidence=0.5,
            ),
            SETUP_QUALITY: ScoreAnswer(
                score=setup,
                legend={str(i): level for i, level in enumerate(SETUP_LEVELS)},
                probabilities={str(i): 1 / levels for i in range(levels)},
                confidence=0.5,
            ),
            LIQUIDITY_OK: NoulAnswer(noul=liquidity),
            DISORDERLY: NoulAnswer(noul=disorderly),
            CUT_POSITION: NoulAnswer(noul=cut),
        },
        input_tokens=600,
        output_tokens=0,
        latency_ms=latency_ms,
        provider="test",
    )


class StubClient:
    """A provider that returns a fixed response and counts calls."""

    provider = "test"
    model = "stub"

    def __init__(self, response=None, latency_ms=120.0, fail_times=0):
        self._response = response
        self.latency_ms = latency_ms
        self.calls = 0
        self.fail_times = fail_times

    def evaluate(self, state):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("provider exploded")
        return self._response or make_response(latency_ms=self.latency_ms)
