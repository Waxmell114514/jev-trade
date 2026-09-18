"""Offline stand-in for Jev on the headline questions.

Read this before drawing any conclusion from an offline run.

This stub sees only what the model sees -- the state dict, i.e. the headline
text -- and it has no way to understand it. So it does the only thing a program
can do without language understanding: match words. It fires on alarming
vocabulary, cannot tell a denial from an event, and cannot tell a fresh report
from a recap.

That makes the offline ``jev`` arm **approximately the keyword arm with extra
steps**, and it should score like it. That is the honest offline result, and it
is precisely the gap the real model has to close. Set TYPESAFE_API_KEY to
measure the real thing.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ..types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer
from .events import keyword_alarm
from .questions import (
    DIRECTION,
    DOWN,
    HAPPENED,
    INFORMED_FLOW,
    MOVES_PRICE,
    NEITHER,
    NEW_INFORMATION,
    REPORT_TYPE,
    SEVERITY,
    SEVERITY_LEVELS,
    UNCLEAR,
    UP,
)

NEGATIVE_WORDS = ("halt", "halts", "halted", "enforcement", "breach",
                  "unauthorised", "liquidating", "outage", "investigation")
POSITIVE_WORDS = ("approved", "approval", "allocation", "sovereign wealth",
                  "raises", "ease")


class MockHeadlineClient:
    """Keyword matching dressed in the System One response schema."""

    provider = "mock"

    def __init__(self, model: str = "mock-keyword", **_ignored: Any) -> None:
        self.model = model

    def evaluate(self, state: dict[str, Any]) -> JevResponse:
        headline = str(state.get("headline", ""))
        low = headline.lower()
        rng = random.Random(hashlib.blake2s(low.encode()).digest()[:8])

        alarm = keyword_alarm(headline)
        moves = 0.78 if alarm else 0.12
        severity_target = 2.3 if alarm else 0.4

        neg = sum(w in low for w in NEGATIVE_WORDS)
        pos = sum(w in low for w in POSITIVE_WORDS)
        if neg > pos:
            probs = {UP: 0.12, DOWN: 0.66, UNCLEAR: 0.22}
        elif pos > neg:
            probs = {UP: 0.66, DOWN: 0.12, UNCLEAR: 0.22}
        else:
            probs = {UP: 0.25, DOWN: 0.25, UNCLEAR: 0.50}

        levels = len(SEVERITY_LEVELS)
        weights = [max(0.02, 1.0 - abs(i - severity_target)) for i in range(levels)]
        total = sum(weights)
        sev_probs = {str(i): w / total for i, w in enumerate(weights)}

        flow = str(state.get("recent_trading", ""))
        one_sided = "almost every recent trade" in flow

        return JevResponse(
            model=self.model,
            answers={
                MOVES_PRICE: NoulAnswer(noul=moves),
                # No language understanding: it cannot see a denial or a recap.
                NEW_INFORMATION: NoulAnswer(noul=0.55 + rng.gauss(0, 0.03)),
                REPORT_TYPE: ChoiceAnswer(
                    choice=HAPPENED if alarm else NEITHER,
                    probabilities={HAPPENED: 0.72 if alarm else 0.30,
                                   "denied": 0.08,
                                   NEITHER: 0.20 if alarm else 0.62},
                    confidence=0.45,
                ),
                DIRECTION: ChoiceAnswer(
                    choice=max(probs, key=probs.__getitem__),
                    probabilities=probs,
                    confidence=0.55 if max(probs.values()) > 0.5 else 0.2,
                ),
                SEVERITY: ScoreAnswer(
                    score=sum(i * p for i, p in enumerate(sev_probs.values())),
                    legend={str(i): s for i, s in enumerate(SEVERITY_LEVELS)},
                    probabilities=sev_probs,
                    confidence=0.4,
                ),
                INFORMED_FLOW: NoulAnswer(noul=0.7 if one_sided else 0.15),
            },
            input_tokens=max(1, len(repr(state)) // 4 + 260),
            output_tokens=0,
            latency_ms=0.05,
            provider=self.provider,
        )
