"""An offline stand-in for Jev.

Jev is a closed, managed API behind an early-access waitlist, so this repo has
to be runnable without a key. ``MockJevClient`` returns schema-valid answers so
every other module can be developed and tested end to end.

It is **not a model and not an approximation of one.** It is a hand-written
scoring function over the same label vocabulary Jev is shown. Two properties
keep it honest:

* It reads only the ``state`` dict -- the same words the real model gets. It
  cannot see the raw features, and it cannot see the simulator's hidden state,
  so it has no oracle advantage.
* Every report produced from it is stamped ``provider=mock``.

Numbers produced with this client say something about the plumbing and nothing
whatsoever about Jev.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Any, Sequence

from .. import discretize as D
from ..questions import (
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
from ..types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer


def _ordinal(label: str, vocabulary: Sequence[str]) -> float:
    """Position of a label within its vocabulary, scaled to 0..1."""
    try:
        index = vocabulary.index(label)
    except ValueError:
        return 0.5
    return index / max(len(vocabulary) - 1, 1)


def _signed(label: str, vocabulary: Sequence[str]) -> float:
    """Position of a label within its vocabulary, scaled to -1..1."""
    return _ordinal(label, vocabulary) * 2.0 - 1.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    top = max(values)
    exps = [math.exp((v - top) / temperature) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _confidence(probabilities: Sequence[float]) -> float:
    """Concentrated distribution -> high confidence, flat -> low.

    Normalised entropy, which matches how TypeSafe describes the statistic:
    "a flatter distribution means lower confidence".
    """
    n = len(probabilities)
    if n <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(n)))


class MockJevClient:
    """Deterministic offline stub with Jev's response shape."""

    provider = "mock"

    def __init__(
        self,
        model: str = "mock-jev",
        *,
        latency_ms: float = 0.0,
        noise: float = 0.35,
        **_ignored: Any,
    ) -> None:
        self.model = model
        self.latency_ms = latency_ms
        self.noise = noise

    def evaluate(self, state: dict[str, Any]) -> JevResponse:
        started = time.perf_counter()
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        answers = self._answer(state)
        # Rough stand-in for the real token count: ~4 characters per token.
        input_tokens = max(1, len(repr(state)) // 4 + 520)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return JevResponse(
            model=self.model,
            answers=answers,
            input_tokens=input_tokens,
            output_tokens=0,
            latency_ms=latency_ms,
            provider=self.provider,
        )

    # ------------------------------------------------------------- scoring

    def _answer(self, state: dict[str, Any]) -> dict[str, Any]:
        price = state.get("price_action", {})
        activity = state.get("activity", {})
        book = state.get("order_book", {})
        ours = state.get("our_book", {})

        move_1 = _signed(price.get("last_snapshot", ""), D.MOVE)
        move_5 = _signed(price.get("last_5_snapshots", ""), D.MOVE)
        move_15 = _signed(price.get("last_15_snapshots", ""), D.MOVE)
        character = _ordinal(price.get("character", ""), D.CHARACTER)
        vs_session = _signed(price.get("versus_session", ""), D.VS_VWAP)

        volatility = _ordinal(activity.get("volatility", ""), D.VOLATILITY)
        volume = _ordinal(activity.get("traded_volume", ""), D.VOLUME)

        resting = _signed(book.get("resting_size", ""), D.BOOK)
        depth = _ordinal(book.get("depth", ""), D.DEPTH)
        cross_cost = _ordinal(book.get("cost_to_cross_the_spread", ""), D.CROSS_COST)

        exposure = _signed(ours.get("exposure", ""), D.EXPOSURE)
        open_pnl = _signed(ours.get("open_position_pnl", ""), D.OPEN_PNL)

        rng = random.Random(
            hashlib.blake2s(repr(sorted(state.items())).encode()).digest()[:8]
        )
        jitter = lambda scale=1.0: rng.gauss(0.0, self.noise * scale)  # noqa: E731

        trend = 0.25 * move_1 + 0.45 * move_5 + 0.30 * move_15
        stretched = vs_session * (1.0 - character)
        lean = (
            1.15 * resting
            + 1.05 * trend * (0.35 + 0.65 * character)
            - 0.55 * stretched
            + jitter(0.8)
        )

        strength = abs(lean) * (0.45 + 0.55 * character)
        p_unclear = min(0.88, max(0.04, 1.0 / (1.0 + 2.4 * strength)))
        remainder = 1.0 - p_unclear
        p_up = remainder * _sigmoid(3.1 * lean)
        direction_probs = {UP: p_up, DOWN: remainder - p_up, UNCLEAR: p_unclear}

        continuation = 1.15 * character + 0.75 * abs(trend) - 0.85 * abs(stretched)
        reversal = 1.0 * abs(stretched) + 0.55 * (1.0 - character) - 0.2
        neither = 1.15 - 1.0 * abs(trend)
        follow_probs = dict(
            zip(
                (CONTINUATION, REVERSAL, NO_PATTERN),
                _softmax(
                    [
                        continuation + jitter(0.5),
                        reversal + jitter(0.5),
                        neither + jitter(0.5),
                    ],
                    temperature=0.75,
                ),
            )
        )

        agreement = 1.6 * strength + 0.8 * character - 0.5
        setup_probs = _softmax(
            [-agreement + jitter(0.4), 0.3 + jitter(0.4), agreement + jitter(0.4),
             1.5 * agreement - 0.9 + jitter(0.4)],
            temperature=0.9,
        )
        setup_score = sum(i * p for i, p in enumerate(setup_probs))

        liquidity = _sigmoid(3.0 * (0.62 - cross_cost) + 2.0 * (depth - 0.3))
        disorderly = _sigmoid(
            3.2 * (volatility - 0.72) + 1.6 * (volume - 0.75) + 1.8 * (0.35 - depth)
        )

        against = -exposure * lean  # positive when the tape fights the position
        cut = _sigmoid(
            2.6 * against
            + 1.5 * max(0.0, abs(exposure) - 0.8)
            - 0.7 * open_pnl
            - 1.1
        )
        if abs(exposure) < 0.05:
            cut = min(cut, 0.05)

        return {
            DIRECTION: ChoiceAnswer(
                choice=max(direction_probs, key=direction_probs.__getitem__),
                probabilities=direction_probs,
                confidence=_confidence(list(direction_probs.values())),
            ),
            FOLLOW_THROUGH: ChoiceAnswer(
                choice=max(follow_probs, key=follow_probs.__getitem__),
                probabilities=follow_probs,
                confidence=_confidence(list(follow_probs.values())),
            ),
            SETUP_QUALITY: ScoreAnswer(
                score=setup_score,
                legend={str(i): level for i, level in enumerate(SETUP_LEVELS)},
                probabilities={str(i): p for i, p in enumerate(setup_probs)},
                confidence=_confidence(setup_probs),
            ),
            LIQUIDITY_OK: NoulAnswer(noul=liquidity),
            DISORDERLY: NoulAnswer(noul=disorderly),
            CUT_POSITION: NoulAnswer(noul=cut),
        }
