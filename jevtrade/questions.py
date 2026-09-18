"""The typed questions Jev answers about the tape.

Design rules, taken straight from TypeSafe's docs:

* **One request, many questions.** Jev ingests the state once and evaluates
  every question against it in parallel, so a seventh question costs a few
  tokens and almost no latency. https://docs.typesafe.ai/patterns/fan-out
* **One judgment per question.** A broad call like "should I trade this?" is
  split into direction, follow-through, setup quality, liquidity, hazard and
  exit -- and combined with weights in ``policy.py``, where they can be tuned
  without touching a prompt. https://docs.typesafe.ai/patterns/composite-scoring
* **Say exactly what you mean.** jev-1.13 answers the question as written, not
  the one you meant, so the boundary cases live in ``criteria``.
* **Choice is relative, Noul is absolute.** The Choice settles *which* way the
  tape leans; the Nouls ask *whether* a condition holds. Their numbers are not
  comparable to each other and the policy never mixes them.
"""

from __future__ import annotations

from typing import Any

DIRECTION = "direction"
FOLLOW_THROUGH = "follow_through"
SETUP_QUALITY = "setup_quality"
LIQUIDITY_OK = "liquidity_ok"
DISORDERLY = "disorderly"
CUT_POSITION = "cut_position"

UP, DOWN, UNCLEAR = "up", "down", "unclear"
CONTINUATION, REVERSAL, NO_PATTERN = "continuation", "reversal", "no_clear_pattern"

SETUP_LEVELS = [
    "Nothing readable. The tape gives no usable lean in either direction.",
    "Weak. There is a slight lean, but it could easily be noise.",
    "Clear. Several parts of the tape agree on the same direction.",
    "Textbook. Price action, traded volume and resting size all point the same way.",
]

QUESTION_IDS = (
    DIRECTION,
    FOLLOW_THROUGH,
    SETUP_QUALITY,
    LIQUIDITY_OK,
    DISORDERLY,
    CUT_POSITION,
)


def trading_questions() -> dict[str, dict[str, Any]]:
    """The full question map sent with every decision."""
    return {
        DIRECTION: {
            "type": "choice",
            "instructions": (
                "Over the next few snapshots, which way is the price of this "
                "instrument more likely to move? Judge only from the tape "
                "described in the state."
            ),
            "criteria": {
                UP: "The next move is more likely to be up than down.",
                DOWN: "The next move is more likely to be down than up.",
                UNCLEAR: (
                    "Neither direction is favoured. The tape does not lean "
                    "either way."
                ),
            },
        },
        FOLLOW_THROUGH: {
            "type": "choice",
            "instructions": (
                "The price move described in price_action is best read as "
                "which of these?"
            ),
            "criteria": {
                CONTINUATION: (
                    "The move has momentum behind it and is more likely to "
                    "extend in the same direction."
                ),
                REVERSAL: (
                    "The move looks stretched or exhausted and is more likely "
                    "to give back some of the ground it covered."
                ),
                NO_PATTERN: (
                    "There is no meaningful move to extend or give back."
                ),
            },
        },
        SETUP_QUALITY: {
            "type": "score",
            "instructions": (
                "How clearly does the tape lean one way, judged only on how "
                "much the parts of the state agree with each other?"
            ),
            "criteria": SETUP_LEVELS,
        },
        LIQUIDITY_OK: {
            "type": "noul",
            "instructions": (
                "Is it practical to open a new position of normal size in this "
                "instrument right now?"
            ),
            "criteria": {
                "true": (
                    "The book is at least normal depth and crossing the spread "
                    "costs little next to a typical move."
                ),
                "false": (
                    "The book is thin, or crossing the spread costs as much as "
                    "a typical move or more."
                ),
            },
        },
        DISORDERLY: {
            "type": "noul",
            "instructions": (
                "Does the tape show disorderly conditions, meaning a violent or "
                "erratic move where opening a new position should be avoided?"
            ),
            "criteria": {
                "true": (
                    "Activity is far above normal, the book has thinned, or the "
                    "move looks like a dislocation rather than ordinary trading."
                ),
                "false": "Trading looks orderly, whatever direction it is going.",
            },
        },
        CUT_POSITION: {
            "type": "noul",
            "instructions": (
                "Should the open position described in our_book be reduced or "
                "closed right now?"
            ),
            "criteria": {
                "true": (
                    "The tape has turned against the open position, or the "
                    "position is at its limit while the move that justified it "
                    "has stalled."
                ),
                "false": (
                    "The open position is still supported by the tape, or there "
                    "is no open position to reduce."
                ),
            },
        },
    }


def build_request(
    state: dict[str, Any], model: str = "jev-latest"
) -> dict[str, Any]:
    """A complete POST /v1/systemone body."""
    return {"state": state, "model": model, "questions": trading_questions()}
