"""Read an announcement with a two-round judgment tree.

Measured, not assumed: one request costs the same whether it carries one
question or twenty-four, and a second request costs another round trip. So
the budget is *rounds*, not judgments, and the tree is wide and shallow.

Round one asks everything that might matter, speculatively: what kind of event
this is, which way it cuts, how big, whether it is conditional -- and, for
every ticker the code found in the text, whether that ticker is what the
announcement is *about*. Round two runs only if some ticker cleared that bar,
and asks the per-token questions plus a reversed-framing check on each. Two
rephrasings of one question make partially independent errors; that is the
cheapest redundancy there is.

Every number here is arithmetic on the model's probabilities; the model is
never asked to compute one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..types import JevResponse
from .announcements import Announcement
from .tickers import candidates

EVENT_TYPE = "event_type"
EFFECT = "effect"
MAGNITUDE = "magnitude"
CONDITIONAL = "conditional"
PRICED_IN = "priced_in"

NEW_LISTING = "new_listing"
MORE_MARKETS = "more_markets"
DELISTING = "delisting"
WARNING = "warning"
HOUSEKEEPING = "housekeeping"
TOKENIZED_STOCK = "tokenized_stock"
EVENT_TYPES = (NEW_LISTING, MORE_MARKETS, DELISTING, WARNING, HOUSEKEEPING, TOKENIZED_STOCK)

POSITIVE, NEGATIVE, NONE = "positive", "negative", "none"

MAGNITUDE_LEVELS = (
    "no effect on the token's price",
    "a small effect, the kind that fades within the hour",
    "a clear effect that traders would position for",
    "a large effect, likely the biggest news of the day for this token",
)


def subject_key(i: int) -> str:
    return f"subject_{i}"


def effect_key(i: int) -> str:
    return f"effect_{i}"


def unaffected_key(i: int) -> str:
    return f"unaffected_{i}"


def _noul(instructions: str, yes: str, no: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": yes, "false": no}}


def round_one_questions(tokens: list[str]) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {
        EVENT_TYPE: {
            "type": "choice",
            "instructions": "What kind of announcement is this?",
            "criteria": {
                NEW_LISTING: "A token gets its first spot market on this exchange.",
                MORE_MARKETS: (
                    "A token that already trades here gets more: a perpetual, "
                    "margin, earn, convert, collateral status, or new pairs."
                ),
                DELISTING: (
                    "A token, or its trading pairs, is being removed, or "
                    "deposits and withdrawals for it are being closed."
                ),
                WARNING: "A risk, monitoring or seed tag is applied to an existing token.",
                HOUSEKEEPING: (
                    "Routine: delivery contracts rolling, contract swaps, "
                    "maintenance, fee promotions, product notices, anything "
                    "that changes no token's standing on the exchange."
                ),
                TOKENIZED_STOCK: "It concerns tokenized stocks or securities, not crypto tokens.",
            },
        },
        EFFECT: {
            "type": "choice",
            "instructions": (
                "For the token(s) this announcement is about, which way does it "
                "push the price?"
            ),
            "criteria": {
                POSITIVE: "Holders are better off: more access, more demand, more legitimacy.",
                NEGATIVE: "Holders are worse off: less access, a warning, a removal.",
                NONE: "It changes nothing about what the token is worth.",
            },
        },
        MAGNITUDE: {
            "type": "score",
            "instructions": "How much would this announcement move the token(s) it is about?",
            "criteria": list(MAGNITUDE_LEVELS),
        },
        CONDITIONAL: _noul(
            "Is the action described conditional, tentative, or subject to change?",
            "It depends on something not yet decided, or the text hedges it.",
            "It is stated as a decision with a time.",
        ),
    }
    for i, token in enumerate(tokens):
        questions[subject_key(i)] = _noul(
            f"Is {token} one of the assets this announcement is about -- an asset "
            "being listed, delisted, tagged, or given a new market -- rather than a "
            "quote or settlement currency, collateral, a fee currency, an example, "
            "an index component, or an unrelated mention?",
            f"{token}'s standing on the exchange changes because of this announcement.",
            f"{token} is only mentioned in passing, or as the other side of a pair.",
        )
    return questions


def round_two_questions(tokens: list[str], indices: list[int]) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {
        PRICED_IN: _noul(
            "Was the substance of this already public or already in effect "
            "before this announcement -- a restated notice, a change that has "
            "already happened, or something the text says was previously announced?",
            "The market already knew this.",
            "This is the first time it is being announced.",
        ),
    }
    for i in indices:
        token = tokens[i]
        questions[effect_key(i)] = {
            "type": "choice",
            "instructions": f"For holders of {token} specifically, which way does this announcement cut?",
            "criteria": {
                POSITIVE: f"Good for {token}: a new market, more access, a tag removed.",
                NEGATIVE: f"Bad for {token}: a removal, a closure, a warning tag.",
                NONE: f"Neither: {token} is not really affected.",
            },
        }
        questions[unaffected_key(i)] = _noul(
            f"Would a holder of {token} have no reason to change their position "
            "after reading this?",
            f"Nothing here changes {token}'s prospects.",
            f"A {token} holder would want to act on this.",
        )
    return questions


def build_state(
    announcement: Announcement,
    tokens: list[str],
    *,
    round_no: int,
    body_chars: int,
    so_far: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "venue": "Binance",
        "title": announcement.title,
        "body": announcement.body[:body_chars],
        "candidate_tokens": tokens,
        "round": round_no,
    }
    if so_far:
        state["first_round_verdicts"] = so_far
    return state


# ------------------------------------------------------------------ verdicts

def strength(
    subject: float,
    confirm: float,
    magnitude: float,
    p_side: float,
    conditional: float,
    priced_in: float,
) -> float:
    """Every factor is a probability or a rubric position; code does the multiplying."""
    value = subject * confirm * magnitude * p_side
    value *= 1.0 - 0.7 * conditional
    value *= 1.0 - 0.5 * priced_in
    return max(0.0, min(1.0, value))


@dataclass
class TokenVerdict:
    token: str
    subject: float  # round one: is the announcement about this token
    confirm: float  # round two, reversed framing: 1 - p(holder unaffected)
    side: str  # positive / negative / none
    p_side: float
    confidence: float
    strength: float

    @property
    def sign(self) -> int:
        return {POSITIVE: 1, NEGATIVE: -1}.get(self.side, 0)


@dataclass
class Reading:
    announcement: Announcement
    tokens: list[str]
    event_type: str
    p_event: float
    magnitude: float
    conditional: float
    priced_in: float
    verdicts: list[TokenVerdict]
    rounds: int
    latency_ms: float
    wall_ms: float
    input_tokens: int
    responses: list[JevResponse] = field(default_factory=list, repr=False)

    def signals(self, threshold: float, *, min_confidence: float = 0.5) -> list[TokenVerdict]:
        return [
            v for v in self.verdicts
            if v.sign != 0 and v.strength > 0.0 and v.strength >= threshold
            and v.confidence >= min_confidence
        ]


class Reader:
    def __init__(
        self,
        client,
        *,
        body_chars: int = 6000,
        max_tokens: int = 24,
        subject_floor: float = 0.5,
    ) -> None:
        self.client = client
        self.body_chars = body_chars
        self.max_tokens = max_tokens
        self.subject_floor = subject_floor

    def _ask(self, questions: dict[str, Any], state: dict[str, Any]) -> JevResponse:
        self.client.questions = questions
        return self.client.evaluate(state)

    def read(self, announcement: Announcement) -> Reading:
        started = time.perf_counter()
        tokens = candidates(announcement.title, announcement.body, limit=self.max_tokens)

        first = self._ask(
            round_one_questions(tokens),
            build_state(announcement, tokens, round_no=1, body_chars=self.body_chars),
        )
        event = first.choice(EVENT_TYPE)
        effect = first.choice(EFFECT)
        magnitude = first.score(MAGNITUDE).normalized
        conditional = first.noul(CONDITIONAL).noul
        subjects = {i: first.noul(subject_key(i)).noul for i in range(len(tokens))}
        responses = [first]

        survivors = [i for i, p in subjects.items() if p >= self.subject_floor]
        priced_in = 0.0
        second: JevResponse | None = None
        if survivors:
            so_far = {
                "event_type": event.choice,
                "effect": effect.choice,
                "tokens_the_announcement_is_about": [tokens[i] for i in survivors],
            }
            second = self._ask(
                round_two_questions(tokens, survivors),
                build_state(
                    announcement, tokens, round_no=2,
                    body_chars=self.body_chars, so_far=so_far,
                ),
            )
            priced_in = second.noul(PRICED_IN).noul
            responses.append(second)

        verdicts: list[TokenVerdict] = []
        for i, token in enumerate(tokens):
            if second is not None and i in survivors:
                side_answer = second.choice(effect_key(i))
                confirm = 1.0 - second.noul(unaffected_key(i)).noul
            else:
                # Never traded: the second round *is* the confirmation, and a
                # token the model says the announcement is not about does not
                # get one. The verdict is still recorded for the audit.
                side_answer = effect
                confirm = 0.0
            side = side_answer.choice
            p_side = side_answer.p(side) if side in (POSITIVE, NEGATIVE) else 0.0
            verdicts.append(
                TokenVerdict(
                    token=token,
                    subject=subjects[i],
                    confirm=confirm,
                    side=side,
                    p_side=p_side,
                    confidence=side_answer.confidence,
                    strength=strength(
                        subjects[i], confirm, magnitude, p_side, conditional, priced_in
                    ),
                )
            )

        return Reading(
            announcement=announcement,
            tokens=tokens,
            event_type=event.choice,
            p_event=event.p(event.choice),
            magnitude=magnitude,
            conditional=conditional,
            priced_in=priced_in,
            verdicts=verdicts,
            rounds=len(responses),
            latency_ms=sum(r.latency_ms for r in responses),
            wall_ms=(time.perf_counter() - started) * 1000.0,
            input_tokens=sum(r.input_tokens for r in responses),
            responses=responses,
        )
