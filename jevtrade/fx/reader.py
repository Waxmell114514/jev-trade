"""Read a central-bank document with the same two-round tree.

Round one is one request and asks everything that could matter: what kind of
text this is, whether it is policy at all, which way it leans for the issuer's
own currency, how big, whether the guidance moved, whether it is a surprise, how
far along the intervention ladder it sits -- and, for each sentence that changed
since the previous edition, which way *that sentence* cuts and whether the
change is material. With a full twelve-sentence diff that is 32 questions in one
400 ms round, which costs the same as one.

Round two runs only when round one found something: a decisive stance, a
material sentence change, or intervention language near the top of the ladder.
It re-asks the direction with the framing reversed -- "which side should a
trader be on", and "would a trader long this currency be unaffected" -- because
two rephrasings of one question make partly independent errors, and that is the
cheapest redundancy available. No second round, no trade.

Sign convention, stated once and tested: **hawkish for the issuer's own currency
means that currency strengthens.** Which way that pushes the *pair* depends on
which side of the pair the currency is quoted, which is what ``PAIRS`` is for; a
hawkish Fed sends EURUSD down, a hawkish BoJ sends USDJPY down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..types import JevResponse
from .documents import Document, local_string

TREE_VERSION = "v1"

KIND = "kind"
POLICY_RELEVANT = "policy_relevant"
STANCE = "stance"
NEW_INFORMATION = "new_information"
MAGNITUDE = "magnitude"
GUIDANCE_CHANGED = "guidance_changed"
SURPRISE = "surprise"
INTERVENTION_TIER = "intervention_tier"
HOLDER_UNAFFECTED = "holder_unaffected"
STANCE_REVERSED = "stance_reversed"
HORIZON = "horizon"

RATE_DECISION = "rate_decision"
MINUTES = "minutes"
SPEECH_OR_TESTIMONY = "speech_or_testimony"
PRESS_CONFERENCE = "press_conference_or_interview"
FX_COMMENT = "fx_or_intervention_comment"
DATA_OR_SURVEY = "data_or_survey"
OPERATIONAL = "operational_or_regulatory"
OTHER = "other"
KINDS = (
    RATE_DECISION, MINUTES, SPEECH_OR_TESTIMONY, PRESS_CONFERENCE,
    FX_COMMENT, DATA_OR_SURVEY, OPERATIONAL, OTHER,
)

HAWKISH, DOVISH, NEUTRAL = "hawkish", "dovish", "neutral"
BUY, SELL, NEITHER = "buy_the_currency", "sell_the_currency", "neither"
MINUTES_H, HOURS_H, DAYS_H = "minutes", "hours", "days_or_more"

MAGNITUDE_LEVELS = (
    "no effect on the exchange rate",
    "a small effect, the kind that fades within the hour",
    "a clear effect that traders would position for",
    "the biggest FX news of the day for this currency",
)

INTERVENTION_LEVELS = (
    "the text does not mention the exchange rate at all",
    "the exchange rate is mentioned, and officials say they are watching it",
    "the text calls recent moves excessive, one-sided, rapid or disorderly",
    "officials say they are ready to take decisive or appropriate action, "
    "or a rate check is reported",
    "an intervention is announced, confirmed, or reported as under way",
)

# Currency -> (spot symbol, +1 if the currency is the BASE of that pair).
# The sign a hawkish reading implies for the pair is this flip: a hawkish Fed
# strengthens USD, and USD is the *quote* side of EURUSD, so EURUSD falls.
PAIRS: dict[str, tuple[str, int]] = {
    "USD": ("EURUSD=X", -1),
    "EUR": ("EURUSD=X", +1),
    "GBP": ("GBPUSD=X", +1),
    "JPY": ("JPY=X", -1),
}

STANCE_SIGN = {HAWKISH: +1, DOVISH: -1, NEUTRAL: 0}


def pair_for(currency: str) -> tuple[str, int] | None:
    return PAIRS.get(currency)


def signed_pair(currency: str, stance: str) -> tuple[str, int] | None:
    """``(symbol, +1 long / -1 short)`` for a stance on a currency."""
    entry = PAIRS.get(currency)
    direction = STANCE_SIGN.get(stance, 0)
    if entry is None or direction == 0:
        return None
    symbol, flip = entry
    return symbol, flip * direction


def diff_stance_key(i: int) -> str:
    return f"diff_stance_{i}"


def diff_material_key(i: int) -> str:
    return f"diff_material_{i}"


def _noul(instructions: str, yes: str, no: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": yes, "false": no}}


# ------------------------------------------------------------------ round one


def round_one_questions(
    currency: str, changes: Sequence[tuple[str, str]]
) -> dict[str, dict[str, Any]]:
    money = currency or "the issuing central bank's currency"
    questions: dict[str, dict[str, Any]] = {
        KIND: {
            "type": "choice",
            "instructions": "What kind of official text is this?",
            "criteria": {
                RATE_DECISION: (
                    "A policy decision and the statement that carries it: a rate "
                    "set or held, a purchase programme changed, a policy guideline revised."
                ),
                MINUTES: "Minutes, an account, or a summary of opinions from a past meeting.",
                SPEECH_OR_TESTIMONY: "A prepared speech, lecture or testimony by an official.",
                PRESS_CONFERENCE: (
                    "A press conference, Q&A or interview: an official answering questions."
                ),
                FX_COMMENT: (
                    "It is mainly about the exchange rate itself -- its level, its "
                    "speed, or official action on it."
                ),
                DATA_OR_SURVEY: "A statistical release, survey or forecast publication.",
                OPERATIONAL: (
                    "Operational or regulatory housekeeping: supervision, enforcement, "
                    "payment systems, staff appointments, market operations mechanics."
                ),
                OTHER: "None of the above.",
            },
        },
        POLICY_RELEVANT: _noul(
            "Does this text carry information about the stance of monetary policy?",
            "A trader in this currency would want to read it before the next policy move.",
            "It says nothing about policy: it is administrative, statistical or ceremonial.",
        ),
        STANCE: {
            "type": "choice",
            "instructions": (
                f"Taken as a whole, which way does this text lean for {money}? "
                "Hawkish means tighter policy than the reader expected, or a "
                "greater willingness to tighten; dovish is the opposite."
            ),
            "criteria": {
                HAWKISH: (
                    "Tighter: higher rates, a longer hold, more concern about "
                    f"inflation, less tolerance for easing. Supportive of {money}."
                ),
                DOVISH: (
                    "Easier: cuts, a shorter hold, more concern about growth or "
                    f"employment, more willingness to ease. A weight on {money}."
                ),
                NEUTRAL: "It leans neither way, or the two sides are evenly balanced.",
            },
        },
        NEW_INFORMATION: _noul(
            "Is there anything here a reader of this bank's previous statements "
            "and speeches did not already know?",
            "It says something new, or says a known thing in a way that changes its weight.",
            "It restates positions this bank has already taken.",
        ),
        MAGNITUDE: {
            "type": "score",
            "instructions": f"How much would this text move {money} against the dollar?",
            "criteria": list(MAGNITUDE_LEVELS),
        },
        GUIDANCE_CHANGED: _noul(
            "Has the forward guidance changed -- what the bank says it will do "
            "next, or what it says it is waiting for?",
            "The conditions, the timing, or the direction of the next move have moved.",
            "The guidance is the same as before, in substance.",
        ),
        SURPRISE: _noul(
            "Is this different from what the text itself implies markets were "
            "expecting -- and, if a forecast and a previous value are given in "
            "the state, different from that forecast?",
            "A reader expecting the consensus would have to revise something.",
            "It is what was expected.",
        ),
        INTERVENTION_TIER: {
            "type": "score",
            "instructions": (
                "How far does this text go about the exchange rate itself? "
                "Judge only the language used, not whether it is justified."
            ),
            "criteria": list(INTERVENTION_LEVELS),
        },
    }
    for i, (was, now) in enumerate(changes):
        where = f"sentence {i + 1} of the changes listed in the state"
        questions[diff_stance_key(i)] = {
            "type": "choice",
            "instructions": (
                f"Compare the old and new versions of {where}. Which way does the "
                f"change cut for {money}?"
            ),
            "criteria": {
                HAWKISH: "The new wording is tighter, firmer or more worried about inflation.",
                DOVISH: "The new wording is softer, easier or more worried about growth.",
                NEUTRAL: "The change does not lean either way.",
            },
        }
        questions[diff_material_key(i)] = _noul(
            f"Is the change in {where} a change of substance?",
            "A reader would update what they expect the bank to do.",
            "It is a rephrasing, a date, a number that had to change anyway, or housekeeping.",
        )
    return questions


def round_two_questions(currency: str) -> dict[str, dict[str, Any]]:
    money = currency or "this currency"
    return {
        HOLDER_UNAFFECTED: _noul(
            f"Would a trader who is long {money} have no reason to change their "
            "position after reading this text?",
            f"Nothing here changes the case for holding {money}.",
            f"A {money} holder would want to act on this.",
        ),
        STANCE_REVERSED: {
            "type": "choice",
            "instructions": (
                f"Forget the question of tone. If you had to take a position in "
                f"{money} for the next hour on this text alone, which side?"
            ),
            "criteria": {
                BUY: f"Buy {money}.",
                SELL: f"Sell {money}.",
                NEITHER: "Neither side: there is nothing here to trade.",
            },
        },
        HORIZON: {
            "type": "choice",
            "instructions": "Over what horizon would this text move the exchange rate?",
            "criteria": {
                MINUTES_H: "Minutes: it is priced almost at once and then done.",
                HOURS_H: "Hours: it takes a session to be read and absorbed.",
                DAYS_H: "Days or more: it changes the path, not the level.",
            },
        },
    }


def build_state(
    document: Document,
    changes: Sequence[tuple[str, str]],
    *,
    round_no: int,
    body_chars: int,
    previous: Document | None = None,
    calendar: dict[str, Any] | None = None,
    so_far: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "issuer": document.issuer.upper(),
        "currency": document.currency,
        "feed_kind": document.kind,
        "title": document.title,
        "speaker": document.speaker,
        "published_utc": document.when.strftime("%Y-%m-%d %H:%M UTC"),
        "published_local": local_string(document.ts, document.issuer),
        "body": (document.body or "")[:body_chars],
        "round": round_no,
    }
    if previous is not None:
        state["compared_with"] = {
            "title": previous.title,
            "published_utc": previous.when.strftime("%Y-%m-%d %H:%M UTC"),
        }
    if changes:
        state["changed_sentences"] = [
            {"i": i + 1, "was": was, "now": now} for i, (was, now) in enumerate(changes)
        ]
    if calendar:
        state["calendar"] = {
            "event": calendar.get("title", ""),
            "impact": calendar.get("impact", ""),
            "forecast": calendar.get("forecast", ""),
            "previous": calendar.get("previous", ""),
        }
    if so_far:
        state["first_round_verdicts"] = so_far
    return state


# ------------------------------------------------------------------ verdicts


def strength(
    p_stance: float,
    confirm: float,
    magnitude: float,
    relevant: float,
    surprise: float,
    priced_in: float,
) -> float:
    """Arithmetic on six probabilities; the model multiplies nothing.

    ``priced_in`` is the complement of ``new_information`` and enters once, as a
    damper: a statement that restates known positions is halved, not cancelled,
    because restating a position at a moment when the market doubted it is
    itself information.
    """
    value = p_stance * confirm * magnitude * relevant
    value *= 0.7 + 0.3 * surprise
    value *= 1.0 - 0.5 * priced_in
    return max(0.0, min(1.0, value))


@dataclass
class DiffVerdict:
    index: int
    was: str
    now: str
    stance: str
    p_stance: float
    material: float

    @property
    def sign(self) -> int:
        return STANCE_SIGN.get(self.stance, 0)


@dataclass
class Verdict:
    pair: str
    sign: int  # +1 long the pair, -1 short it
    p_side: float
    confidence: float
    strength: float


@dataclass
class Reading:
    document: Document
    kind: str
    p_kind: float
    policy_relevant: float
    stance: str
    p_stance: float
    confidence: float
    new_information: float
    magnitude: float
    guidance_changed: float
    surprise: float
    intervention: float
    confirm: float
    horizon: str
    diffs: list[DiffVerdict]
    verdict: Verdict | None
    rounds: int
    latency_ms: float
    wall_ms: float
    input_tokens: int
    questions_asked: int = 0
    responses: list[JevResponse] = field(default_factory=list, repr=False)

    @property
    def material_diffs(self) -> int:
        return sum(1 for d in self.diffs if d.material >= 0.5)

    def signals(self, threshold: float, *, min_confidence: float = 0.5) -> list[Verdict]:
        verdict = self.verdict
        if verdict is None or verdict.sign == 0:
            return []
        if verdict.strength <= 0.0 or verdict.strength < threshold:
            return []
        if verdict.confidence < min_confidence:
            return []
        return [verdict]


class Reader:
    def __init__(
        self,
        client,
        *,
        body_chars: int = 6000,
        max_diffs: int = 12,
        stance_floor: float = 0.5,
        material_floor: float = 0.5,
        intervention_floor: float = 0.6,
    ) -> None:
        self.client = client
        self.body_chars = body_chars
        self.max_diffs = max_diffs
        self.stance_floor = stance_floor
        self.material_floor = material_floor
        self.intervention_floor = intervention_floor

    def _ask(self, questions: dict[str, Any], state: dict[str, Any]) -> JevResponse:
        self.client.questions = questions
        return self.client.evaluate(state)

    def read(
        self,
        document: Document,
        *,
        previous: Document | None = None,
        changes: Sequence[tuple[str, str]] = (),
        calendar: dict[str, Any] | None = None,
    ) -> Reading:
        started = time.perf_counter()
        changes = list(changes)[: self.max_diffs]
        currency = document.currency

        questions = round_one_questions(currency, changes)
        first = self._ask(
            questions,
            build_state(
                document, changes, round_no=1, body_chars=self.body_chars,
                previous=previous, calendar=calendar,
            ),
        )
        kind = first.choice(KIND)
        stance = first.choice(STANCE)
        relevant = first.noul(POLICY_RELEVANT).noul
        new_information = first.noul(NEW_INFORMATION).noul
        magnitude = first.score(MAGNITUDE).normalized
        guidance = first.noul(GUIDANCE_CHANGED).noul
        surprise = first.noul(SURPRISE).noul
        intervention = first.score(INTERVENTION_TIER).normalized
        diffs: list[DiffVerdict] = []
        for i, (was, now) in enumerate(changes):
            answer = first.choice(diff_stance_key(i))
            diffs.append(
                DiffVerdict(
                    index=i,
                    was=was,
                    now=now,
                    stance=answer.choice,
                    p_stance=answer.p(answer.choice),
                    material=first.noul(diff_material_key(i)).noul,
                )
            )
        responses = [first]
        asked = len(questions)

        decisive = stance.choice != NEUTRAL and stance.p(stance.choice) >= self.stance_floor
        material = any(d.material >= self.material_floor for d in diffs)
        loud = intervention >= self.intervention_floor
        confirm = 0.0
        horizon = ""
        if decisive or material or loud:
            so_far = {
                "kind": kind.choice,
                "stance": stance.choice,
                "policy_relevant": round(relevant, 3),
                "material_sentence_changes": sum(
                    1 for d in diffs if d.material >= self.material_floor
                ),
                "intervention_tier": round(intervention, 3),
            }
            second_questions = round_two_questions(currency)
            second = self._ask(
                second_questions,
                build_state(
                    document, changes, round_no=2, body_chars=self.body_chars,
                    previous=previous, calendar=calendar, so_far=so_far,
                ),
            )
            responses.append(second)
            asked += len(second_questions)
            side = second.choice(STANCE_REVERSED)
            want = {HAWKISH: BUY, DOVISH: SELL}.get(stance.choice)
            agree = side.p(want) if want else 0.0
            confirm = (1.0 - second.noul(HOLDER_UNAFFECTED).noul) * agree
            horizon = second.choice(HORIZON).choice

        verdict: Verdict | None = None
        signed = signed_pair(currency, stance.choice)
        if signed is not None:
            symbol, sign = signed
            p_side = stance.p(stance.choice)
            verdict = Verdict(
                pair=symbol,
                sign=sign,
                p_side=p_side,
                confidence=stance.confidence,
                strength=strength(
                    p_side, confirm, magnitude, relevant, surprise,
                    1.0 - new_information,
                ),
            )

        return Reading(
            document=document,
            kind=kind.choice,
            p_kind=kind.p(kind.choice),
            policy_relevant=relevant,
            stance=stance.choice,
            p_stance=stance.p(stance.choice),
            confidence=stance.confidence,
            new_information=new_information,
            magnitude=magnitude,
            guidance_changed=guidance,
            surprise=surprise,
            intervention=intervention,
            confirm=confirm,
            horizon=horizon,
            diffs=diffs,
            verdict=verdict,
            rounds=len(responses),
            latency_ms=sum(r.latency_ms for r in responses),
            wall_ms=(time.perf_counter() - started) * 1000.0,
            input_tokens=sum(r.input_tokens for r in responses),
            questions_asked=asked,
            responses=responses,
        )


__all__ = [
    "DiffVerdict", "KINDS", "MAGNITUDE_LEVELS", "INTERVENTION_LEVELS", "PAIRS",
    "Reader", "Reading", "TREE_VERSION", "Verdict", "build_state", "pair_for",
    "round_one_questions", "round_two_questions", "signed_pair", "strength",
]
