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

There are two modes. The **absolute** one is the tree above: read the text, say
which way it leans. The **context** one adds the question the absolute tree
cannot ask -- *more hawkish than what?* -- because it is handed the pre-release
context ``fx/context.py`` assembles and asks for the stance **relative** to it:
what the market expected, what the statement did, and which way the difference
cuts. The sign of a context signal comes from ``relative_stance`` and never from
``stance``: on 2024-12-18 the statement was a cut and read dovish, and the
market took it as hawkish, which is a disagreement the absolute tree has no
vocabulary for.

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
# The context tree is a different set of questions over a different state, so it
# gets its own cache tag and the absolute reader's ``v1`` answers stay reusable.
CONTEXT_VERSION = "ctx1"
ABSOLUTE, CONTEXT = "absolute", "context"
MODES = (ABSOLUTE, CONTEXT)
TREE_VERSIONS = {ABSOLUTE: TREE_VERSION, CONTEXT: CONTEXT_VERSION}


def tree_version(mode: str = ABSOLUTE) -> str:
    return TREE_VERSIONS.get(mode, TREE_VERSION)


KIND = "kind"
POLICY_RELEVANT = "policy_relevant"
STANCE = "stance"
NEW_INFORMATION = "new_information"
MAGNITUDE = "magnitude"
GUIDANCE_CHANGED = "guidance_changed"
SURPRISE = "surprise"
INTERVENTION_TIER = "intervention_tier"
EXPECTED_ACTION = "expected_action"
ACTUAL_ACTION = "actual_action"
RELATIVE_STANCE = "relative_stance"
SURPRISE_CHANNEL = "surprise_channel"
SURPRISE_SIZE = "surprise_size"
VERSUS_MINUTES = "versus_minutes"
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

HIKE, HOLD, CUT = "hike", "hold", "cut"
ACTIONS = (HIKE, HOLD, CUT)
# How far up the ladder an action sits, so "more hawkish than expected" is a
# comparison code can make and not a judgment the model has to repeat.
ACTION_RANK = {CUT: -1, HOLD: 0, HIKE: +1}

MORE_HAWKISH = "more_hawkish_than_expected"
IN_LINE = "in_line_with_expectations"
MORE_DOVISH = "more_dovish_than_expected"
RELATIVE_OPTIONS = (MORE_HAWKISH, IN_LINE, MORE_DOVISH)
# The relative call is what a context signal is signed by. In-line is flat.
RELATIVE_STANCES = {MORE_HAWKISH: HAWKISH, MORE_DOVISH: DOVISH, IN_LINE: NEUTRAL}

RATE_CHANNEL = "rate_decision"
GUIDANCE_CHANNEL = "forward_guidance"
BALANCE_SHEET_CHANNEL = "balance_sheet"
DOTS_CHANNEL = "projections_or_dots"
VOTE_CHANNEL = "vote_or_dissent"
ASSESSMENT_CHANNEL = "economic_assessment"
NO_CHANNEL = "nothing_surprising"
CHANNELS = (
    RATE_CHANNEL, GUIDANCE_CHANNEL, BALANCE_SHEET_CHANNEL, DOTS_CHANNEL,
    VOTE_CHANNEL, ASSESSMENT_CHANNEL, NO_CHANNEL,
)

VS_MORE_HAWKISH, VS_CONSISTENT, VS_MORE_DOVISH = "more_hawkish", "consistent", "more_dovish"
VERSUS_OPTIONS = (VS_MORE_HAWKISH, VS_CONSISTENT, VS_MORE_DOVISH)

SURPRISE_LEVELS = (
    "nothing the market did not already have",
    "a nuance desks will argue about",
    "a clear surprise traders reposition on",
    "the kind of surprise that sets the day",
)

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

# The same pairs as Dukascopy names them, for the tick judge. Yahoo's "JPY=X"
# is USDJPY, which is the one place the two vocabularies disagree about which
# currency is the base, so the mapping is spelled out rather than derived.
TICK_SYMBOLS: dict[str, str] = {
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "JPY=X": "USDJPY",
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


def context_questions(currency: str) -> dict[str, dict[str, Any]]:
    """The six relative questions: expected, actual, and the gap between them.

    Every one of these is answered *against the context block in the state* and
    is meaningless without it. ``expected_action`` and ``actual_action`` are
    asked separately, and separately from ``relative_stance``, because code can
    check the second against the rate it parsed out of the statement -- which
    turns "the model cannot read" and "the model read the market wrong" into two
    different findings instead of one shrug.
    """
    money = currency or "the issuing central bank's currency"
    return {
        EXPECTED_ACTION: {
            "type": "choice",
            "instructions": (
                "Read only the context block, not the statement. What did the market "
                "expect this meeting to do with the policy rate?"
            ),
            "criteria": {
                HIKE: "Raise the target range at this meeting.",
                HOLD: "Leave the target range where it is at this meeting.",
                CUT: "Lower the target range at this meeting.",
            },
        },
        ACTUAL_ACTION: {
            "type": "choice",
            "instructions": "What did this statement actually do with the policy rate?",
            "criteria": {
                HIKE: "It raised the target range.",
                HOLD: "It left the target range unchanged.",
                CUT: "It lowered the target range.",
            },
        },
        RELATIVE_STANCE: {
            "type": "choice",
            "instructions": (
                f"Against the context -- what was priced, what the projections did, what "
                f"the last statement and the minutes said, what officials said between the "
                f"meetings -- is this statement more hawkish or more dovish than that "
                f"context implies, for {money}? Judge the statement relative to the "
                "expectation, not on its own."
            ),
            "criteria": {
                MORE_HAWKISH: (
                    f"Tighter than the context implies: a reader holding those "
                    f"expectations has to revise towards higher rates. Supportive of {money}."
                ),
                IN_LINE: "It is what that context implies. Nothing to revise.",
                MORE_DOVISH: (
                    f"Easier than the context implies: a reader holding those "
                    f"expectations has to revise towards lower rates. A weight on {money}."
                ),
            },
        },
        SURPRISE_CHANNEL: {
            "type": "choice",
            "instructions": "Where does the difference from the context sit, if anywhere?",
            "criteria": {
                RATE_CHANNEL: "The rate decision itself was not the one the context implied.",
                GUIDANCE_CHANNEL: (
                    "What the Committee says it will do next, or what it says it is "
                    "waiting for, moved."
                ),
                BALANCE_SHEET_CHANNEL: (
                    "The balance sheet: purchases, runoff, reinvestment, holdings."
                ),
                DOTS_CHANNEL: (
                    "The projections released with the statement: the median path, the "
                    "dots, the forecasts."
                ),
                VOTE_CHANNEL: "The vote: a dissent, a new dissenter, a unanimity that broke.",
                ASSESSMENT_CHANNEL: (
                    "The description of the economy: growth, the labour market, inflation, "
                    "the balance of risks."
                ),
                NO_CHANNEL: "Nothing here differs from what the context implied.",
            },
        },
        SURPRISE_SIZE: {
            "type": "score",
            "instructions": (
                "How big is the gap between this statement and what the context implied?"
            ),
            "criteria": list(SURPRISE_LEVELS),
        },
        VERSUS_MINUTES: {
            "type": "choice",
            "instructions": (
                "Set the statement against the minutes and the intermeeting speeches in "
                "the context. Which way has the Committee moved since it was last heard?"
            ),
            "criteria": {
                VS_MORE_HAWKISH: "The statement is firmer than the minutes and the speeches were.",
                VS_CONSISTENT: "It says what those already said.",
                VS_MORE_DOVISH: "The statement is softer than the minutes and the speeches were.",
            },
        },
    }


def round_one_questions(
    currency: str, changes: Sequence[tuple[str, str]], *, mode: str = ABSOLUTE
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
    if mode == CONTEXT:
        # Width is free, so the context mode carries the whole absolute tree and
        # adds to it; the two readings are then comparable question by question.
        questions.update(context_questions(currency))
    return questions


def round_two_questions(currency: str, *, mode: str = ABSOLUTE) -> dict[str, dict[str, Any]]:
    money = currency or "this currency"
    if mode == CONTEXT:
        return {
            HOLDER_UNAFFECTED: _noul(
                f"Would a trader who is long {money} and had already read the context "
                "have no reason to change their position after this statement?",
                f"The context already had it; nothing here changes the case for holding {money}.",
                f"A {money} holder who had read the context would want to act on this.",
            ),
            STANCE_REVERSED: {
                "type": "choice",
                "instructions": (
                    f"Forget the question of tone. A desk that had read this context and "
                    f"then read this statement: which side of {money} for the next hour?"
                ),
                "criteria": {
                    BUY: f"Buy {money}: the statement is tighter than the desk was positioned for.",
                    SELL: (f"Sell {money}: the statement is easier than the desk "
                           "was positioned for."),
                    NEITHER: "Neither side: the desk had this already.",
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
    context: str = "",
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
    if context:
        # Everything the market already had, assembled by ``fx/context.py`` and
        # asserted there to predate the release. It goes in as one block of
        # prose because that is what it is: sentences, not features.
        state["context_before_the_release"] = context
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


def context_strength(
    p_relative: float, confirm: float, surprise: float, priced_in: float
) -> float:
    """Arithmetic on four probabilities; the model multiplies nothing.

    The absolute tree's ``strength`` dampens a fully-priced-in reading by half,
    because restating a known position at a moment of doubt is itself news. The
    context tree does not: the whole point of handing the reader the
    expectation is that "the market already had this" is a complete answer, and
    a statement that adds nothing to the context should not be traded at all.
    """
    value = p_relative * confirm * surprise * (1.0 - priced_in)
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
    # Context mode only; empty strings and zeros on an absolute reading.
    mode: str = ABSOLUTE
    expected_action: str = ""
    p_expected: float = 0.0
    actual_action: str = ""
    p_actual: float = 0.0
    relative: str = ""
    p_relative: float = 0.0
    surprise_channel: str = ""
    p_channel: float = 0.0
    surprise_size: float = 0.0
    versus_minutes: str = ""
    context_chars: int = 0
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
        surprise_floor: float = 0.6,
        mode: str = ABSOLUTE,
    ) -> None:
        self.client = client
        self.body_chars = body_chars
        self.max_diffs = max_diffs
        self.stance_floor = stance_floor
        self.material_floor = material_floor
        self.intervention_floor = intervention_floor
        # Context mode only: how surprising round one has to find a statement
        # for round two to be worth a request on its own.
        self.surprise_floor = surprise_floor
        self.mode = mode if mode in MODES else ABSOLUTE

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
        context: str = "",
    ) -> Reading:
        started = time.perf_counter()
        changes = list(changes)[: self.max_diffs]
        currency = document.currency

        questions = round_one_questions(currency, changes, mode=self.mode)
        first = self._ask(
            questions,
            build_state(
                document, changes, round_no=1, body_chars=self.body_chars,
                previous=previous, calendar=calendar, context=context,
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

        # The context answers, or blanks when this is the absolute tree.
        expected = actual = relative = channel = versus = ""
        p_expected = p_actual = p_relative = p_channel = surprise_size = 0.0
        relative_confidence = 0.0
        if self.mode == CONTEXT:
            expected_answer = first.choice(EXPECTED_ACTION)
            actual_answer = first.choice(ACTUAL_ACTION)
            relative_answer = first.choice(RELATIVE_STANCE)
            channel_answer = first.choice(SURPRISE_CHANNEL)
            expected = expected_answer.choice
            p_expected = expected_answer.p(expected)
            actual = actual_answer.choice
            p_actual = actual_answer.p(actual)
            relative = relative_answer.choice
            p_relative = relative_answer.p(relative)
            relative_confidence = relative_answer.confidence
            channel = channel_answer.choice
            p_channel = channel_answer.p(channel)
            surprise_size = first.score(SURPRISE_SIZE).normalized
            versus = first.choice(VERSUS_MINUTES).choice
            # The relative call is the whole point, so it alone opens round two
            # -- with the size as a second door, because a statement the model
            # calls in-line but enormous is exactly the case worth a re-ask.
            decisive = relative != IN_LINE and p_relative >= self.stance_floor
            material = surprise_size >= self.surprise_floor
            loud = False
        else:
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
            if self.mode == CONTEXT:
                so_far.update({
                    "expected_action": expected, "actual_action": actual,
                    "relative_stance": relative, "surprise_channel": channel,
                    "surprise_size": round(surprise_size, 3),
                })
            second_questions = round_two_questions(currency, mode=self.mode)
            second = self._ask(
                second_questions,
                build_state(
                    document, changes, round_no=2, body_chars=self.body_chars,
                    previous=previous, calendar=calendar, so_far=so_far, context=context,
                ),
            )
            responses.append(second)
            asked += len(second_questions)
            side = second.choice(STANCE_REVERSED)
            lean = (RELATIVE_STANCES.get(relative, NEUTRAL) if self.mode == CONTEXT
                    else stance.choice)
            want = {HAWKISH: BUY, DOVISH: SELL}.get(lean)
            agree = side.p(want) if want else 0.0
            confirm = (1.0 - second.noul(HOLDER_UNAFFECTED).noul) * agree
            horizon = second.choice(HORIZON).choice

        verdict: Verdict | None = None
        if self.mode == CONTEXT:
            signed = signed_pair(currency, RELATIVE_STANCES.get(relative, NEUTRAL))
            if signed is not None:
                symbol, sign = signed
                verdict = Verdict(
                    pair=symbol,
                    sign=sign,
                    p_side=p_relative,
                    confidence=relative_confidence,
                    strength=context_strength(
                        p_relative, confirm, surprise_size, 1.0 - new_information,
                    ),
                )
        else:
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
            mode=self.mode,
            expected_action=expected,
            p_expected=p_expected,
            actual_action=actual,
            p_actual=p_actual,
            relative=relative,
            p_relative=p_relative,
            surprise_channel=channel,
            p_channel=p_channel,
            surprise_size=surprise_size,
            versus_minutes=versus,
            context_chars=len(context),
            responses=responses,
        )


__all__ = [
    "ABSOLUTE", "ACTIONS", "ACTION_RANK", "CHANNELS", "CONTEXT", "CONTEXT_VERSION",
    "DiffVerdict", "KINDS", "MAGNITUDE_LEVELS", "INTERVENTION_LEVELS", "MODES", "PAIRS",
    "RELATIVE_OPTIONS", "RELATIVE_STANCES", "Reader", "Reading", "SURPRISE_LEVELS",
    "TICK_SYMBOLS", "TREE_VERSION", "TREE_VERSIONS", "VERSUS_OPTIONS", "Verdict",
    "build_state", "context_questions", "context_strength", "pair_for",
    "round_one_questions", "round_two_questions", "signed_pair", "strength",
    "tree_version",
]
