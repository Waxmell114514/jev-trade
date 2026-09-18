"""What we ask Jev about a headline.

A market maker does not need to know where the price is going. It needs to know
whether the next minute is a bad time to be showing a particular side. That is
a defensive, low-cardinality judgment, which is the shape a System One model is
good at -- and, unlike directional alpha, it is answerable from text.

The two questions that earn their place are ``new_information`` and
``report_type``. They exist because of the two headlines a keyword rule cannot
read: a re-report of old news, and a *denial*. jev-1.13 answers the question as
literally written, so the denial case is asked literally -- "does this report
that the thing happened, or that it did not?" -- rather than hoping a question
about severity implies it.
"""

from __future__ import annotations

from typing import Any

MOVES_PRICE = "moves_price"
DIRECTION = "direction"
SEVERITY = "severity"
NEW_INFORMATION = "new_information"
REPORT_TYPE = "report_type"
INFORMED_FLOW = "informed_flow"

UP, DOWN, UNCLEAR = "up", "down", "unclear"
HAPPENED, DENIED, NEITHER = "happened", "denied", "neither"

SEVERITY_LEVELS = [
    "No move. The headline is commentary, opinion or routine housekeeping.",
    "Small move. Mildly relevant, the kind of item that moves price briefly.",
    "Notable move. Directly affects the asset's supply, demand or access.",
    "Large move. Changes whether people can trade, hold or redeem the asset.",
]

QUESTION_IDS = (MOVES_PRICE, DIRECTION, SEVERITY, NEW_INFORMATION,
                REPORT_TYPE, INFORMED_FLOW)


def headline_questions() -> dict[str, dict[str, Any]]:
    return {
        MOVES_PRICE: {
            "type": "noul",
            "instructions": (
                "Will the price of this instrument move by a meaningful amount "
                "within the next minute as a direct result of this headline?"
            ),
            "criteria": {
                "true": "Traders who read this headline would reprice the asset.",
                "false": (
                    "Traders who read this headline would not change what they "
                    "are willing to pay."
                ),
            },
        },
        DIRECTION: {
            "type": "choice",
            "instructions": (
                "If the price moves because of this headline, which way does it "
                "move?"
            ),
            "criteria": {
                UP: "The headline is good news for the asset's price.",
                DOWN: "The headline is bad news for the asset's price.",
                UNCLEAR: (
                    "The headline gives no basis for choosing a direction, or "
                    "it is not news about this asset at all."
                ),
            },
        },
        SEVERITY: {
            "type": "score",
            "instructions": "How large a price move would this headline justify?",
            "criteria": SEVERITY_LEVELS,
        },
        NEW_INFORMATION: {
            "type": "noul",
            "instructions": (
                "Does this headline report something that is happening now or "
                "has just been disclosed, rather than restating something "
                "already reported earlier?"
            ),
            "criteria": {
                "true": "The information in it is new.",
                "false": (
                    "It revisits, recaps, explains or republishes something "
                    "that was already public."
                ),
            },
        },
        REPORT_TYPE: {
            "type": "choice",
            "instructions": (
                "Does this headline report that the event described in it "
                "happened, or report that it did not happen?"
            ),
            "criteria": {
                HAPPENED: "It reports the event as having occurred.",
                DENIED: (
                    "It reports a denial, a rejection, or that the event did "
                    "not occur and was not true."
                ),
                NEITHER: (
                    "It describes no specific event, only commentary or "
                    "background."
                ),
            },
        },
        INFORMED_FLOW: {
            "type": "noul",
            "instructions": (
                "Does the trading described under recent_trading look like "
                "people acting on information, rather than ordinary two-way "
                "business?"
            ),
            "criteria": {
                "true": (
                    "Buying and selling have become one-sided and persistent, "
                    "the way flow looks when one side knows something."
                ),
                "false": (
                    "Buying and selling are roughly balanced, or the activity "
                    "is unremarkable."
                ),
            },
        },
    }


def build_state(
    *,
    headline: str,
    instrument: str,
    inventory: str,
    recent_trading: str,
    trigger: str,
) -> dict[str, Any]:
    """Small on purpose: the headline is the decision, the rest is context.

    One request shape serves both triggers -- a headline arriving, and the tape
    turning one-sided. Whichever fired, the other field is still filled in,
    because the two together are more informative than either alone: one-sided
    flow right after a denial means something different than one-sided flow out
    of nowhere.
    """
    return {
        "instrument": instrument,
        "headline": headline,
        "recent_trading": recent_trading,
        "our_book": {"inventory": inventory},
        "why_we_are_asking": trigger,
    }
