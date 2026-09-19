"""The two incumbents: counting words, and comparing a number to a forecast.

``keyword_bot`` is what a cheap headline scraper does and what most "central
bank sentiment" dashboards still do: count hawkish words against dovish ones and
trade the difference. It is fast, free, and it reads "the Committee no longer
expects to raise rates and will not tighten further" as hawkish, because the two
words it knows are both in there. If the reader cannot beat it, the reader is
not buying anything.

``surprise_bot`` is the *other* incumbent, the one that actually wins on numeric
releases: take the printed number, subtract the consensus forecast, trade the
sign. It needs a forecast, and a forecast can only come from a calendar snapshot
taken *before* the event. ForexFactory serves this week only, so this arm is
live for the weeks somebody ran ``--snapshot-calendar`` and reports "not
available" for every other week. It is never back-filled: a consensus
reconstructed after the fact is not a consensus, and an arm that quietly invents
one would beat everything here for the wrong reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .documents import Document, match_calendar
from .mock import DOVISH_WORDS, HAWKISH_WORDS
from .reader import DOVISH, HAWKISH, signed_pair

# Titles that name a rate decision. The surprise bot refuses everything else:
# "CPI y/y" has a forecast too, but the Fed does not publish CPI.
RATE_TITLE = re.compile(
    r"fomc statement|federal funds rate|monetary policy statement|policy rate|"
    r"official bank rate|bank rate|monetary policy summary|"
    r"guideline for money market operations|interest rate decision",
    re.I,
)
# "3-3/4 to 4 percent", "4.00%", "0.75 per cent"
_FRACTION = re.compile(r"(\d+)-(\d+)/(\d+)")
_NUMBER = re.compile(r"\d+-\d+/\d+|\d+/\d+|\d+(?:\.\d+)?")
_PERCENT = re.compile(r"(\d+(?:-\d+/\d+)?(?:\.\d+)?)\s*(?:percent|per cent|%)", re.I)
_TARGET = re.compile(
    r"(?:target range|target rate|federal funds rate|bank rate|policy rate|"
    r"uncollateralized overnight call rate)[^.]{0,160}?\bto\s+([^.]{0,60}?)"
    r"(?:percent|per cent|%)",
    re.I,
)


@dataclass(frozen=True)
class BotSignal:
    pair: str
    sign: int  # +1 long the pair, -1 short it
    note: str = ""


def _is_percent(text: Any) -> bool:
    """A calendar cell that is a rate ("4.00%", "<1.25%"), not a vote ("3-0-6")."""
    return isinstance(text, str) and "%" in text and not re.search(r"\d+-\d+-\d+", text)


def _value(text: str) -> float | None:
    """A percentage written as ``4``, ``4.00`` or ``3-3/4`` -> a float."""
    text = (text or "").strip().rstrip("%").strip()
    match = _FRACTION.search(text)
    if match:
        whole, num, den = (int(g) for g in match.groups())
        return whole + num / den if den else float(whole)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _range_top(text: str) -> float | None:
    """The top of a written range: "3-3/4 to 4" is 4, "3-1/2 to 3-3/4" is 3.75."""
    values = [v for v in (_value(m) for m in _NUMBER.findall(text or "")) if v is not None]
    return max(values) if values else None


def word_lean(text: str) -> tuple[str, int, int]:
    """``(stance, hawkish hits, dovish hits)`` -- the whole of the incumbent."""
    hawks = len(HAWKISH_WORDS.findall(text or ""))
    doves = len(DOVISH_WORDS.findall(text or ""))
    if hawks == doves:
        return "neutral", hawks, doves
    return (HAWKISH if hawks > doves else DOVISH), hawks, doves


def keyword_bot(document: Document) -> list[BotSignal]:
    """One signal on the issuer's pair, from hawkish minus dovish word counts."""
    stance, hawks, doves = word_lean(f"{document.title}\n{document.body}")
    signed = signed_pair(document.currency, stance)
    if signed is None:
        return []
    pair, sign = signed
    return [BotSignal(pair, sign, note=f"{stance} {hawks}-{doves}")]


_RANGE = r"\d+(?:-\d+/\d+)?(?:\.\d+)?(?:\s+to\s+\d+(?:-\d+/\d+)?(?:\.\d+)?)?"
# "Bank rate maintained at 3.75%", "raises Bank Rate to 4%"
_TITLE_RATE = re.compile(r"\b(?:at|to)\s+(" + _RANGE + r")\s*(?:percent|per cent|%)", re.I)
# "decided to raise the target range for the federal funds rate by 1/4 percentage
# point to 3-3/4 to 4 percent", "voted ... to maintain Bank Rate at 3.75%"
_DECISION = re.compile(
    r"\b(?:maintain|maintained|maintains|keep|kept|keeps|hold|held|holds|raise|raised|raises|"
    r"increase|increased|increases|lower|lowered|lowers|cut|cuts|reduce|reduced|reduces)\s+"
    r"(?:the\s+)?(?:target range for the federal funds rate|federal funds rate|bank rate|"
    r"policy rate|uncollateralized overnight call rate|deposit facility rate|cash rate|"
    r"official cash rate|policy interest rate|overnight rate target|overnight rate)\s+"
    r"(?:unchanged\s+)?(?:at|to|by\s+[^.]{0,40}?\bto)\s+(" + _RANGE + r")\s*(?:percent|per cent|%)",
    re.I,
)


def announced_rate(text: str) -> float | None:
    """The rate the text says was set, in percent, or ``None``.

    The top of a target range is used, because that is what the calendar quotes
    (the September FOMC "3-3/4 to 4 percent" is 4.00% on ForexFactory). The title
    is read first ("Bank rate maintained at 3.75%"), then the decision sentence
    (a verb, the rate's name, "at" or "to", the number); a page full of other
    percentages -- the 2% target, CPI, vote shares -- is not allowed to answer.
    """
    title, _, body = (text or "").partition("\n")
    match = _TITLE_RATE.search(title)
    if match:
        value = _range_top(match.group(1))
        if value is not None:
            return value
    match = _DECISION.search(body) or _DECISION.search(title)
    if match:
        value = _range_top(match.group(1))
        if value is not None:
            return value
    match = _TARGET.search(text or "")
    if match:
        value = _range_top(match.group(1))
        if value is not None:
            return value
    found = _PERCENT.findall(text or "")
    if len(found) == 1:
        return _value(found[0])
    return None


def surprise_bot(
    document: Document, rows: Sequence[dict[str, Any]]
) -> list[BotSignal]:
    """Trade the printed rate against the snapshotted forecast, or nothing."""
    if not RATE_TITLE.search(document.title or ""):
        return []
    row = match_calendar(document, list(rows))
    if row is None or not _is_percent(row.get("forecast")):
        return []
    forecast = _value(row["forecast"])
    actual = announced_rate(f"{document.title}\n{document.body}")
    if forecast is None or actual is None or abs(actual - forecast) < 1e-9:
        return []
    stance = HAWKISH if actual > forecast else DOVISH
    signed = signed_pair(document.currency, stance)
    if signed is None:
        return []
    pair, sign = signed
    return [BotSignal(pair, sign, note=f"{actual:g} vs forecast {forecast:g}")]


__all__ = ["BotSignal", "RATE_TITLE", "announced_rate", "keyword_bot", "surprise_bot", "word_lean"]
