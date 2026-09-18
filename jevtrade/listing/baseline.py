"""The incumbent: a bot that matches the title and buys every ticker in it.

This is what public "Binance listing sniper" bots do, and it is fast enough --
it polls the same endpoint and fires in well under a second. Its limits are
the ones reading fixes: it cannot tell a listing from a notice that mentions
one, cannot tell which of three tickers is the subject, and reads "Will Remove
the Seed Tag from X" as a removal.

``body_bot`` is the control that separates *seeing the body* from *reading
it*: the same rules, applied to tickers from the body too. If body_bot matches
the reader, the win was extraction, not judgment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .announcements import Announcement
from .tickers import candidates

BEARISH = re.compile(
    r"\b(delist\w*|removal|remove\w*|monitoring tag|suspend\w*|cease\w*|"
    r"discontinu\w*|terminat\w*)\b",
    re.I,
)
BULLISH = re.compile(
    r"\b(will list|lists?|will add|adds?|will launch|launch\w*|will support|"
    r"supports?|now available)\b",
    re.I,
)


@dataclass(frozen=True)
class BotSignal:
    token: str
    sign: int  # +1 long, -1 short


def side_from_title(title: str) -> int:
    if BEARISH.search(title):
        return -1
    if BULLISH.search(title):
        return 1
    return 0


def title_bot(announcement: Announcement) -> list[BotSignal]:
    sign = side_from_title(announcement.title)
    if sign == 0:
        return []
    return [BotSignal(t, sign) for t in candidates(announcement.title)]


def body_bot(announcement: Announcement) -> list[BotSignal]:
    sign = side_from_title(announcement.title)
    if sign == 0:
        return []
    return [BotSignal(t, sign) for t in candidates(announcement.title, announcement.body)]
