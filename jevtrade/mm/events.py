"""Synthetic news events.

The point of this module is to build a world in which **semantics decide the
answer**, because that is the hypothesis under test: that there are moments a
market maker must react to where a keyword rule gets it wrong and judgment gets
it right.

That construction is also the honest caveat. Five event kinds:

* ``material_up`` / ``material_down`` -- real news, the tape really moves.
* ``noise``      -- newsy-sounding, no impact. A jumpy rule over-reacts.
* ``denial``     -- "Exchange denies reports of a withdrawal halt". Contains
                    every alarming keyword and means the opposite.
* ``priced_in``  -- a re-report of something already known. Strong words, no
                    move, because the market saw it days ago.

The last two are the whole argument. A keyword matcher reads the alarming words
and pulls quotes; a reader understands that a denial is not the event. Whether
real news distributes like this is a separate question this repo does not
answer -- see the README.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# (headline, kind, subtle, alarming_keywords_present)
TEMPLATES: list[tuple[str, str, bool, bool]] = [
    # --- genuinely material, and it reads that way ------------------------
    ("Major exchange halts all BTC withdrawals citing wallet issue",
     "material_down", False, True),
    ("Regulator opens enforcement action against the largest BTC venue",
     "material_down", False, True),
    ("Top-three exchange reports unauthorised access to a hot wallet",
     "material_down", False, True),
    ("Spot BTC ETF approved for the second-largest asset manager",
     "material_up", False, True),
    ("Sovereign wealth fund discloses a multi-billion BTC allocation",
     "material_up", False, True),
    ("Central bank signals it will hold rates and ease balance-sheet runoff",
     "material_up", False, True),

    # --- genuinely material, but it does NOT read that way ---------------
    ("Regulator declines to appeal last month's custody ruling",
     "material_up", True, False),
    ("Quarterly filing shows the trust's redemption window closing early",
     "material_down", True, False),
    ("Exchange quietly updates its terms to permit rehypothecation",
     "material_down", True, False),
    ("Custodian's insurance underwriter raises its coverage ceiling",
     "material_up", True, False),

    # --- a denial: alarming words, opposite meaning -----------------------
    ("Exchange denies reports that BTC withdrawals have been halted",
     "denial", True, True),
    ("Regulator says no enforcement action against the venue is planned",
     "denial", True, True),
    ("Custodian calls the reported wallet breach 'entirely false'",
     "denial", True, True),
    ("Firm rejects rumours it is liquidating its BTC treasury",
     "denial", True, True),

    # --- already in the price --------------------------------------------
    ("Analysts revisit the unlock schedule announced in March",
     "priced_in", True, True),
    ("Weekly recap: the ETF approval that moved markets last Tuesday",
     "priced_in", True, True),
    ("Explainer republished on the exchange outage from two weeks ago",
     "priced_in", True, True),

    # --- newsy noise ------------------------------------------------------
    ("Analyst reiterates long-term constructive view on digital assets",
     "noise", False, False),
    ("Conference panel debates the future of on-chain settlement",
     "noise", False, False),
    ("Survey finds retail interest in crypto broadly unchanged",
     "noise", False, False),
    ("Miner publishes its routine monthly production update",
     "noise", False, False),
    ("Commentator argues the halving narrative is overstated",
     "noise", False, False),
]

# The keyword rule's dictionary. Built to be a *strong* competitor: it fires on
# every obviously material headline in TEMPLATES, in both directions. Its only
# failures are the ones that need reading -- a denial, a re-report, and news
# whose significance is not in its vocabulary. Anything less would be a
# strawman, and beating a strawman would prove nothing.
ALARM_WORDS = (
    "halt", "halts", "halted", "suspends", "suspended", "withdrawals",
    "enforcement", "breach", "unauthorised", "liquidating", "outage",
    "approved", "approval", "denies", "denied", "rejects", "false",
    "rumour", "rumours", "unlock", "sovereign wealth", "allocation",
    "central bank", "rates", "hack", "insolvent", "investigation",
)


@dataclass(frozen=True)
class NewsEvent:
    seq: int  # the tick it lands on
    headline: str
    kind: str
    subtle: bool
    alarming: bool  # would a keyword matcher light up?
    impact_bps: float  # what the tape will actually do

    @property
    def material(self) -> bool:
        return self.kind in ("material_up", "material_down")


class EventGenerator:
    """Drops events onto a tick stream at a configurable rate."""

    def __init__(
        self,
        seed: int = 3,
        rate_per_1000: float = 12.0,
        impact_bps: float = 22.0,
        impact_spread: float = 0.45,
    ) -> None:
        self.rng = random.Random(seed)
        self.rate = rate_per_1000 / 1000.0
        self.impact_bps = impact_bps
        self.impact_spread = impact_spread

    def audit(self) -> dict[str, int]:
        """How the keyword rule scores on the template set, by construction."""
        hits = misses = false_alarms = quiet = 0
        for headline, kind, _subtle, _alarming in TEMPLATES:
            fires = keyword_alarm(headline)
            moves = kind in ("material_up", "material_down")
            if moves and fires:
                hits += 1
            elif moves:
                misses += 1
            elif fires:
                false_alarms += 1
            else:
                quiet += 1
        return {"hits": hits, "misses": misses,
                "false_alarms": false_alarms, "correctly_quiet": quiet}

    def maybe(self, seq: int) -> NewsEvent | None:
        if self.rng.random() >= self.rate:
            return None
        headline, kind, subtle, alarming = self.rng.choice(TEMPLATES)

        size = self.impact_bps * (1 + self.rng.gauss(0.0, self.impact_spread))
        size = max(size, self.impact_bps * 0.25)
        if kind == "material_up":
            impact = size
        elif kind == "material_down":
            impact = -size
        elif kind == "denial":
            # A denial resolves a fear, so it drifts slightly the other way.
            impact = size * 0.18
        else:  # noise, priced_in
            impact = 0.0

        return NewsEvent(
            seq=seq,
            headline=headline,
            kind=kind,
            subtle=subtle,
            alarming=alarming,
            impact_bps=impact,
        )


def keyword_alarm(headline: str) -> bool:
    """The competitor: does this headline contain an alarming word?

    Deliberately the *good* version of a naive rule -- it is case-insensitive
    and covers both directions. It still cannot tell a denial from an event,
    which is the point.
    """
    low = headline.lower()
    # Word boundaries, so "rates" does not fire inside "reiterates". Without
    # this the rule looks dumber than a real one would be.
    return any(re.search(rf"\b{re.escape(word)}\b", low) for word in ALARM_WORDS)
