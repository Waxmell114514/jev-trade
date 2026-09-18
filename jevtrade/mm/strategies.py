"""The four arms of the experiment.

Each one answers the same question -- what stance should the quoter take right
now -- and the comparison between them is the whole point.

* ``naive``   quotes through everything. The floor.
* ``vol``     the traditional defence: realised volatility spikes, widen both
              sides. It cannot act until the move has already begun, and it has
              no idea which side is about to be picked off.
* ``keyword`` what a real desk usually runs: a news-word trigger that pauses
              quoting for a few seconds. Strong protection, but it fires on
              every denial and every re-report too.
* ``jev``     reads the headline and the tape, and sets a stance per side.

``jev`` is the only one that can pull the exposed side and keep quoting the
other. Its posture also arrives *late*, by the measured round trip, because the
quoter never blocks on a network call.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..jev.client import JevClient, JevError
from .events import keyword_alarm
from .market import Observation
from .questions import (
    DENIED,
    DIRECTION,
    DOWN,
    INFORMED_FLOW,
    MOVES_PRICE,
    NEW_INFORMATION,
    REPORT_TYPE,
    SEVERITY,
    UP,
    build_state,
    headline_questions,
)
from .quoter import Posture


@dataclass
class PostureUpdate:
    """A stance, and the tick it actually reaches the quoter."""

    posture: Posture
    apply_seq: int
    latency_ms: float = 0.0
    tokens: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


class Strategy:
    name = "base"

    def evaluate(self, obs: Observation, inventory: float) -> PostureUpdate | None:
        return None


class NaiveStrategy(Strategy):
    name = "naive"


class VolStrategy(Strategy):
    """Widen symmetrically when realised volatility spikes."""

    name = "vol"

    def __init__(self, window: int = 24, trigger: float = 2.0,
                 widen: float = 4.0, ttl: int = 12) -> None:
        self.window, self.trigger, self.widen, self.ttl = window, trigger, widen, ttl
        self._mids: deque[float] = deque(maxlen=window + 1)
        self._baseline: deque[float] = deque(maxlen=400)

    def evaluate(self, obs: Observation, inventory: float) -> PostureUpdate | None:
        self._mids.append(obs.mid)
        if len(self._mids) < self.window:
            return None
        rets = [abs(math.log(b / a)) for a, b in zip(self._mids, list(self._mids)[1:])]
        recent = statistics.fmean(rets)
        self._baseline.append(recent)
        if len(self._baseline) < 120:
            return None
        typical = statistics.median(self._baseline)
        if typical > 0 and recent > self.trigger * typical:
            return PostureUpdate(
                Posture(bid_spread_mult=self.widen, ask_spread_mult=self.widen,
                        reason="volatility spike", ttl=self.ttl),
                apply_seq=obs.seq,  # local computation, no network
            )
        return None


class KeywordStrategy(Strategy):
    """Pause quoting for a few seconds when a news word appears."""

    name = "keyword"

    def __init__(self, ttl: int = 14) -> None:
        self.ttl = ttl

    def evaluate(self, obs: Observation, inventory: float) -> PostureUpdate | None:
        if obs.event is None or not keyword_alarm(obs.event.headline):
            return None
        return PostureUpdate(
            Posture(bid_size_mult=0.0, ask_size_mult=0.0,
                    bid_spread_mult=6.0, ask_spread_mult=6.0,
                    reason="news keyword", ttl=self.ttl),
            apply_seq=obs.seq,
        )


def headline_risk(response) -> float:
    """How much this headline should worry a market maker, in 0..1.

    Shared by the trading arm and the news study so both score a headline the
    same way.

    Deliberately not a plain product of four probabilities -- that lands near
    zero for everything and no threshold can separate the cases. "Would it
    move, and how much" carries the weight; the other two are discounts on it:

    * a report that the event did *not* happen is near-dispositive, however
      alarming its vocabulary (the model separates these cleanly, returning
      p(denied) ~ 0.99 on denials and ~0.00 on real events);
    * staleness discounts rather than annihilates, since a recap of real news
      can still matter a little.

    One-sided flow with no headline at all is its own reason to be careful, so
    it sets a floor.
    """
    moves = response.noul(MOVES_PRICE).noul
    severity = response.score(SEVERITY).normalized
    freshness = 0.2 + 0.8 * response.noul(NEW_INFORMATION).noul
    denial_discount = 1.0 - 0.93 * response.choice(REPORT_TYPE).p(DENIED)
    risk = moves * severity * freshness * denial_discount
    return max(risk, 0.85 * response.noul(INFORMED_FLOW).noul)


@dataclass
class JevConfig:
    ttl: int = 14
    risk_floor: float = 0.30  # below this, quote normally
    widen_max: float = 8.0
    anomaly_threshold: float = 0.55  # one-sided flow that triggers a look
    anomaly_cooldown: int = 20
    direction_confidence: float = 0.45  # below this, widen both sides instead
    # How decisively to act once the floor is cleared. A trigger that fires
    # wrongly must act gently; one that never does can afford to commit.
    size_pull_strength: float = 0.6  # strength above which the side is withdrawn
    pull_both_strength: float = 2.0  # above this, withdraw both sides (>1 = off)
    instrument: str = "BTC-USD spot"


class JevStrategy(Strategy):
    """Ask Jev, turn the typed answers into a per-side stance."""

    name = "jev"

    def __init__(self, client: JevClient, config: JevConfig | None = None) -> None:
        self.client = client
        self.config = config or JevConfig()
        self._questions = headline_questions()
        # This loop asks a different set than the directional trader does.
        if hasattr(client, "questions"):
            client.questions = self._questions
        self._last_headline: str | None = None
        self._last_headline_seq: int | None = None
        self._last_anomaly_seq = -10_000
        self.calls = 0
        self.errors = 0
        self.latencies: list[float] = []
        self.tokens = 0
        self.log: list[dict[str, Any]] = []

    # -------------------------------------------------------------- trigger

    def _trigger(self, obs: Observation) -> str | None:
        if obs.event is not None:
            return "a news headline just arrived"
        if (
            abs(obs.recent_flow_imbalance) >= self.config.anomaly_threshold
            and obs.seq - self._last_anomaly_seq >= self.config.anomaly_cooldown
        ):
            return "the tape has turned one-sided without any headline"
        return None

    def evaluate(self, obs: Observation, inventory: float) -> PostureUpdate | None:
        if obs.event is not None:
            self._last_headline = obs.event.headline
            self._last_headline_seq = obs.seq

        why = self._trigger(obs)
        if why is None:
            return None
        if obs.event is None:
            self._last_anomaly_seq = obs.seq

        state = build_state(
            headline=self._describe_headline(obs),
            instrument=self.config.instrument,
            inventory=_inventory_words(inventory),
            recent_trading=_flow_words(obs.recent_flow_imbalance),
            trigger=why,
        )

        try:
            response = self.client.evaluate(state)
        except JevError:
            self.errors += 1
            # A provider failure must fail safe, not fail open.
            return PostureUpdate(
                Posture(bid_spread_mult=2.5, ask_spread_mult=2.5,
                        reason="model unavailable", ttl=6),
                apply_seq=obs.seq,
            )

        self.calls += 1
        self.latencies.append(response.latency_ms)
        self.tokens += response.input_tokens
        posture, detail = self._posture_from(response, obs)
        return PostureUpdate(
            posture=posture,
            apply_seq=obs.seq,  # the engine adds the latency delay
            latency_ms=response.latency_ms,
            tokens=response.input_tokens,
            detail=detail,
        )

    def _describe_headline(self, obs: Observation) -> str:
        if obs.event is not None:
            return obs.event.headline
        if (
            self._last_headline is not None
            and self._last_headline_seq is not None
            and obs.seq - self._last_headline_seq <= 40
        ):
            return f"(no new headline; the most recent one was: {self._last_headline})"
        return "(no headline in the last few minutes)"

    # -------------------------------------------------------------- mapping

    def _posture_from(self, response, obs: Observation):
        cfg = self.config
        moves = response.noul(MOVES_PRICE).noul
        fresh = response.noul(NEW_INFORMATION).noul
        informed = response.noul(INFORMED_FLOW).noul
        severity = response.score(SEVERITY).normalized
        direction = response.choice(DIRECTION)
        report = response.choice(REPORT_TYPE)
        risk = headline_risk(response)
        detail = {
            "moves": round(moves, 3), "fresh": round(fresh, 3),
            "severity": round(severity, 3), "informed": round(informed, 3),
            "denial_p": round(report.p(DENIED), 3),
            "direction": direction.choice,
            "direction_conf": round(direction.confidence, 3),
            "risk": round(risk, 3),
            "headline": obs.event.headline if obs.event else "(tape)",
        }

        if risk < cfg.risk_floor:
            detail["stance"] = "normal"
            return Posture.normal(), detail

        strength = min(1.0, (risk - cfg.risk_floor) / (1.0 - cfg.risk_floor))
        widen = 1.0 + strength * (cfg.widen_max - 1.0)

        p_up, p_down = direction.p(UP), direction.p(DOWN)
        lean = p_up - p_down
        confident = direction.confidence >= cfg.direction_confidence and abs(lean) > 0.2

        if obs.event is None:
            # Tape-only trigger: there is no headline to read a direction from.
            # Which side is exposed is arithmetic, so code does it -- people
            # lifting our offer means the price is going up.
            imbalance = obs.recent_flow_imbalance
            if abs(imbalance) > 0.3:
                lean, confident = imbalance, True
                detail["direction"] = "up" if imbalance > 0 else "down"
                detail["direction_from"] = "tape"
            else:
                confident = False

        if strength >= cfg.pull_both_strength:
            detail["stance"] = "stand aside"
            return Posture(bid_spread_mult=widen, ask_spread_mult=widen,
                           bid_size_mult=0.0, ask_size_mult=0.0,
                           reason="material, severe", ttl=cfg.ttl), detail

        if not confident:
            # Uncertainty has a safe direction here: widen both sides and keep
            # earning, rather than stand aside and earn nothing.
            detail["stance"] = "widen both"
            return Posture(bid_spread_mult=widen, ask_spread_mult=widen,
                           reason="material, direction unclear", ttl=cfg.ttl), detail

        # Price about to rise -> our offer is the side that gets lifted.
        exposed_ask = lean > 0
        keep = 1.0
        detail["stance"] = "pull offer" if exposed_ask else "pull bid"
        posture = Posture(
            bid_spread_mult=widen if not exposed_ask else keep,
            ask_spread_mult=widen if exposed_ask else keep,
            bid_size_mult=0.0 if (not exposed_ask and strength > cfg.size_pull_strength) else 1.0,
            ask_size_mult=0.0 if (exposed_ask and strength > cfg.size_pull_strength) else 1.0,
            reason=f"material, {direction.choice}",
            ttl=cfg.ttl,
        )
        return posture, detail


def _inventory_words(inventory: float) -> str:
    if abs(inventory) < 0.5:
        return "flat"
    side = "long" if inventory > 0 else "short"
    size = "heavily" if abs(inventory) > 4 else "modestly"
    return f"{size} {side}"


def _flow_words(imbalance: float) -> str:
    if imbalance > 0.6:
        return "almost every recent trade has been someone buying from us"
    if imbalance > 0.25:
        return "recent trades have leaned towards people buying from us"
    if imbalance < -0.6:
        return "almost every recent trade has been someone selling to us"
    if imbalance < -0.25:
        return "recent trades have leaned towards people selling to us"
    return "buying and selling have been roughly balanced"
