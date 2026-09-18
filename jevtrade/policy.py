"""Jev's answers in, a target position out.

The model is never asked "how much should I buy". It answers six judgment
questions; this module turns those into a quantity with weights that live in
code, where they can be tuned and tested without rewriting a prompt. That
split is the whole architecture:

    "Design AI-powered software by keeping code in control and giving System
     One narrow, structured decisions."
    -- https://docs.typesafe.ai/concepts/how-to-build-with-system-one

Confidence is used as a second axis, per the confidence-gated routing pattern:
the answer says which way, confidence says whether to act at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from .features import Features
from .questions import (
    CONTINUATION,
    CUT_POSITION,
    DIRECTION,
    DISORDERLY,
    DOWN,
    FOLLOW_THROUGH,
    LIQUIDITY_OK,
    REVERSAL,
    SETUP_QUALITY,
    UP,
)
from .types import Decision, JevResponse


@dataclass
class PolicyConfig:
    """Every tunable in one place. None of these reach the model."""

    # A clip small enough not to move the book: ~$7.6k against a touch that
    # holds ~$300k in the synthetic feed. Oversized clips pay huge impact.
    max_units: float = 0.1

    # Gates. Ordered from "never trade" to "trade smaller".
    hazard_max: float = 0.70  # disorderly above this -> flatten
    cut_threshold: float = 0.65  # cut_position above this -> flatten
    liquidity_min: float = 0.40  # below this -> no new risk, exits still allowed
    min_confidence: float = 0.20  # direction confidence below this -> stand aside
    conviction_floor: float = 0.30  # setup quality below this -> stand aside
    edge_deadband: float = 0.10  # |P(up) - P(down)| below this -> stand aside

    # Sizing.
    continuation_boost: float = 0.45
    reversal_damp: float = 0.60
    confidence_weight: float = 0.5  # 0 = ignore confidence when sizing, 1 = fully
    gain: float = 1.6  # maps a raw signal of ~0.6 onto full size

    # Turnover control: ignore target changes smaller than this share of the cap.
    rebalance_deadband: float = 0.18


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def decide(
    features: Features,
    response: JevResponse,
    position_units: float,
    config: PolicyConfig,
) -> Decision:
    """Combine the six answers into a target position in base units."""
    direction = response.choice(DIRECTION)
    follow = response.choice(FOLLOW_THROUGH)
    setup = response.score(SETUP_QUALITY)
    liquidity = response.noul(LIQUIDITY_OK).noul
    hazard = response.noul(DISORDERLY).noul
    cut = response.noul(CUT_POSITION).noul

    edge = direction.p(UP) - direction.p(DOWN)
    conviction = setup.normalized
    confidence = direction.confidence

    # Follow-through modifies how much of the directional read to trust, not
    # which way it points -- a reversal read means the recent move is likely to
    # give back, so lean on it less.
    follow_multiplier = _clamp(
        1.0
        + config.continuation_boost * follow.p(CONTINUATION)
        - config.reversal_damp * follow.p(REVERSAL),
        0.25,
        1.5,
    )
    confidence_multiplier = (
        1.0 - config.confidence_weight
    ) + config.confidence_weight * confidence

    raw = edge * conviction * follow_multiplier * confidence_multiplier * config.gain
    ideal = _clamp(raw, -1.0, 1.0) * config.max_units

    gate = ""
    if hazard > config.hazard_max:
        ideal, gate = 0.0, "disorderly"
    elif cut > config.cut_threshold:
        ideal, gate = 0.0, "cut_position"
    elif confidence < config.min_confidence:
        ideal, gate = 0.0, "low_confidence"
    elif conviction < config.conviction_floor:
        ideal, gate = 0.0, "weak_setup"
    elif abs(edge) < config.edge_deadband:
        ideal, gate = 0.0, "no_edge"
    elif liquidity < config.liquidity_min and abs(ideal) > abs(position_units):
        # Illiquid: hold what we have, but do not add to it. Exits stay open,
        # which is why this gate is checked last and only blocks increases.
        ideal, gate = position_units, "illiquid"

    # Turnover control. Crossing the spread costs money every time, so ignore
    # small adjustments -- but never ignore an instruction to go flat.
    target = ideal
    if (
        abs(ideal - position_units) < config.rebalance_deadband * config.max_units
        and not (ideal == 0.0 and position_units != 0.0)
    ):
        target = position_units

    return Decision(
        seq=features.seq,
        ts=features.ts,
        target_units=target,
        p_up=direction.p(UP),
        p_down=direction.p(DOWN),
        edge=edge,
        conviction=conviction,
        confidence=confidence,
        hazard=hazard,
        tradeable=liquidity,
        reduce_risk=cut,
        gate=gate,
        latency_ms=response.latency_ms,
        usage_tokens=response.input_tokens,
    )
