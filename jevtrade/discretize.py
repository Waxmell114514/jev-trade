"""Numbers in, words out.

TypeSafe's jaggedness notes are blunt about this:

    "jev-1.13 will perform better on semantic representations than numeric.
     ... do the conversion in code and pass in either the computed number or a
     named bucket. Keep the model for the part that is genuinely a judgment."

So every feature crosses into the model's state as a phrase from a closed
vocabulary. The raw floats never reach Jev; they are kept on the decision
record for the audit trail instead.

One deliberate choice worth calling out: the spread is described relative to
the typical tick move rather than in basis points. "Costs more than a typical
move to cross" is the fact that actually decides whether a scalp is viable,
and it is a judgment the model can use; "3.4 bps" is not.
"""

from __future__ import annotations

from typing import Any, Sequence

from .features import Features

# --- closed vocabularies -----------------------------------------------------

MOVE = (
    "falling hard",
    "falling",
    "drifting down",
    "going nowhere",
    "drifting up",
    "rising",
    "rising hard",
)
MOVE_CUTS = (-2.0, -1.0, -0.3, 0.3, 1.0, 2.0)

VOLATILITY = (
    "much quieter than usual",
    "quieter than usual",
    "normal",
    "more active than usual",
    "far more active than usual",
)
VOLATILITY_CUTS = (0.6, 0.85, 1.2, 1.7)

VOLUME = ("very light", "light", "normal", "heavy", "surging")
VOLUME_CUTS = (-1.0, -0.35, 0.75, 2.0)

BOOK = (
    "heavily weighted to sellers",
    "weighted to sellers",
    "balanced",
    "weighted to buyers",
    "heavily weighted to buyers",
)
BOOK_CUTS = (-0.5, -0.15, 0.15, 0.5)

DEPTH = ("much thinner than usual", "thinner than usual", "normal", "deeper than usual")
DEPTH_CUTS = (0.5, 0.8, 1.3)

CROSS_COST = (
    "negligible next to a typical move",
    "small next to a typical move",
    "about the size of a typical move",
    "larger than a typical move",
    "far larger than a typical move",
)
CROSS_COST_CUTS = (0.25, 0.6, 1.0, 2.0)

CHARACTER = (
    "pure chop, no direction holds",
    "mostly chop",
    "mixed",
    "directional",
    "a clean one-way run",
)
CHARACTER_CUTS = (0.15, 0.3, 0.5, 0.7)

VS_VWAP = (
    "far below the session's average traded price",
    "below the session's average traded price",
    "around the session's average traded price",
    "above the session's average traded price",
    "far above the session's average traded price",
)
VS_VWAP_CUTS = (-2.0, -0.6, 0.6, 2.0)

EXPOSURE = (
    "at the maximum short",
    "meaningfully short",
    "slightly short",
    "flat, no position",
    "slightly long",
    "meaningfully long",
    "at the maximum long",
)
EXPOSURE_CUTS = (-0.85, -0.45, -0.05, 0.05, 0.45, 0.85)

OPEN_PNL = (
    "a significant loss",
    "a small loss",
    "roughly break-even",
    "a small gain",
    "a significant gain",
)
OPEN_PNL_CUTS = (-2.0, -0.5, 0.5, 2.0)

ALL_VOCABULARIES = (
    MOVE,
    VOLATILITY,
    VOLUME,
    BOOK,
    DEPTH,
    CROSS_COST,
    CHARACTER,
    VS_VWAP,
    EXPOSURE,
    OPEN_PNL,
)


def bucket(value: float, cuts: Sequence[float], labels: Sequence[str]) -> str:
    """Map a number to a label. ``labels`` must be one longer than ``cuts``."""
    if len(labels) != len(cuts) + 1:
        raise ValueError("labels must be exactly one longer than cuts")
    for index, cut in enumerate(cuts):
        if value < cut:
            return labels[index]
    return labels[-1]


def _streak(run_length: int) -> str:
    if run_length == 0:
        return "no streak"
    direction = "up" if run_length > 0 else "down"
    count = abs(run_length)
    tick = "tick" if count == 1 else "ticks"
    return f"{count} consecutive {direction} {tick}"


def build_state(
    features: Features,
    *,
    position_units: float,
    max_units: float,
    open_pnl_vols: float = 0.0,
    interval_ms: int = 1000,
    reconstructed_book: bool = False,
) -> dict[str, Any]:
    """Assemble the ``state`` object for a /v1/systemone request.

    Kept deliberately small. Accuracy falls as the state grows with content
    unrelated to the decision, so this carries the ten facts the questions
    actually ask about and nothing else.
    """
    vol = max(features.vol_bps, 1e-9)
    exposure_frac = position_units / max_units if max_units > 0 else 0.0
    interval = (
        f"{interval_ms} ms" if interval_ms < 1000 else f"{interval_ms // 1000} second"
    )

    book_note = (
        "sizes are reconstructed from bar data, treat as approximate"
        if reconstructed_book
        else "top of book"
    )

    return {
        "instrument": f"{features.symbol} spot, {interval} snapshots",
        "price_action": {
            "last_snapshot": bucket(features.ret_1_bps / vol, MOVE_CUTS, MOVE),
            "last_5_snapshots": bucket(features.move_z, MOVE_CUTS, MOVE),
            "last_15_snapshots": bucket(
                features.ret_15_bps / (vol * 3.873), MOVE_CUTS, MOVE
            ),
            "streak": _streak(features.run_length),
            "recent_balance": (
                f"{round(features.up_fraction * 10)} of the last 10 snapshots "
                "closed higher"
            ),
            "character": bucket(
                features.trend_efficiency, CHARACTER_CUTS, CHARACTER
            ),
            "versus_session": bucket(
                features.dist_vwap_bps / (vol * 5), VS_VWAP_CUTS, VS_VWAP
            ),
        },
        "activity": {
            "volatility": bucket(features.vol_ratio, VOLATILITY_CUTS, VOLATILITY),
            "traded_volume": bucket(features.volume_z, VOLUME_CUTS, VOLUME),
        },
        "order_book": {
            "note": book_note,
            "resting_size": bucket(features.imbalance, BOOK_CUTS, BOOK),
            "depth": bucket(features.depth_ratio, DEPTH_CUTS, DEPTH),
            "cost_to_cross_the_spread": bucket(
                features.spread_bps / vol, CROSS_COST_CUTS, CROSS_COST
            ),
        },
        "our_book": {
            "exposure": bucket(exposure_frac, EXPOSURE_CUTS, EXPOSURE),
            "open_position_pnl": (
                "no open position"
                if abs(position_units) < 1e-12
                else bucket(open_pnl_vols, OPEN_PNL_CUTS, OPEN_PNL)
            ),
        },
    }
