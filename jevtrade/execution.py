"""A taker-only execution simulator.

Nothing here is clever, but it is pessimistic in the two ways that matter for
a high-frequency strategy, because those are what usually turn a promising
backtest into a losing one:

* Every trade crosses the spread. Buys lift the ask, sells hit the bid.
* Every trade pays a taker fee and moves the price against itself in
  proportion to how much of the resting size it consumes.

A strategy that trades on a 1 bp edge through a 1.2 bp spread and a 2 bp fee
loses money, and it should be visible in the output that it does.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Fill, Position, Tick

MIN_TRADE_UNITS = 1e-9


@dataclass
class ExecutionConfig:
    taker_fee_bps: float = 2.0
    # Basis points of adverse price movement when a trade consumes the whole
    # of the resting size at the touch.
    impact_bps_at_full_book: float = 3.0
    # Trades larger than this share of the touch are assumed to walk the book.
    max_touch_participation: float = 1.0


class Broker:
    """Holds the position and turns target quantities into fills."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()
        self.position = Position()

    def fill_price(self, tick: Tick, side: str, units: float) -> float:
        """Touch price plus size-dependent impact."""
        resting = tick.ask_size if side == "buy" else tick.bid_size
        participation = units / resting if resting > 0 else 1.0
        impact_bps = self.config.impact_bps_at_full_book * min(participation, 5.0)
        touch = tick.ask if side == "buy" else tick.bid
        direction = 1.0 if side == "buy" else -1.0
        return touch * (1.0 + direction * impact_bps / 1e4)

    def move_to(self, tick: Tick, target_units: float) -> Fill | None:
        """Trade the difference between the current and target position."""
        delta = target_units - self.position.units
        if abs(delta) < MIN_TRADE_UNITS:
            return None

        side = "buy" if delta > 0 else "sell"
        units = abs(delta)
        price = self.fill_price(tick, side, units)
        fee = price * units * self.config.taker_fee_bps / 1e4

        fill = Fill(seq=tick.seq, ts=tick.ts, side=side, units=units, price=price, fee=fee)
        self._apply(fill)
        return fill

    def _apply(self, fill: Fill) -> None:
        position = self.position
        signed = fill.units if fill.side == "buy" else -fill.units
        before = position.units
        after = before + signed

        same_direction = before == 0.0 or (before > 0) == (signed > 0)
        if same_direction:
            total = abs(before) + fill.units
            position.avg_price = (
                position.avg_price * abs(before) + fill.price * fill.units
            ) / total
        else:
            closed = min(abs(before), fill.units)
            sign = 1.0 if before > 0 else -1.0
            position.realized_pnl += (fill.price - position.avg_price) * closed * sign
            if fill.units > abs(before):
                position.avg_price = fill.price  # flipped through flat

        position.units = 0.0 if abs(after) < MIN_TRADE_UNITS else after
        if position.units == 0.0:
            position.avg_price = 0.0
        position.fees_paid += fill.fee
        position.fills.append(fill)
