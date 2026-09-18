"""The mechanical half: a two-sided quoting engine, pure code, no model.

This is deliberately the part nobody needs an LLM for. Reservation price is
skewed against inventory so the quoter naturally works its position back to
flat, and the half-spread widens with volatility. It is a simplified
Avellaneda-Stoikov quoter, and it runs every tick in microseconds.

Everything a model contributes enters through :class:`Posture`, which the
quoter reads but never waits for.
"""

from __future__ import annotations

from dataclasses import dataclass

from .market import Observation, Quote


@dataclass(frozen=True)
class Posture:
    """A risk stance, applied per side.

    Per side is the point. A volatility trigger can only widen symmetrically,
    because it has no idea which way the move is coming. Knowing the direction
    lets you pull the side that is about to be picked off and keep quoting the
    other -- staying in business and positioning at the same time.
    """

    bid_spread_mult: float = 1.0
    ask_spread_mult: float = 1.0
    bid_size_mult: float = 1.0
    ask_size_mult: float = 1.0
    reason: str = ""
    ttl: int = 0  # ticks this posture still applies for

    @staticmethod
    def normal() -> "Posture":
        return Posture()

    def tick(self) -> "Posture":
        if self.ttl <= 1:
            return Posture.normal()
        return Posture(
            self.bid_spread_mult, self.ask_spread_mult,
            self.bid_size_mult, self.ask_size_mult,
            self.reason, self.ttl - 1,
        )

    @property
    def defensive(self) -> bool:
        return (
            self.bid_spread_mult > 1.01 or self.ask_spread_mult > 1.01
            or self.bid_size_mult < 0.99 or self.ask_size_mult < 0.99
        )


@dataclass
class QuoterConfig:
    base_half_spread_bps: float = 2.0
    quote_size: float = 1.0
    max_inventory: float = 8.0
    inventory_skew_bps: float = 2.5  # at full inventory
    vol_widen: float = 0.35  # extra half-spread per bp of tick vol


class Quoter:
    def __init__(self, config: QuoterConfig | None = None) -> None:
        self.config = config or QuoterConfig()

    def quote(
        self, obs: Observation, inventory: float, posture: Posture
    ) -> Quote:
        cfg = self.config
        ratio = max(-1.0, min(1.0, inventory / cfg.max_inventory))

        # Long inventory pushes both quotes down, so we are more likely to sell.
        reservation = obs.mid * (1.0 - ratio * cfg.inventory_skew_bps / 1e4)
        half_bps = cfg.base_half_spread_bps + cfg.vol_widen * obs.vol_bps

        bid_bps = half_bps * posture.bid_spread_mult
        ask_bps = half_bps * posture.ask_spread_mult

        bid_size = cfg.quote_size * posture.bid_size_mult
        ask_size = cfg.quote_size * posture.ask_size_mult
        # Never add to a position that is already at the limit.
        if inventory >= cfg.max_inventory:
            bid_size = 0.0
        if inventory <= -cfg.max_inventory:
            ask_size = 0.0

        return Quote(
            bid=reservation * (1 - bid_bps / 1e4),
            ask=reservation * (1 + ask_bps / 1e4),
            bid_size=max(bid_size, 0.0),
            ask_size=max(ask_size, 0.0),
        )
