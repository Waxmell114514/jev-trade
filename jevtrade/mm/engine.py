"""The market-making loop.

Order of operations each tick, chosen so nothing can see the future:

1. observe the mid (and any headline landing on this tick)
2. apply any posture whose latency has now elapsed
3. quote
4. order flow arrives and fills against that quote
5. the strategy may start a new evaluation, which lands later

Step 2 and step 5 are the architecture: the quoter reads the posture, it never
waits for it. A slow model means the *old* stance stays up for longer, which
costs money but cannot stall the quoter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .events import NewsEvent
from .market import MarketConfig, MarketSim, MMFill
from .quoter import Posture, Quoter, QuoterConfig
from .strategies import PostureUpdate, Strategy


@dataclass
class MMRun:
    name: str
    mids: list[float] = field(default_factory=list)
    fills: list[MMFill] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    inventory_curve: list[float] = field(default_factory=list)
    events: list[NewsEvent] = field(default_factory=list)
    posture_log: list[dict] = field(default_factory=list)
    defensive_ticks: int = 0
    quoting_ticks: int = 0
    strategy: Strategy | None = None
    interval_ms: int = 250


def run(
    strategy: Strategy,
    *,
    market: MarketConfig | None = None,
    quoter: QuoterConfig | None = None,
    extra_latency_ms: float = 0.0,
) -> MMRun:
    market = market or MarketConfig()
    sim = MarketSim(market)
    engine_quoter = Quoter(quoter)

    # Keep the instrument the model is told about identical to the one being
    # simulated. Letting these drift silently cost a full experiment once: Jev
    # was asked whether BTC headlines would move ETH, correctly said no, and
    # never took a stance.
    config = getattr(strategy, "config", None)
    if config is not None and hasattr(config, "instrument"):
        config.instrument = market.symbol

    out = MMRun(name=strategy.name, strategy=strategy, interval_ms=market.interval_ms)
    posture = Posture.normal()
    pending: list[PostureUpdate] = []
    inventory = 0.0
    cash = 0.0

    while (obs := sim.observe()) is not None:
        if obs.event is not None:
            out.events.append(obs.event)

        # --- a stance computed earlier may now have arrived
        still: list[PostureUpdate] = []
        for update in pending:
            if update.apply_seq <= obs.seq:
                posture = update.posture
            else:
                still.append(update)
        pending = still

        quote = engine_quoter.quote(obs, inventory, posture)
        if posture.defensive:
            out.defensive_ticks += 1
        if quote.bid_size > 0 or quote.ask_size > 0:
            out.quoting_ticks += 1

        for fill in sim.advance(quote):
            out.fills.append(fill)
            if fill.side == "buy":
                inventory += fill.size
                cash -= fill.price * fill.size
            else:
                inventory -= fill.size
                cash += fill.price * fill.size

        out.mids.append(obs.mid)
        out.inventory_curve.append(inventory)
        out.equity_curve.append(cash + inventory * obs.mid)

        # --- the strategy may start a new evaluation; it lands later
        update = strategy.evaluate(obs, inventory)
        if update is not None:
            total_ms = update.latency_ms + extra_latency_ms
            delay = math.ceil(total_ms / market.interval_ms) if total_ms > 0 else 0
            update.apply_seq = obs.seq + delay
            pending.append(update)
            if update.detail:
                out.posture_log.append(
                    {"seq": obs.seq, "delay_ticks": delay, **update.detail}
                )

        posture = posture.tick()

    return out
