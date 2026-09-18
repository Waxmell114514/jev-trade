"""Rule-based strategies to compare the model against.

A P&L number on its own means nothing. These run over the identical tick
stream through the identical execution simulator, with the same one-tick
execution delay, so the comparison is honest.

``imbalance`` is the important one. In the synthetic feed the book imbalance is
*where the edge actually is*, so a two-line rule that reads it is close to the
best a strategy can do. If a model cannot beat two lines of arithmetic, that is
the finding, and it should be easy to see rather than easy to miss.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable

from .engine import EngineConfig, EngineResult
from .execution import Broker, ExecutionConfig
from .features import Features, FeatureWindow
from .policy import PolicyConfig
from .types import Tick

Signal = Callable[[Features], float]


def flat(_: Features) -> float:
    return 0.0


def buy_and_hold(_: Features) -> float:
    return 1.0


def momentum(features: Features) -> float:
    """Lean with the last five snapshots, scaled by how big the move was."""
    return max(-1.0, min(1.0, features.move_z / 2.0))


def mean_reversion(features: Features) -> float:
    return -momentum(features)


def imbalance(features: Features) -> float:
    """Lean with the resting size at the touch."""
    return max(-1.0, min(1.0, features.imbalance * 1.5))


def random_signal(seed: int = 11) -> Signal:
    rng = random.Random(seed)

    def signal(_: Features) -> float:
        return rng.choice((-1.0, 0.0, 1.0))

    return signal


def registry() -> dict[str, Signal]:
    return {
        "flat": flat,
        "buy_and_hold": buy_and_hold,
        "momentum": momentum,
        "mean_reversion": mean_reversion,
        "imbalance": imbalance,
        "random": random_signal(),
    }


def run_rule(
    feed: Iterable[Tick],
    signal: Signal,
    *,
    name: str = "rule",
    policy: PolicyConfig | None = None,
    execution: ExecutionConfig | None = None,
    engine: EngineConfig | None = None,
    symbol: str = "",
    interval_ms: int = 1000,
) -> EngineResult:
    """Same loop as :func:`jevtrade.engine.run`, minus the model."""
    policy = policy or PolicyConfig()
    engine = engine or EngineConfig()
    broker = Broker(execution)
    window = FeatureWindow(size=engine.window_size, warmup=engine.warmup_ticks)
    result = EngineResult(
        symbol=symbol, interval_ms=interval_ms, provider="rule", model=name
    )

    pending: tuple[int, float] | None = None
    for tick in feed:
        result.ticks += 1
        result.mids.append(tick.mid)
        result.seqs.append(tick.seq)
        features = window.update(tick)

        if pending is not None and tick.seq >= pending[0]:
            fill = broker.move_to(tick, pending[1])
            if fill is not None:
                result.fills.append(fill)
                result.turnover += fill.units * fill.price
            pending = None

        result.equity_curve.append(broker.position.equity(tick.mid))

        if features is None or tick.seq % engine.decide_every != 0:
            continue

        target = max(-1.0, min(1.0, signal(features))) * policy.max_units
        if (
            abs(target - broker.position.units)
            < policy.rebalance_deadband * policy.max_units
        ):
            continue
        pending = (tick.seq + 1, target)

    last_mid = result.mids[-1] if result.mids else 0.0
    result.final_equity = broker.position.equity(last_mid)
    result.gross_pnl = broker.position.gross_pnl(last_mid)
    result.fees_paid = broker.position.fees_paid
    return result
