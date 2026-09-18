"""The trading loop.

The interesting part of this file is that **latency is modelled explicitly**.

A decision formed from the tape at tick *t* cannot be acted on at tick *t*. It
lands at *t + ceil(latency / tick_interval)*, and it is filled at whatever the
price has become by then. Decisions slower than ``deadline_ms`` are discarded
rather than acted on, because a stale read of the tape is worse than no read.
Only one request is ever in flight, so a slow provider does not just arrive
late -- it also thins out how often the strategy gets to look at the market.

That makes the loop a measuring instrument: turn ``extra_latency_ms`` up and
watch the P&L, the drop rate and the skip rate move. ``jevtrade sweep`` does
exactly that.

Two safeguards run in code on every tick, independent of the model: a stop
loss and a drawdown kill switch. Nothing that has to make a network call
should be the only thing standing between a position and a loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .discretize import build_state
from .execution import Broker, ExecutionConfig
from .features import FeatureWindow
from .jev.client import JevClient
from .policy import PolicyConfig, decide
from .types import Decision, Fill, Tick


@dataclass
class EngineConfig:
    decide_every: int = 8  # ticks between decision points
    deadline_ms: float = 400.0  # later than this is dropped; 0 disables
    extra_latency_ms: float = 0.0  # pretend the provider is this much slower
    stop_loss_bps: float | None = 10.0  # code-side, runs every tick
    max_drawdown: float | None = None  # kill switch, in quote currency
    horizon_ticks: int = 8  # forward window used to score decisions
    window_size: int = 120
    warmup_ticks: int = 60
    reconstructed_book: bool = False


@dataclass
class EngineResult:
    symbol: str
    interval_ms: int
    provider: str
    model: str
    ticks: int = 0
    mids: list[float] = field(default_factory=list)
    seqs: list[int] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    decision_mids: list[float] = field(default_factory=list)
    applied_seqs: list[int] = field(default_factory=list)  # when each landed
    equity_curve: list[float] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    expired: int = 0
    skipped_busy: int = 0
    stops: int = 0
    halted: bool = False
    input_tokens: int = 0
    final_equity: float = 0.0
    gross_pnl: float = 0.0
    fees_paid: float = 0.0
    turnover: float = 0.0
    errors: int = 0

    @property
    def latencies_ms(self) -> list[float]:
        return [d.latency_ms for d in self.decisions]


@dataclass
class _Pending:
    decision: Decision
    apply_seq: int


def run(
    feed: Iterable[Tick],
    client: JevClient,
    *,
    policy: PolicyConfig | None = None,
    execution: ExecutionConfig | None = None,
    engine: EngineConfig | None = None,
    symbol: str = "",
    interval_ms: int = 1000,
    on_decision: Any = None,
) -> EngineResult:
    policy = policy or PolicyConfig()
    engine = engine or EngineConfig()
    broker = Broker(execution)
    window = FeatureWindow(size=engine.window_size, warmup=engine.warmup_ticks)

    result = EngineResult(
        symbol=symbol,
        interval_ms=interval_ms,
        provider=client.provider,
        model=client.model,
    )
    pending: _Pending | None = None
    peak_equity = 0.0

    for tick in feed:
        result.ticks += 1
        result.mids.append(tick.mid)
        result.seqs.append(tick.seq)
        features = window.update(tick)
        position = broker.position

        # --- 1. a decision made earlier may now be actionable
        if pending is not None and tick.seq >= pending.apply_seq:
            fill = broker.move_to(tick, pending.decision.target_units)
            if fill is not None:
                result.fills.append(fill)
                result.turnover += fill.units * fill.price
            pending = None

        # --- 2. code-side safeguards, every tick, no network involved
        if engine.stop_loss_bps is not None and position.units != 0.0:
            sign = 1.0 if position.units > 0 else -1.0
            excursion_bps = (
                (tick.mid - position.avg_price) / position.avg_price * 1e4 * sign
            )
            if excursion_bps < -engine.stop_loss_bps:
                fill = broker.move_to(tick, 0.0)
                if fill is not None:
                    result.fills.append(fill)
                    result.turnover += fill.units * fill.price
                    result.stops += 1
                pending = None

        equity = broker.position.equity(tick.mid)
        result.equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)

        if (
            engine.max_drawdown is not None
            and not result.halted
            and peak_equity - equity > engine.max_drawdown
        ):
            result.halted = True
            fill = broker.move_to(tick, 0.0)
            if fill is not None:
                result.fills.append(fill)
                result.turnover += fill.units * fill.price
            pending = None

        # --- 3. decision point
        if result.halted or features is None:
            continue
        if tick.seq % engine.decide_every != 0:
            continue
        if pending is not None:
            result.skipped_busy += 1
            continue

        state = _state_for(tick, features, broker, policy, engine, interval_ms)
        try:
            response = client.evaluate(state)
        except Exception:  # a provider failure must not take the book with it
            result.errors += 1
            continue

        result.input_tokens += response.input_tokens
        effective_ms = response.latency_ms + engine.extra_latency_ms
        if engine.deadline_ms > 0 and effective_ms > engine.deadline_ms:
            result.expired += 1
            continue

        decision = decide(features, response, broker.position.units, policy)
        result.decisions.append(decision)
        result.decision_mids.append(tick.mid)
        if on_decision is not None:
            on_decision(tick, features, state, response, decision)

        delay_ticks = max(1, math.ceil(effective_ms / interval_ms))
        result.applied_seqs.append(tick.seq + delay_ticks)
        pending = _Pending(decision=decision, apply_seq=tick.seq + delay_ticks)

    last_mid = result.mids[-1] if result.mids else 0.0
    result.final_equity = broker.position.equity(last_mid)
    result.gross_pnl = broker.position.gross_pnl(last_mid)
    result.fees_paid = broker.position.fees_paid
    return result


def _state_for(
    tick: Tick,
    features,
    broker: Broker,
    policy: PolicyConfig,
    engine: EngineConfig,
    interval_ms: int,
) -> dict[str, Any]:
    position = broker.position
    open_pnl_vols = 0.0
    if position.units != 0.0 and position.avg_price > 0:
        sign = 1.0 if position.units > 0 else -1.0
        pnl_bps = (tick.mid - position.avg_price) / position.avg_price * 1e4 * sign
        open_pnl_vols = pnl_bps / max(features.vol_bps, 1e-9)

    return build_state(
        features,
        position_units=position.units,
        max_units=policy.max_units,
        open_pnl_vols=open_pnl_vols,
        interval_ms=interval_ms,
        reconstructed_book=engine.reconstructed_book,
    )
