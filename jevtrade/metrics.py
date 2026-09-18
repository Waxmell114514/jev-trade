"""Scoring a run.

Beyond the usual P&L statistics there are two measurements here aimed
specifically at a decision model:

* **Hit rate** -- of the times the model called a direction, how often did the
  mid actually go that way over the next few snapshots? A profitable-looking
  equity curve with a 50% hit rate made its money by luck or by sizing.
* **Calibration** -- bucket the model's own P(up) and compare it to how often
  up actually happened. Calibrated probabilities are Jev's central claim
  (TypeSafe trains for it with RLCD), and a decision layer that gates on
  confidence is only as sound as that calibration. If the 70% bucket comes in
  at 50%, the thresholds in ``PolicyConfig`` are built on sand.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from .engine import EngineResult
from .jev.client import usd_cost

SECONDS_PER_YEAR = 365 * 24 * 3600


@dataclass
class CalibrationBucket:
    low: float
    high: float
    n: int
    predicted: float
    realized: float


@dataclass
class Metrics:
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    turnover: float = 0.0
    fills: int = 0
    decisions: int = 0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    return_on_capital_pct: float = 0.0
    hit_rate: float = float("nan")  # scored from the tick the decision was formed on
    executed_hit_rate: float = float("nan")  # scored from the tick it reached the market
    scored_decisions: int = 0
    mean_abs_edge: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    expired: int = 0
    skipped_busy: int = 0
    stops: int = 0
    errors: int = 0
    halted: bool = False
    input_tokens: int = 0
    cost_usd: float = 0.0
    buy_and_hold: float = 0.0
    break_even_fee_bps: float = 0.0
    calibration: list[CalibrationBucket] = field(default_factory=list)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    low, high = math.floor(k), math.ceil(k)
    if low == high:
        return ordered[int(k)]
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def max_drawdown(curve: list[float]) -> float:
    peak = float("-inf")
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return -worst


def evaluate(
    result: EngineResult,
    *,
    horizon_ticks: int = 5,
    capital: float | None = None,
    edge_threshold: float = 0.05,
) -> Metrics:
    metrics = Metrics(
        net_pnl=result.final_equity,
        gross_pnl=result.gross_pnl,
        fees=result.fees_paid,
        turnover=result.turnover,
        fills=len(result.fills),
        decisions=len(result.decisions),
        expired=result.expired,
        skipped_busy=result.skipped_busy,
        stops=result.stops,
        errors=result.errors,
        halted=result.halted,
        input_tokens=result.input_tokens,
        cost_usd=usd_cost(result.input_tokens),
    )
    if not result.mids:
        return metrics

    capital = capital or result.mids[0]
    metrics.max_drawdown = max_drawdown(result.equity_curve)
    metrics.return_on_capital_pct = result.final_equity / capital * 100.0
    # Buy-and-hold sized to the same capital, so the comparison is fair.
    units = capital / result.mids[0] if result.mids[0] > 0 else 0.0
    metrics.buy_and_hold = (result.mids[-1] - result.mids[0]) * units

    # The taker fee at which this strategy exactly breaks even. Below it the
    # edge survives costs; above it no amount of accuracy saves the strategy.
    if result.turnover > 0:
        metrics.break_even_fee_bps = result.gross_pnl / result.turnover * 1e4

    # Sharpe from per-tick P&L, annualised by the tick interval.
    steps = [
        b - a for a, b in zip(result.equity_curve, result.equity_curve[1:])
    ]
    if len(steps) > 2:
        sigma = statistics.pstdev(steps)
        if sigma > 0:
            per_year = SECONDS_PER_YEAR / (result.interval_ms / 1000.0)
            metrics.sharpe = (
                statistics.fmean(steps) / sigma * math.sqrt(per_year)
            )

    latencies = result.latencies_ms
    metrics.latency_p50 = _percentile(latencies, 0.50)
    metrics.latency_p95 = _percentile(latencies, 0.95)
    metrics.latency_p99 = _percentile(latencies, 0.99)

    if result.decisions:
        metrics.mean_abs_edge = statistics.fmean(
            abs(d.edge) for d in result.decisions
        )

    _score_directions(result, metrics, horizon_ticks, edge_threshold)
    return metrics


def _score_directions(
    result: EngineResult,
    metrics: Metrics,
    horizon_ticks: int,
    edge_threshold: float,
) -> None:
    """Hit rate and calibration against realised forward returns."""
    position = {seq: i for i, seq in enumerate(result.seqs)}
    hits = 0
    scored = 0
    executed_hits = 0
    executed_scored = 0
    samples: list[tuple[float, bool]] = []

    applied = result.applied_seqs or [d.seq for d in result.decisions]
    for decision, mid, applied_seq in zip(
        result.decisions, result.decision_mids, applied
    ):
        index = position.get(decision.seq)
        if index is None:
            continue
        forward = index + horizon_ticks
        if forward >= len(result.mids) or mid <= 0:
            continue
        change = result.mids[forward] - mid
        if change == 0:
            continue

        directional = decision.p_up + decision.p_down
        if directional > 1e-9:
            samples.append((decision.p_up / directional, change > 0))

        if abs(decision.edge) >= edge_threshold:
            scored += 1
            if (decision.edge > 0) == (change > 0):
                hits += 1

            # The same call, but scored from where it actually got filled.
            # This is the number latency destroys: the tape has already moved.
            entry_index = position.get(applied_seq)
            if entry_index is not None:
                exit_index = entry_index + horizon_ticks
                if exit_index < len(result.mids):
                    entry = result.mids[entry_index]
                    realised = result.mids[exit_index] - entry
                    if realised != 0:
                        executed_scored += 1
                        if (decision.edge > 0) == (realised > 0):
                            executed_hits += 1

    metrics.scored_decisions = scored
    if scored:
        metrics.hit_rate = hits / scored
    if executed_scored:
        metrics.executed_hit_rate = executed_hits / executed_scored
    metrics.calibration = _calibration(samples)


def _calibration(
    samples: list[tuple[float, bool]], bins: int = 5
) -> list[CalibrationBucket]:
    buckets: list[CalibrationBucket] = []
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        inside = [
            (p, up) for p, up in samples if (low <= p < high or (i == bins - 1 and p == 1.0))
        ]
        if not inside:
            continue
        buckets.append(
            CalibrationBucket(
                low=low,
                high=high,
                n=len(inside),
                predicted=statistics.fmean(p for p, _ in inside),
                realized=statistics.fmean(1.0 if up else 0.0 for _, up in inside),
            )
        )
    return buckets
