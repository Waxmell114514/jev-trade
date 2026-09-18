"""Markout: the only measure that tells a market maker anything.

A fill's P&L splits cleanly in two:

* **capture** -- the half-spread earned at the instant of the fill,
  ``(mid - price) * signed_size``. Always positive; it is why you quote.
* **adverse selection** -- what the mid did *afterwards*,
  ``(mid_{t+h} - mid_t) * signed_size``. Negative when the counterparty knew
  something.

Their sum is the fill's true P&L at horizon h. A strategy that widens its
quotes trades capture away for protection; one that pulls gives up both. The
question this module answers is whether a stance bought more protection than it
cost.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..jev.client import usd_cost
from .engine import MMRun

HORIZONS = (4, 20, 80)  # ticks; at 250 ms these are 1 s, 5 s, 20 s


@dataclass
class Split:
    fills: int = 0
    notional: float = 0.0
    capture: float = 0.0
    adverse: dict[int, float] = field(default_factory=dict)

    def net(self, horizon: int) -> float:
        return self.capture + self.adverse.get(horizon, 0.0)


@dataclass
class MMMetrics:
    name: str = ""
    final_equity: float = 0.0
    end_inventory: float = 0.0
    max_drawdown: float = 0.0
    all: Split = field(default_factory=Split)
    informed: Split = field(default_factory=Split)
    benign: Split = field(default_factory=Split)
    quoting_pct: float = 0.0
    defensive_pct: float = 0.0
    calls: int = 0
    errors: int = 0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    tokens: int = 0
    cost_usd: float = 0.0


def _accumulate(split: Split, fill, mids: list[float]) -> None:
    split.fills += 1
    split.notional += fill.price * fill.size
    signed = fill.signed
    split.capture += (fill.mid_at_fill - fill.price) * signed
    for h in HORIZONS:
        index = min(fill.seq + h, len(mids) - 1)
        drift = (mids[index] - fill.mid_at_fill) * signed
        split.adverse[h] = split.adverse.get(h, 0.0) + drift


def evaluate(run: MMRun) -> MMMetrics:
    m = MMMetrics(name=run.name)
    if not run.mids:
        return m

    for fill in run.fills:
        _accumulate(m.all, fill, run.mids)
        _accumulate(m.informed if fill.informed else m.benign, fill, run.mids)

    m.final_equity = run.equity_curve[-1] if run.equity_curve else 0.0
    m.end_inventory = run.inventory_curve[-1] if run.inventory_curve else 0.0

    peak, worst = float("-inf"), 0.0
    for value in run.equity_curve:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    m.max_drawdown = -worst

    ticks = len(run.mids)
    m.quoting_pct = run.quoting_ticks / ticks * 100
    m.defensive_pct = run.defensive_ticks / ticks * 100

    strategy = run.strategy
    lats = list(getattr(strategy, "latencies", []) or [])
    m.calls = getattr(strategy, "calls", 0)
    m.errors = getattr(strategy, "errors", 0)
    m.tokens = getattr(strategy, "tokens", 0)
    m.cost_usd = usd_cost(m.tokens)
    if lats:
        ordered = sorted(lats)
        m.latency_p50 = statistics.median(ordered)
        m.latency_p95 = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
    return m


def render(rows: list[MMMetrics], horizon: int = 20) -> str:
    head = (
        f"{'arm':<10}{'net':>10}{'capture':>10}{'adverse':>10}"
        f"{'fills':>7}{'toxic':>7}{'tox adv':>10}{'quoting':>9}{'maxDD':>9}"
    )
    lines = [head, "-" * len(head)]
    for m in rows:
        lines.append(
            f"{m.name:<10}{m.final_equity:>10,.1f}{m.all.capture:>10,.1f}"
            f"{m.all.adverse.get(horizon, 0.0):>10,.1f}"
            f"{m.all.fills:>7}{m.informed.fills:>7}"
            f"{m.informed.adverse.get(horizon, 0.0):>10,.1f}"
            f"{m.quoting_pct:>8.0f}%{m.max_drawdown:>9,.1f}"
        )
    return "\n".join(lines)
