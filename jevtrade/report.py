"""Text rendering of a run."""

from __future__ import annotations

from .engine import EngineResult
from .metrics import Metrics

MOCK_WARNING = (
    "!! provider=mock -- these answers came from a hand-written scoring\n"
    "!! function, NOT from Jev. Numbers below describe this repo's plumbing\n"
    "!! and say nothing about the model. Set TYPESAFE_API_KEY for the real one."
)


def _money(value: float) -> str:
    return f"{value:>12,.2f}"


def render(result: EngineResult, metrics: Metrics, *, title: str = "") -> str:
    lines: list[str] = []
    if title:
        lines += [title, "=" * len(title)]
    if result.provider == "mock":
        lines += [MOCK_WARNING, ""]

    hit = (
        f"{metrics.hit_rate:.1%} over {metrics.scored_decisions} calls"
        if metrics.scored_decisions
        else "not enough directional calls to score"
    )
    executed = (
        f"{metrics.executed_hit_rate:.1%}"
        if metrics.executed_hit_rate == metrics.executed_hit_rate
        else "-"
    )

    lines += [
        f"instrument      {result.symbol}  ({result.interval_ms} ms snapshots)",
        f"provider        {result.provider} / {result.model}",
        f"ticks           {result.ticks:,}",
        "",
        "P&L",
        f"  gross       {_money(metrics.gross_pnl)}",
        f"  fees        {_money(-metrics.fees)}",
        f"  net         {_money(metrics.net_pnl)}   "
        f"({metrics.return_on_capital_pct:+.3f}% of capital at risk)",
        f"  buy & hold  {_money(metrics.buy_and_hold)}   (same capital, same window)",
        "",
        "Risk",
        f"  max drawdown  {metrics.max_drawdown:,.2f}",
        f"  sharpe        {metrics.sharpe:,.1f}  (annualised from per-tick P&L)",
        f"  turnover      {metrics.turnover:,.0f}  over {metrics.fills} fills",
        f"  break-even fee {metrics.break_even_fee_bps:>6.3f} bps"
        f"   (the strategy is viable only below this taker fee)",
        f"  stops fired   {metrics.stops}"
        + ("   KILL SWITCH TRIPPED" if metrics.halted else ""),
        "",
        "Decisions",
        f"  made          {metrics.decisions}",
        f"  hit rate      {hit}   (scored where the decision was formed)",
        f"  after latency {executed}   (same calls, scored where they were filled)",
        f"  mean |edge|   {metrics.mean_abs_edge:.3f}",
        f"  dropped late  {metrics.expired}"
        f"   skipped (busy) {metrics.skipped_busy}"
        f"   errors {metrics.errors}",
        "",
        "Model latency (ms)",
        f"  p50 {metrics.latency_p50:7.1f}   p95 {metrics.latency_p95:7.1f}"
        f"   p99 {metrics.latency_p99:7.1f}",
        "",
        "Cost",
        f"  {metrics.input_tokens:,} input tokens = ${metrics.cost_usd:.4f}"
        f"  (${metrics.cost_usd / max(metrics.decisions, 1):.6f} per decision)",
    ]

    if metrics.calibration:
        lines += ["", "Calibration of P(up | directional)"]
        lines += ["  bucket      n    predicted   realised"]
        for bucket in metrics.calibration:
            lines.append(
                f"  {bucket.low:.1f}-{bucket.high:.1f}  {bucket.n:5d}     "
                f"{bucket.predicted:.2f}       {bucket.realized:.2f}"
            )
    return "\n".join(lines)


def render_comparison(rows: list[tuple[str, Metrics]]) -> str:
    header = (
        f"{'strategy':<16}{'net':>12}{'gross':>12}{'fees':>11}"
        f"{'fills':>8}{'maxDD':>11}{'hit':>8}"
    )
    lines = [header, "-" * len(header)]
    for name, m in rows:
        hit = f"{m.hit_rate:.1%}" if m.scored_decisions else "-"
        lines.append(
            f"{name:<16}{m.net_pnl:>12,.2f}{m.gross_pnl:>12,.2f}"
            f"{m.fees:>11,.2f}{m.fills:>8}{m.max_drawdown:>11,.2f}{hit:>8}"
        )
    return "\n".join(lines)
