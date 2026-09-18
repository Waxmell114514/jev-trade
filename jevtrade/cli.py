"""Command line entry points.

    jevtrade backtest   run the loop over a tick stream and score it
    jevtrade decide     make one decision and print the whole exchange
    jevtrade sweep      measure what latency does to the strategy
    jevtrade fetch      pull real BTC/ETH bars from Kraken into a CSV
    jevtrade models     list the models the API key can reach
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Iterable

from .baselines import registry, run_rule
from .engine import EngineConfig, EngineResult, run
from .execution import ExecutionConfig
from .feed import CsvReplayFeed, SyntheticConfig, SyntheticFeed, fetch_kraken_ohlc, write_csv
from .jev.client import JevError, resolve_client
from .metrics import Metrics, evaluate
from .policy import PolicyConfig
from .questions import trading_questions
from .report import render, render_comparison
from .types import Tick

SWEEP_DEFAULT = "0,50,150,300,600,1200,3000"


# ------------------------------------------------------------------ plumbing


def _load_ticks(args: argparse.Namespace) -> tuple[list[Tick], int, str, bool]:
    """Returns (ticks, interval_ms, symbol, book_is_reconstructed)."""
    if args.source == "csv":
        feed = CsvReplayFeed(args.csv, limit=args.ticks or None)
        return list(feed), feed.interval_ms, feed.symbol, True
    if args.source == "live":
        ticks = fetch_kraken_ohlc(args.symbol, interval_min=args.interval_min)
        if args.ticks:
            ticks = ticks[-args.ticks :]
        return (
            ticks,
            args.interval_min * 60_000,
            f"{args.symbol.upper()}-USD",
            True,
        )
    config = SyntheticConfig(
        symbol=f"{args.symbol.upper()}-USD",
        start_price=args.start_price,
        n_ticks=args.ticks or 3000,
        interval_ms=args.interval_ms,
        seed=args.seed,
        alpha=args.alpha,
    )
    return list(SyntheticFeed(config)), config.interval_ms, config.symbol, False


def _configs(args: argparse.Namespace, reconstructed: bool):
    policy = PolicyConfig(max_units=args.max_units)
    execution = ExecutionConfig(taker_fee_bps=args.fee_bps)
    engine = EngineConfig(
        decide_every=args.decide_every,
        deadline_ms=args.deadline_ms,
        extra_latency_ms=args.extra_latency_ms,
        horizon_ticks=args.horizon,
        reconstructed_book=reconstructed,
    )
    return policy, execution, engine


def _client(args: argparse.Namespace):
    return resolve_client(args.provider, latency_ms=args.mock_latency_ms)


def _metrics(result: EngineResult, args: argparse.Namespace) -> Metrics:
    # Capital at risk is one full clip, not one whole coin.
    capital = args.max_units * (result.mids[0] if result.mids else 0.0)
    return evaluate(result, horizon_ticks=args.horizon, capital=capital or None)


# ------------------------------------------------------------------ commands


def cmd_backtest(args: argparse.Namespace) -> int:
    ticks, interval_ms, symbol, reconstructed = _load_ticks(args)
    policy, execution, engine = _configs(args, reconstructed)
    client = _client(args)

    result = run(
        ticks,
        client,
        policy=policy,
        execution=execution,
        engine=engine,
        symbol=symbol,
        interval_ms=interval_ms,
    )
    metrics = _metrics(result, args)

    if args.json:
        print(json.dumps(_as_json(result, metrics), indent=2))
        return 0

    print(render(result, metrics, title=f"jev-trade backtest: {symbol}"))

    if args.baselines:
        rows: list[tuple[str, Metrics]] = [(f"jev ({client.provider})", metrics)]
        for name, signal in registry().items():
            baseline = run_rule(
                ticks,
                signal,
                name=name,
                policy=policy,
                execution=execution,
                engine=engine,
                symbol=symbol,
                interval_ms=interval_ms,
            )
            rows.append((name, _metrics(baseline, args)))
        print("\n\nSame ticks, same fees, same one-tick execution delay")
        print(render_comparison(rows))
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    """One decision, fully unpacked. The clearest view of the integration."""
    from .discretize import build_state
    from .features import FeatureWindow
    from .policy import decide

    ticks, interval_ms, symbol, reconstructed = _load_ticks(args)
    policy, _execution, engine = _configs(args, reconstructed)

    window = FeatureWindow(size=engine.window_size, warmup=engine.warmup_ticks)
    features = None
    for tick in ticks:
        features = window.update(tick) or features
    if features is None:
        print("not enough ticks to warm up", file=sys.stderr)
        return 1

    state = build_state(
        features,
        position_units=args.position,
        max_units=policy.max_units,
        interval_ms=interval_ms,
        reconstructed_book=reconstructed,
    )
    client = _client(args)

    print("--- state sent to Jev " + "-" * 45)
    print(json.dumps(state, indent=2))
    print("\n--- questions " + "-" * 53)
    for key, question in trading_questions().items():
        print(f"  {key:<16} {question['type']}")
    print(f"\n--- answers from {client.provider}/{client.model} " + "-" * 30)

    try:
        response = client.evaluate(state)
    except JevError as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    for key, answer in response.answers.items():
        if answer.type == "noul":
            print(f"  {key:<16} {answer.noul:.3f}")
        elif answer.type == "choice":
            probs = "  ".join(
                f"{k}={v:.3f}" for k, v in sorted(answer.probabilities.items())
            )
            print(
                f"  {key:<16} {answer.choice:<16} conf={answer.confidence:.3f}"
                f"   [{probs}]"
            )
        else:
            print(
                f"  {key:<16} {answer.score:.2f}/{len(answer.legend) - 1}"
                f"        conf={answer.confidence:.3f}"
            )

    decision = decide(features, response, args.position, policy)
    print("\n--- decision from code " + "-" * 44)
    print(f"  latency        {response.latency_ms:.1f} ms")
    print(f"  edge           {decision.edge:+.3f}")
    print(f"  conviction     {decision.conviction:.3f}")
    print(f"  gate           {decision.gate or '(none)'}")
    print(f"  position       {args.position:+.4f} -> {decision.target_units:+.4f} units")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """What does added latency cost this strategy?"""
    ticks, interval_ms, symbol, reconstructed = _load_ticks(args)
    policy, execution, engine = _configs(args, reconstructed)
    client = _client(args)

    values = [float(v) for v in args.latencies.split(",")]
    # P&L on a single path is noisy enough to hide the effect, so average over
    # several synthetic paths. Replayed data has only the one path.
    seeds = [args.seed]
    if args.source == "synthetic" and args.seeds > 1:
        seeds = [args.seed + i for i in range(args.seeds)]

    datasets = []
    for seed in seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        datasets.append(_load_ticks(seed_args)[0])

    header = (
        f"{'added latency':>14}{'decisions':>11}{'dropped':>9}{'skipped':>9}"
        f"{'formed':>9}{'executed':>10}{'gross':>11}{'net':>11}"
    )
    deadline = "off" if engine.deadline_ms <= 0 else f"{engine.deadline_ms:.0f} ms"
    print(
        f"latency sweep: {symbol}, {interval_ms} ms snapshots, deadline={deadline}, "
        f"{len(datasets)} path(s)"
    )
    print(header)
    print("-" * len(header))

    for extra in values:
        engine.extra_latency_ms = extra
        runs = []
        for ticks_for_seed in datasets:
            result = run(
                ticks_for_seed,
                client,
                policy=policy,
                execution=execution,
                engine=engine,
                symbol=symbol,
                interval_ms=interval_ms,
            )
            runs.append(_metrics(result, args))

        def avg(get):
            values_ = [get(m) for m in runs if get(m) == get(m)]
            return sum(values_) / len(values_) if values_ else float("nan")

        formed = avg(lambda m: m.hit_rate)
        executed = avg(lambda m: m.executed_hit_rate)
        print(
            f"{extra:>11.0f} ms{avg(lambda m: m.decisions):>11.0f}"
            f"{avg(lambda m: m.expired):>9.0f}{avg(lambda m: m.skipped_busy):>9.0f}"
            f"{formed:>8.1%}{executed:>10.1%}"
            f"{avg(lambda m: m.gross_pnl):>11,.2f}{avg(lambda m: m.net_pnl):>11,.2f}"
        )
    print(
        "\n'formed'   = the call was right, scored from the tick it was made on."
        "\n'executed' = the same call, scored from the tick it actually reached"
        "\n             the market. The gap between the two columns is the cost"
        "\n             of latency: the judgement stays good, the market moves on."
        "\n\nOnly one request is ever in flight, so a slow provider also gets"
        "\nfewer looks at the market ('skipped'). With a deadline set, late"
        "\ndecisions are dropped outright ('dropped') rather than acted on."
    )
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    ticks = fetch_kraken_ohlc(args.symbol, interval_min=args.interval_min)
    count = write_csv(args.out, ticks)
    first, last = ticks[0], ticks[-1]
    print(f"wrote {count} bars to {args.out}")
    print(f"  {first.symbol}  {first.last:,.2f} -> {last.last:,.2f}")
    print("  note: bid/ask and sizes are reconstructed from OHLCV, not real L1")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    client = resolve_client("jev")
    for model in client.list_models():
        print(f"  {model.get('name'):<14} {model.get('release_date', ''):<12} "
              f"{model.get('description', '')}")
    return 0


def _as_json(result: EngineResult, metrics: Metrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["calibration"] = [asdict(b) for b in metrics.calibration]
    payload["provider"] = result.provider
    payload["model"] = result.model
    payload["symbol"] = result.symbol
    payload["ticks"] = result.ticks
    return payload


# --------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevtrade",
        description="Simulated high-frequency trading driven by TypeSafe AI's Jev.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbol", default="BTC", help="BTC, ETH, SOL")
        p.add_argument(
            "--source", default="synthetic", choices=("synthetic", "csv", "live")
        )
        p.add_argument("--csv", default="data/ticks.csv")
        p.add_argument("--ticks", type=int, default=0, help="0 = all / default")
        p.add_argument("--interval-ms", type=int, default=250)
        p.add_argument("--interval-min", type=int, default=1, help="live bar size")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--start-price", type=float, default=76_000.0)
        p.add_argument(
            "--alpha",
            type=float,
            default=0.35,
            help="how much genuine predictability the synthetic tape has (0 = none)",
        )
        p.add_argument(
            "--provider", default="auto", choices=("auto", "jev", "mock")
        )
        p.add_argument("--mock-latency-ms", type=float, default=0.0)
        p.add_argument("--max-units", type=float, default=0.1)
        p.add_argument("--fee-bps", type=float, default=1.0)
        p.add_argument("--decide-every", type=int, default=8)
        p.add_argument("--deadline-ms", type=float, default=400.0)
        p.add_argument("--extra-latency-ms", type=float, default=0.0)
        p.add_argument("--horizon", type=int, default=8)

    backtest = sub.add_parser("backtest", help="run and score the loop")
    common(backtest)
    backtest.add_argument("--baselines", action="store_true")
    backtest.add_argument("--json", action="store_true")
    backtest.set_defaults(func=cmd_backtest)

    decide = sub.add_parser("decide", help="one decision, fully unpacked")
    common(decide)
    decide.add_argument("--position", type=float, default=0.0)
    decide.set_defaults(func=cmd_decide)

    sweep = sub.add_parser("sweep", help="measure the cost of latency")
    common(sweep)
    sweep.add_argument("--latencies", default=SWEEP_DEFAULT)
    sweep.add_argument("--seeds", type=int, default=5,
                       help="average over this many synthetic paths")
    sweep.set_defaults(func=cmd_sweep, deadline_ms=0.0)

    fetch = sub.add_parser("fetch", help="download real bars from Kraken")
    fetch.add_argument("--symbol", default="BTC")
    fetch.add_argument("--interval-min", type=int, default=1)
    fetch.add_argument("--out", default="data/ticks.csv")
    fetch.set_defaults(func=cmd_fetch)

    models = sub.add_parser("models", help="list models (needs an API key)")
    models.set_defaults(func=cmd_models)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return args.func(args)
    except JevError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
