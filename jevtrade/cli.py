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


def cmd_mm(args: argparse.Namespace) -> int:
    """Four arms, identical markets, measured by markout."""
    import math
    import statistics

    from .mm.cache import CachingClient
    from .mm.engine import run as mm_run
    from .mm.market import MarketConfig
    from .mm.metrics import evaluate as mm_evaluate
    from .mm.strategies import (
        JevConfig,
        JevStrategy,
        KeywordStrategy,
        NaiveStrategy,
        VolStrategy,
    )

    if args.provider == "mock":
        from .mm.mock import MockHeadlineClient

        client = MockHeadlineClient()
    else:
        try:
            client = resolve_client(args.provider)
        except JevError:
            from .mm.mock import MockHeadlineClient

            client = MockHeadlineClient()
    # One fixed set of model answers across every arm and every market, so the
    # comparison isolates the strategy rather than the model's sampling noise.
    shared = CachingClient(client)

    seeds = [args.first_seed + i for i in range(args.seeds)]
    results: dict[str, list[float]] = {}
    acted = on_material = material_seen = 0

    for seed in seeds:
        config = MarketConfig(
            n_ticks=args.ticks,
            seed=seed,
            toxicity=args.toxicity,
            event_rate_per_1000=args.event_rate,
            event_impact_bps=args.event_impact_bps,
        )
        arms = (
            ("naive", NaiveStrategy()),
            ("vol", VolStrategy()),
            ("keyword", KeywordStrategy()),
            ("jev", JevStrategy(shared, JevConfig(risk_floor=args.risk_floor))),
        )
        for name, strategy in arms:
            run_result = mm_run(strategy, market=config)
            results.setdefault(name, []).append(
                mm_evaluate(run_result).final_equity
            )
            if name != "jev":
                continue
            events = {e.seq: e for e in run_result.events}
            for entry in run_result.posture_log:
                event = events.get(entry["seq"])
                if event is None:
                    continue
                material_seen += int(event.material)
                if entry["stance"] != "normal":
                    acted += 1
                    on_material += int(event.material)

    def stats(values: list[float]) -> tuple[float, float]:
        mean = statistics.fmean(values)
        err = (
            statistics.pstdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
        return mean, err

    if shared.provider == "mock":
        print(
            "!! provider=mock -- the headline stub is a keyword matcher with no\n"
            "!! language understanding, so the 'jev' arm here is roughly the\n"
            "!! 'keyword' arm. Set TYPESAFE_API_KEY to measure the real model.\n"
        )
    print(
        f"{len(seeds)} independent markets x {args.ticks:,} ticks  |  "
        f"toxicity {args.toxicity}  events {args.event_rate}/1000  "
        f"impact {args.event_impact_bps} bp"
    )
    print(f"\n{'arm':<10}{'mean net':>12}{'std error':>12}{'vs naive':>12}")
    print("-" * 46)
    base, _ = stats(results["naive"])
    for name in ("naive", "vol", "keyword", "jev"):
        mean, err = stats(results[name])
        print(f"{name:<10}{mean:>12,.0f}{err:>12,.0f}{mean - base:>+12,.0f}")

    paired = [j - k for j, k in zip(results["jev"], results["keyword"])]
    diff, diff_err = stats(paired)
    verdict = "inside the noise" if abs(diff) < 2 * diff_err else "outside the noise"
    print(f"\npaired jev - keyword: {diff:+,.0f} +- {diff_err:,.0f}  ({verdict})")
    if acted:
        print(
            f"jev precision {on_material / acted:.0%} "
            f"({on_material}/{acted} stances on genuinely material news)   "
            f"recall {on_material / max(material_seen, 1):.0%}"
        )
    return 0


def cmd_news(args: argparse.Namespace) -> int:
    """Measure the assumption the market-making experiment rests on."""
    from .news.feeds import collect
    from .news.label import fetch_bars, label
    from .news.study import crux, jev_scores, keyword_hit, score_arm

    headlines, errors = collect()
    bars = fetch_bars(interval=args.interval)
    rows = label(headlines, bars, horizon_bars=args.horizon)
    if not rows:
        print("no headlines fell inside the available price history", file=sys.stderr)
        return 1

    window_h = (rows[-1].headline.ts - rows[0].headline.ts) / 3600
    print(
        f"{len(rows)} headlines over {window_h:.0f}h from "
        f"{len(collect.__globals__['FEEDS']) - len(errors)} feeds, "
        f"against {args.interval}m BTC bars (sigma {bars.sigma * 1e4:.1f} bp)"
    )
    if errors:
        print(f"  feeds that failed: {', '.join(errors)}")

    client = resolve_client(args.provider)
    scores = jev_scores(rows, client)
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    keyword = [r for r in rows if keyword_hit(r.headline.title)]
    top = [rows[i] for i in order[: args.top]]

    print(
        f"\nHow often is a headline followed by a >={args.threshold} sigma move in the next "
        f"{args.horizon * args.interval} min,\nwithout one already underway? "
        f"'null' is randomly timed fake headlines."
    )
    print(f"\n{'arm':<20}{'alerts':>8}{'fire':>7}{'moved':>7}{'null':>8}{'z':>7}")
    print("-" * 57)
    for name, selected in (
        ("all headlines", rows),
        ("keyword rule", keyword),
        (f"jev top-{args.top}", top),
    ):
        result = score_arm(
            name, selected, rows, bars,
            threshold=args.threshold, horizon_bars=args.horizon,
        )
        print(
            f"{name:<20}{result.selected:>8}{result.fire_rate:>6.0%}"
            f"{result.rate:>7.1%}{result.null_rate:>8.1%}{result.z:>+7.2f}"
        )

    span = (rows[0].headline.ts, rows[-1].headline.ts)
    print(
        "\nThe crux: does a headline predict movement once the tape's own recent"
        "\nvolatility is held fixed? Controls are drawn from moments with a"
        "\ncomparable move already behind them."
    )
    print(f"\n{'group':<14}{'condition':<17}{'n':>5}{'after':>8}{'null':>8}{'z':>7}")
    print("-" * 59)
    for name, selected in ((f"jev top-{args.top}", top), ("keyword", keyword)):
        for row in crux(selected, bars, span, group=name):
            print(
                f"{row.group:<14}{row.condition:<17}{row.n:>5}{row.after:>8.2f}"
                f"{row.null:>8.2f}{row.z:>+7.2f}"
            )

    print("\nhighest-risk headlines as Jev read them:")
    for i in order[:6]:
        row = rows[i]
        moved = "moved" if row.clean_mover(args.threshold) else "  -  "
        print(f"  {scores[i]:.3f} {moved}  {row.headline.title[:66]}")
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

    mm = sub.add_parser("mm", help="market-making experiment (news + tape)")
    mm.add_argument("--seeds", type=int, default=8, help="independent markets")
    mm.add_argument("--first-seed", type=int, default=101)
    mm.add_argument("--ticks", type=int, default=6000)
    mm.add_argument("--provider", default="auto", choices=("auto", "jev", "mock"))
    mm.add_argument("--toxicity", type=float, default=1.0,
                    help="how much informed flow; 0 = nobody knows anything")
    mm.add_argument("--event-rate", type=float, default=12.0,
                    help="headlines per 1000 ticks")
    mm.add_argument("--event-impact-bps", type=float, default=22.0,
                    help="0 makes every headline cosmetic (the falsifiability run)")
    mm.add_argument("--risk-floor", type=float, default=0.12)
    mm.set_defaults(func=cmd_mm)

    news = sub.add_parser("news", help="does real news flow carry tradeable signal?")
    news.add_argument("--provider", default="auto", choices=("auto", "jev", "mock"))
    news.add_argument("--interval", type=int, default=5, help="price bar minutes")
    news.add_argument("--horizon", type=int, default=3, help="bars after a headline")
    news.add_argument("--threshold", type=float, default=2.0, help="sigmas = 'moved'")
    news.add_argument("--top", type=int, default=40, help="headlines jev flags")
    news.set_defaults(func=cmd_news)

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
