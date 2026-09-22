"""Command line entry points.

    jevtrade backtest   run the loop over a tick stream and score it
    jevtrade decide     make one decision and print the whole exchange
    jevtrade sweep      measure what latency does to the strategy
    jevtrade fetch      pull real BTC/ETH bars from Kraken into a CSV
    jevtrade listing    read exchange announcements; the tape grades every arm
    jevtrade fx         read central-bank text; spot FX grades every arm
    jevtrade models     list the models the API key can reach
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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


def cmd_listing(args: argparse.Namespace) -> int:
    """Read exchange announcements in one second; let the tape grade every arm."""
    import os
    import statistics
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from .listing import study as S
    from .listing.announcements import CATALOGS, collect
    from .listing.reader import Reader
    from .listing.store import Store

    store = Store(args.cache)
    catalogs = tuple(int(c) for c in args.catalogs.split(","))
    horizons = tuple(int(h) for h in args.horizons.split(","))
    since = time.time() - args.days * 86400
    announcements = collect(store, since=since, catalogs=catalogs)
    if args.limit:
        announcements = announcements[-args.limit:]
    if not announcements:
        print("no announcements in range", file=sys.stderr)
        return 1
    first, last = announcements[0].when, announcements[-1].when
    print(
        f"{len(announcements)} Binance announcements, {first:%Y-%m-%d} to {last:%Y-%m-%d}, "
        f"catalogues: {', '.join(CATALOGS.get(c, str(c)) for c in catalogs)}"
    )

    def make_reader() -> Reader:
        if args.provider == "mock" or (
            args.provider == "auto" and not os.environ.get("TYPESAFE_API_KEY")
        ):
            from .listing.mock import MockListingClient

            return Reader(MockListingClient())
        return Reader(resolve_client(args.provider, timeout_s=20.0))

    tag = "mock" if make_reader().client.provider == "mock" else "jev"
    started = time.perf_counter()
    readings = S.read_all(
        announcements, make_reader, store=store if not args.no_reading_cache else None,
        cache_tag=f"{tag}:v1", workers=args.workers,
    )
    provider = make_reader().client.provider
    walls = sorted(r.wall_ms for r in readings)
    print(
        f"reader ({provider}): {len(readings)} announcements, "
        f"{sum(r.rounds == 2 for r in readings)} went to round two, "
        f"median {statistics.median(walls):.0f} ms per announcement "
        f"(p90 {walls[int(0.9 * (len(walls) - 1))]:.0f} ms), "
        f"${S.cost_usd(readings):.3f} of input tokens, "
        f"{(time.perf_counter() - started):.0f}s wall for this run"
    )

    arms = [
        ("title-bot", S.bot_signals(announcements, S.title_bot)),
        ("body-bot", S.bot_signals(announcements, S.body_bot)),
        (f"reader >={args.threshold:.2f}", S.reader_signals(readings, args.threshold)),
        ("all mentions", S.mention_signals(announcements)),
    ]
    summaries: list[S.ArmSummary] = []
    outcome_pool: list[S.Outcome] = []
    for name, signals in arms:
        outcomes = S.measure(store, signals, horizons=horizons, workers=args.workers_io)
        nulls = S.measure(store, S.null_signals(outcomes, per=args.null_per),
                          horizons=horizons, workers=args.workers_io)
        summaries.append(S.summarize(name, signals, outcomes, nulls, horizons=horizons))
        outcome_pool.extend(outcomes)

    hz = "".join(f"{'+' + str(h) + 'm':>9}" for h in horizons)
    print(
        "\nSigned log return per signal, bps, entering at the open of the minute AFTER"
        "\nthe announcement. 'pre' is the 15 min before it; 'bar' is the release minute"
        "\nitself (what the fastest actors saw). z is against the same tokens and"
        "\nsides at random moments within 5 days."
    )
    print(f"\n{'arm':<16}{'signals':>8}{'traded':>7}{'pre':>7}{'bar':>7}{hz}{'hit15':>7}{'z15':>7}")
    print("-" * (16 + 8 + 7 + 7 + 7 + 9 * len(horizons) + 14))
    for s in summaries:
        cells = "".join(f"{s.fwd[h].mean:>+9.0f}" for h in horizons)
        print(
            f"{s.name:<16}{s.signals:>8}{s.measured:>7}{s.pre.mean:>+7.0f}"
            f"{s.release_bar.mean:>+7.0f}{cells}{s.hit[15] if 15 in s.hit else 0:>7.0%}"
            f"{s.z[15] if 15 in s.z else 0:>+7.1f}"
        )
        cells = "".join(f"{'+-' + format(s.fwd[h].se, '.0f'):>9}" for h in horizons)
        null = "".join(f"{s.null[h].mean:>+9.0f}" for h in horizons)
        print(f"{'  s.e.':<16}{'':>8}{'':>7}{'':>7}{'':>7}{cells}")
        print(f"{'  null':<16}{'':>8}{'':>7}{'':>7}{'':>7}{null}")

    print("\nreader threshold sweep (+15m, same cached answers):")
    print(f"{'thr':>6}{'signals':>9}{'traded':>8}{'+15m':>8}{'s.e.':>7}{'null':>8}{'z':>7}")
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        signals = S.reader_signals(readings, thr)
        outcomes = S.measure(store, signals, horizons=(15,), workers=args.workers_io)
        nulls = S.measure(store, S.null_signals(outcomes, per=args.null_per),
                          horizons=(15,), workers=args.workers_io)
        s = S.summarize(f"{thr}", signals, outcomes, nulls, horizons=(15,))
        print(
            f"{thr:>6.2f}{s.signals:>9}{s.measured:>8}{s.fwd[15].mean:>+8.0f}"
            f"{s.fwd[15].se:>7.0f}{s.null[15].mean:>+8.0f}{s.z[15]:>+7.1f}"
        )

    kinds: dict[str, int] = {}
    for r in readings:
        kinds[r.event_type] = kinds.get(r.event_type, 0) + 1
    print("\nwhat the reader thinks the feed is made of: " + ", ".join(
        f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])
    ))

    diffs = S.disagreements(readings, args.threshold, outcome_pool)
    diffs.sort(key=lambda d: -max([abs(v) for v in d.outcomes.values()] or [0.0]))
    print(f"\nwhere matching the title and reading it traded differently ({len(diffs)} of {len(readings)}); +15m bps per token:")
    for d in diffs[: args.show]:
        when = datetime.fromtimestamp(d.when, timezone.utc)
        fmt = lambda rows: ", ".join(f"{'long' if s > 0 else 'short'} {t}" for t, s in rows) or "nothing"  # noqa: E731
        moves = ", ".join(f"{t} {v:+.0f}" for t, v in d.outcomes.items()) or "no tape"
        print(f"  {when:%m-%d %H:%M} {d.title[:70]}")
        print(f"      bot: {fmt(d.bot)[:70]}")
        print(f"   reader: {fmt(d.reader)[:70]}")
        print(f"     tape: {moves[:70]}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "provider": provider, "days": args.days, "catalogs": catalogs,
            "threshold": args.threshold, "horizons": horizons,
            "announcements": len(announcements),
            "arms": [
                {"name": s.name, "signals": s.signals, "measured": s.measured,
                 "pre": asdict(s.pre), "release_bar": asdict(s.release_bar),
                 "fwd": {h: asdict(v) for h, v in s.fwd.items()},
                 "null": {h: asdict(v) for h, v in s.null.items()},
                 "hit": s.hit, "z": s.z}
                for s in summaries
            ],
            "readings": [
                {"code": r.announcement.code, "ts": r.announcement.ts,
                 "title": r.announcement.title, **S.reading_to_dict(r)}
                for r in readings
            ],
        }, ensure_ascii=False, indent=1))
        print(f"\nwrote {out}")
    return 0


def cmd_fx_context(args: argparse.Namespace) -> int:
    """Read FOMC statements *against* what the market already had.

    The absolute reader scored a coin flip on the 150 statements in the archive:
    50% at every horizon, and 43% in its most confident bucket. This arm hands
    the same tree the pre-release context and asks the relative question
    instead, next to two numeric baselines that ask it with arithmetic.
    """
    import concurrent.futures
    import os
    import statistics
    import threading
    import time
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Callable

    from .fx import context as C
    from .fx import diff as DF
    from .fx import documents as D
    from .fx import reader as R
    from .fx import study as S
    from .fx.diff import Editions
    from .fx.reader import Reader
    from .listing.store import Store

    def day_epoch(text: str) -> float:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()

    store = Store(args.cache)
    horizons = tuple(int(h) for h in args.horizons.split(","))
    tape_name = args.tape or "yahoo"
    ticks = tape_name == "dukascopy"
    now = time.time()
    since = day_epoch(args.since) if args.since else now - args.days * 86400
    until = day_epoch(args.until) + 86400 if args.until else now
    if until <= since:
        print("--until is not after --since", file=sys.stderr)
        return 1

    # --context is a Fed study by construction: the ECB, BoJ and BoE feeds are
    # RSS and fifteen items deep, so there is no prior context to assemble.
    found = D.collect(store, since=since, until=until, issuers=("fed",),
                      kinds=("monetary_policy", "speech", "testimony"),
                      limit=args.limit, workers=args.workers_io)
    documents = found.documents
    statements = S.statements(documents)
    if not statements:
        print("no FOMC statements in range", file=sys.stderr)
        return 1
    first, last = statements[0].when, statements[-1].when
    print(f"{len(statements)} FOMC statements, {first:%Y-%m-%d} to {last:%Y-%m-%d}, "
          f"out of {len(documents)} Fed documents in the window")

    # The minutes that precede the first statement were released before the
    # window opens, so the archive is read a quarter earlier than the study.
    archive_rows, _ = D.fed_archive(store, since - 120 * 86400, until=until)
    table = C.rates_h15(store)
    if not len(table):
        # H.15 is primary; the Treasury's own par-yield files are the fallback
        # and carry no funds rate, so the pricing sentences lose their anchor.
        years = range(datetime.fromtimestamp(since, timezone.utc).year,
                      datetime.fromtimestamp(until, timezone.utc).year + 1)
        table = C.rates_treasury(store, years)
        print("H.15 gave nothing; falling back to the Treasury par-yield files",
              file=sys.stderr)
    if len(table):
        print(f"rates: {len(table)} daily rows, {table.dates[0]} to {table.dates[-1]}, "
              f"series {', '.join(sorted(table.series))}")
    else:
        print("rates: none -- the context will carry no pricing sentences", file=sys.stderr)

    tapes = S.TickTapes(store, workers=args.workers_io) if ticks else None
    drift_tape = tapes.get("EURUSD=X") if tapes is not None else None
    editions = Editions(statements)
    calendar = D.calendar_rows(store)

    started = time.perf_counter()
    done = {"n": 0}
    guard = threading.Lock()

    def build(document: D.Document) -> tuple[D.Document, C.Context | None, str]:
        previous = C.previous_statement(document, statements, editions=editions)
        local = D.local_string(document.ts, "fed")[:10].replace("-", "")
        try:
            context = C.context_for(
                document, store=store, table=table, archive_rows=archive_rows,
                documents=documents, previous=previous, tape=drift_tape, local_date=local,
            )
            note = ""
        except C.LookaheadError as exc:
            context, note = None, str(exc)
        with guard:
            done["n"] += 1
            seen = done["n"]
        if seen % 25 == 0 or seen == len(statements):
            print(f"  context {seen}/{len(statements)}, "
                  f"{time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)
        return document, context, note

    with concurrent.futures.ThreadPoolExecutor(max(1, args.workers_io)) as pool:
        built = list(pool.map(build, statements))
    contexts = {d.id: c for d, c, _ in built if c is not None}
    lookahead = [(d, note) for d, c, note in built if c is None]
    previous_of_id = {d.id: contexts[d.id].previous for d in statements
                      if d.id in contexts and contexts[d.id].previous is not None}
    texts = {doc_id: ctx.as_text(args.context_chars) for doc_id, ctx in contexts.items()}

    have = {k: sum(1 for c in contexts.values() if c.has[k]) for k in
            ("rates", "dots", "previous", "minutes", "chair", "communication", "drift")}
    lengths = sorted(len(t) for t in texts.values()) or [0]
    print("context: " + ", ".join(f"{k} {v}/{len(statements)}" for k, v in have.items())
          + f"; median {lengths[len(lengths) // 2]} chars"
          + f" (cap {args.context_chars})")
    if lookahead:
        print(f"  {len(lookahead)} statement(s) refused for lookahead:")
        for document, note in lookahead[: args.show]:
            print(f"    {document.when:%Y-%m-%d} {note}")

    def make_reader(mode: str) -> Callable[[], Reader]:
        def factory() -> Reader:
            if args.provider == "mock" or (
                args.provider == "auto" and not os.environ.get("TYPESAFE_API_KEY")
            ):
                from .fx.mock import MockFxClient

                return Reader(MockFxClient(), mode=mode)
            return Reader(resolve_client(args.provider, timeout_s=20.0), mode=mode)
        return factory

    provider = make_reader(R.ABSOLUTE)().client.provider

    # How many of the absolute readings this run can reuse instead of re-asking.
    cached = 0
    for document in statements:
        previous, changes = DF.diff_for(document, editions, limit=12)
        row = D.match_calendar(document, calendar) if calendar else None
        key = (f"reading:{provider}:{R.TREE_VERSION}:{document.id}:"
               f"{S.state_hash(document, changes, row)}")
        cached += 1 if store.get(key)[0] else 0
    print(f"reader-absolute: {cached}/{len(statements)} already in the "
          f"{provider}:{R.TREE_VERSION} cache and reused as they are")

    read_started = time.perf_counter()

    def progress(tag: str):
        def report(done: int, total: int, round_two: int, cost: float) -> None:
            print(f"  {tag} {done}/{total}, {round_two} to round two, ${cost:.3f}, "
                  f"{time.perf_counter() - read_started:.0f}s", file=sys.stderr, flush=True)
        return report

    absolute = S.read_all(
        statements, make_reader(R.ABSOLUTE),
        store=None if args.no_reading_cache else store,
        cache_tag=f"{provider}:{R.TREE_VERSION}", calendar=calendar,
        workers=args.workers, progress=progress("absolute"), progress_every=50,
    )
    contextual = S.read_all(
        statements, make_reader(R.CONTEXT),
        store=None if args.no_reading_cache else store,
        cache_tag=f"{provider}:{R.CONTEXT_VERSION}", calendar=calendar,
        contexts=texts, workers=args.workers, progress=progress("context"),
        progress_every=50,
    )
    widths = sorted(r.questions_asked for r in contextual)
    walls = sorted(r.wall_ms for r in contextual)
    print(
        f"reader-context ({provider}): {len(contextual)} statements, "
        f"{sum(r.rounds == 2 for r in contextual)} went to round two, "
        f"{widths[len(widths) // 2]} questions in the median statement, "
        f"median {statistics.median(walls):.0f} ms, "
        f"${S.cost_usd(contextual):.3f} of input tokens, "
        f"{time.perf_counter() - read_started:.0f}s wall"
    )

    if ticks:
        tape = tapes
    else:
        span = max(1.0, (until - since) / 86400.0)
        tape = S.Tape(store, bar_min=args.bar,
                      days=min(int(max(span, 7)), 60 if args.bar >= 5 else 7))
    graded = dict(horizons=horizons, workers=args.workers_io, latency_s=args.latency)

    absolute_signals = S.reader_signals(absolute, args.threshold)
    context_signals = S.reader_signals(contextual, args.threshold)
    arms: list[tuple[str, list[S.Signal], str]] = [
        ("reader-absolute", absolute_signals, "the v1 tree, reused from cache"),
        ("reader-context", context_signals, "the ctx1 tree, relative to the context"),
        ("bill-surprise", S.bill_surprise_signals(statements, contexts,
                                                  previous_of_id=previous_of_id),
         "crude proxy for fed funds futures"),
        ("dots-surprise", S.dots_surprise_signals(statements, contexts),
         "projection meetings only"),
        ("all statements", S.all_text_signals(statements), "keyword sign, every statement"),
    ]
    summaries: list[S.ArmSummary] = []
    per_arm: list[list[S.Outcome]] = []
    pool_outcomes: list[S.Outcome] = []
    for name, signals, note in arms:
        outcomes = S.measure(tape, signals, **graded)
        nulls = S.measure(tape, S.null_signals(tape, outcomes, per=args.null_per,
                                               horizons=horizons, latency_s=args.latency,
                                               workers=args.workers_io), **graded)
        summaries.append(S.summarize(name, signals, outcomes, nulls, horizons=horizons,
                                     note=note, latency_s=args.latency if ticks else 0.0))
        per_arm.append(outcomes)
        pool_outcomes.extend(outcomes)

    hz = "".join(f"{'+' + str(h) + 'm':>9}" for h in horizons)
    lead = f"{'pre':>7}{'rush':>7}{'sprd':>6}" if ticks else f"{'pre':>7}{'bar':>7}"
    print(
        "\nSigned log return per signal, bps, on FOMC statements only"
        + (f", entering on the first tick at\nor after the release + {args.latency:g}s, "
           "paying the ask to go long and the bid to go short."
           if ticks else f", entering at the open of the first\n{args.bar}-minute bar after it.")
        + "\nz is against the same pair and side at random moments within 5 days."
    )
    print(f"\n{'arm':<16}{'signals':>8}{'traded':>7}{lead}{hz}{'hit15':>7}{'z15':>7}")
    print("-" * (16 + 8 + 7 + len(lead) + 9 * len(horizons) + 14))
    for s in summaries:
        cells = "".join(f"{s.fwd[h].mean:>+9.0f}" for h in horizons)
        values = (f"{s.pre.mean:>+7.0f}{s.rush.mean:>+7.0f}{s.spread.mean:>6.1f}" if ticks
                  else f"{s.pre.mean:>+7.0f}{s.release_bar.mean:>+7.0f}")
        blank = " " * len(lead)
        print(f"{s.name:<16}{s.signals:>8}{s.measured:>7}{values}{cells}"
              f"{s.hit.get(15, 0.0):>7.0%}{s.z.get(15, 0.0):>+7.1f}")
        errs = "".join(f"{'+-' + format(s.fwd[h].se, '.0f'):>9}" for h in horizons)
        null = "".join(f"{s.null[h].mean:>+9.0f}" for h in horizons)
        print(f"{'  s.e.':<16}{'':>8}{'':>7}{blank}{errs}")
        print(f"{'  null':<16}{'':>8}{'':>7}{blank}{null}")
        if s.note:
            print(f"{'  (' + s.note + ')':<16}")

    print("\nhit rate and mean signed return per arm, at every horizon:")
    print(f"{'arm':<16}{'traded':>8}"
          + "".join(f"{'hit+' + str(h) + 'm':>9}{'mean':>7}" for h in horizons))
    for s in summaries:
        print(f"{s.name:<16}{s.measured:>8}"
              + "".join(f"{s.hit.get(h, 0.0):>9.0%}{s.fwd[h].mean:>+7.0f}" for h in horizons))

    sweep: list[S.ArmSummary] = []
    if args.latency_sweep and not ticks:
        print("\n--latency-sweep needs --tape dukascopy; 5-minute bars have one latency",
              file=sys.stderr)
    elif args.latency_sweep:
        swept = context_signals if len(context_signals) >= 5 else arms[4][1]
        sweep_h = tuple(h for h in (15, 60) if h in horizons) or (horizons[-1],)
        sweep = S.latency_sweep(tape, swept, horizons=sweep_h, per=args.null_per,
                                workers=args.workers_io)
        print(f"\nlatency sweep (reader-context, {len(swept)} signals, net of the half spread):")
        cols = "".join(f"{'+' + str(h) + 'm':>8}{'s.e.':>7}{'null':>8}{'z':>6}" for h in sweep_h)
        print(f"{'entry':>8}{'traded':>8}{'rush':>7}{'sprd':>6}{cols}")
        for row in sweep:
            cells = "".join(
                f"{row.fwd[h].mean:>+8.0f}{row.fwd[h].se:>7.0f}"
                f"{row.null[h].mean:>+8.0f}{row.z[h]:>+6.1f}" for h in sweep_h)
            print(f"{row.name:>8}{row.measured:>8}{row.rush.mean:>+7.0f}"
                  f"{row.spread.mean:>6.1f}{cells}")

    actual = S.action_confusion(contextual, previous_of_id, field_name="actual_action")
    expected = S.action_confusion(contextual, previous_of_id, field_name="expected_action")
    print(f"\nwhat the model said the statement did, against the rate the code parsed"
          f" ({actual.n} statements, {actual.unparsed} unparseable):")
    print(f"{'parsed':>10}" + "".join(f"{a:>8}" for a in R.ACTIONS))
    for parsed in R.ACTIONS:
        print(f"{parsed:>10}" + "".join(
            f"{actual.rows.get((parsed, model), 0):>8}" for model in R.ACTIONS))
    print(f"  actual_action agrees with the parsed decision {actual.agreed}/{actual.n} "
          f"({actual.rate:.0%}); expected_action matches what happened "
          f"{expected.agreed}/{expected.n} ({expected.rate:.0%})")

    channels = S.channel_counts(contextual)
    print("where it put the surprise: " + (", ".join(
        f"{k} {v}" for k, v in sorted(channels.items(), key=lambda kv: -kv[1])) or "nowhere"))
    relatives: dict[str, int] = {}
    for reading in contextual:
        relatives[reading.relative] = relatives.get(reading.relative, 0) + 1
    print("relative stance: " + ", ".join(
        f"{k or 'none'} {v}" for k, v in sorted(relatives.items(), key=lambda kv: -kv[1])))

    splits = S.signal_disagreements(absolute_signals, context_signals, pool_outcomes,
                                    horizon=horizons[min(1, len(horizons) - 1)])
    splits.sort(key=lambda d: -max([abs(v) for v in d.outcomes.values()] or [0.0]))
    horizon = horizons[min(1, len(horizons) - 1)]
    print(f"\nwhere the absolute and the context readers traded differently "
          f"({len(splits)} of {len(statements)}); +{horizon}m bps per pair:")
    for d in splits[: args.show]:
        when = datetime.fromtimestamp(d.when, timezone.utc)
        fmt = lambda rows: ", ".join(  # noqa: E731
            f"{'long' if s > 0 else 'short'} {p}" for p, s in rows) or "nothing"
        moves = ", ".join(f"{p} {v:+.0f}" for p, v in d.outcomes.items()) or "no tape"
        print(f"  {when:%Y-%m-%d %H:%M} {d.title[:62]}")
        print(f"  absolute: {fmt(d.bot)[:70]}")
        print(f"   context: {fmt(d.reader)[:70]}")
        print(f"      tape: {moves[:70]}")

    if args.out:
        def outcome_row(o: S.Outcome) -> dict[str, Any]:
            return {
                "ts": o.signal.ts, "title": o.signal.title, "pair": o.signal.pair,
                "side": o.signal.sign, "strength": o.signal.strength,
                "pre_bps": o.pre_bps, "rush_bps": o.rush_bps,
                "spread_bps": o.spread_bps, "latency_s": o.latency_s,
                "entry_ts": o.entry_ts,
                "fwd_bps": {h: o.fwd_bps.get(h) for h in horizons},
                "fwd_mid_bps": {h: o.fwd_mid_bps.get(h) for h in horizons},
            }

        by_id = {r.document.id: r for r in absolute}
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "provider": provider, "tree": R.CONTEXT_VERSION, "mode": "context",
            "since": since, "until": until, "tape": tape_name,
            "latency_s": args.latency if ticks else None,
            "threshold": args.threshold, "horizons": horizons,
            "context_chars": args.context_chars,
            "statements": len(statements), "documents": len(documents),
            "coverage": have,
            "absolute_cached": cached,
            "lookahead_refused": [
                {"ts": d.ts, "title": d.title, "why": note} for d, note in lookahead
            ],
            "arms": [
                {"name": s.name, "signals": s.signals, "measured": s.measured,
                 "note": s.note, "pre": asdict(s.pre),
                 "spread": asdict(s.spread), "rush": asdict(s.rush),
                 "fwd": {h: asdict(v) for h, v in s.fwd.items()},
                 "fwd_mid": {h: asdict(v) for h, v in s.fwd_mid.items()},
                 "null": {h: asdict(v) for h, v in s.null.items()},
                 "hit": s.hit, "z": s.z,
                 "outcomes": [outcome_row(o) for o in outcomes]}
                for s, outcomes in zip(summaries, per_arm)
            ],
            "latency_sweep": [
                {"latency_s": r.latency_s, "signals": r.signals, "measured": r.measured,
                 "spread": asdict(r.spread), "rush": asdict(r.rush),
                 "fwd": {h: asdict(v) for h, v in r.fwd.items()},
                 "null": {h: asdict(v) for h, v in r.null.items()}, "z": r.z}
                for r in sweep
            ],
            "confusion": {
                "actual": {f"{k[0]}->{k[1]}": v for k, v in actual.rows.items()},
                "expected": {f"{k[0]}->{k[1]}": v for k, v in expected.rows.items()},
                "unparsed": actual.unparsed,
            },
            "surprise_channels": channels,
            "statements_read": [
                {
                    "id": r.document.id, "ts": r.document.ts, "title": r.document.title,
                    "context_sources": [
                        {"kind": s.kind, "label": s.label, "ts": s.ts,
                         "concurrent": s.concurrent}
                        for s in (contexts[r.document.id].sources
                                  if r.document.id in contexts else [])
                    ],
                    "context_chars": r.context_chars,
                    "parsed_decision": S.decision_from_rates(
                        r.document, previous_of_id.get(r.document.id)),
                    "context_reading": S.reading_to_dict(r),
                    "absolute_reading": (S.reading_to_dict(by_id[r.document.id])
                                         if r.document.id in by_id else None),
                }
                for r in contextual
            ],
        }, ensure_ascii=False, indent=1))
        print(f"\nwrote {out}")
    return 0


def _arms_table(summaries, horizons, *, ticks: bool) -> None:
    """The table every arm in this repository is printed in, one row plus its error."""
    hz = "".join(f"{'+' + str(h) + 'm':>9}" for h in horizons)
    lead = f"{'pre':>7}{'rush':>7}{'sprd':>6}" if ticks else f"{'pre':>7}{'bar':>7}"
    print(f"\n{'arm':<22}{'signals':>8}{'traded':>7}{lead}{hz}{'hit15':>7}{'z15':>7}")
    print("-" * (22 + 8 + 7 + len(lead) + 9 * len(horizons) + 14))
    for s in summaries:
        cells = "".join(f"{s.fwd[h].mean:>+9.0f}" for h in horizons)
        values = (f"{s.pre.mean:>+7.0f}{s.rush.mean:>+7.0f}{s.spread.mean:>6.1f}" if ticks
                  else f"{s.pre.mean:>+7.0f}{s.release_bar.mean:>+7.0f}")
        blank = " " * len(lead)
        # An arm with nothing in it has no hit rate and no z; the null's own
        # error would otherwise print one.
        tail = (f"{s.hit.get(15, 0.0):>7.0%}{s.z.get(15, 0.0):>+7.1f}" if s.measured
                else f"{'-':>7}{'-':>7}")
        print(f"{s.name:<22}{s.signals:>8}{s.measured:>7}{values}{cells}{tail}")
        errs = "".join(f"{'+-' + format(s.fwd[h].se, '.0f'):>9}" for h in horizons)
        null = "".join(f"{s.null[h].mean:>+9.0f}" for h in horizons)
        print(f"{'  s.e.':<22}{'':>8}{'':>7}{blank}{errs}")
        print(f"{'  null':<22}{'':>8}{'':>7}{blank}{null}")
        if s.note:
            print(f"  ({s.note})")


def cmd_fx_dots(args: argparse.Namespace) -> int:
    """Validate the dots rule out of sample and forward. No model is called at all.

    The rule came out of the context run: the sign of the change in next year's
    median funds-rate projection against the previous SEP, +19 +- 6 bp of
    EURUSD at fifteen minutes over 31 projection meetings. This command tests
    it where it was not found -- the histogram years before the printed median,
    two more pairs, six entry latencies -- and prints the meetings still ahead
    so that running it again after each one is the forward test.
    """
    import concurrent.futures
    import statistics
    import time
    from pathlib import Path

    from .fx import dots as X
    from .fx import study as S
    from .listing.store import Store

    store = Store(args.cache)
    horizons = X.horizon_default(args.horizons)
    # Both of these modes reach years back, where there are no 5-minute bars, so
    # ticks are the default judge and ``--tape yahoo`` is the deliberate opt-out.
    tape_name = args.tape or "dukascopy"
    ticks = tape_name == "dukascopy"
    now = time.time()
    since = X.day_epoch(args.since) if args.since else X.day_epoch("2012-01-01")
    until = X.day_epoch(args.until) + 86400 if args.until else now
    if until <= since:
        print("--until is not after --since", file=sys.stderr)
        return 1
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    unknown = [p for p in pairs if p not in X.USD_SIDE]
    if unknown:
        print(f"unknown pair(s): {', '.join(unknown)}; known: "
              f"{', '.join(sorted(X.USD_SIDE))}", file=sys.stderr)
        return 1

    meetings = X.meeting_dates(store, since=since, until=until)
    if not meetings:
        print("no FOMC statements in range", file=sys.stderr)
        return 1
    print(f"{len(meetings)} FOMC statements, {X.pretty_date(meetings[0]['date'])} to "
          f"{X.pretty_date(meetings[-1]['date'])} (the window is opened "
          f"{int(X.previous_window(0) / -86400)} days early so the first meeting has a "
          f"previous SEP)")

    def probe(row):
        return X.probe_projection(store, row["date"], ts=row["ts"])

    with concurrent.futures.ThreadPoolExecutor(max(1, args.workers_io)) as pool:
        probed = list(pool.map(probe, meetings))
    projections = [p for p, _why in probed if p is not None]
    unreadable = [(row["date"], why) for row, (p, why) in zip(meetings, probed)
                  if p is None and "no projection page" not in why]
    early = [p for p in projections if not p.in_sample]
    late = [p for p in projections if p.in_sample]
    print(f"projection tables: {len(projections)} parsed "
          f"({len(early)} before the printed median row, {len(late)} from "
          f"{X.pretty_date(X.FIRST_PRINTED_MEDIAN)} on)")
    if unreadable:
        print(f"  {len(unreadable)} page(s) answered and did not parse:")
        for date, why in unreadable[: args.show]:
            print(f"    {X.pretty_date(date)}: {why}")

    agreed = compared = 0
    notes: list[str] = []
    with_printed = 0
    for projection in projections:
        if not projection.printed:
            continue
        with_printed += 1
        a, c, n = projection.agreement()
        agreed, compared, notes = agreed + a, compared + c, notes + n
    print(f"histogram against the printed median: {agreed}/{compared} year-cells agree "
          f"over {with_printed} meetings that print both")
    for note in notes[: args.show]:
        print(f"    {note}")

    rows = X.records(projections)
    print(f"{X.describe_window(rows)}; the rule signs "
          f"{sum(1 for r in rows if r.rule is not None and r.rule.sign)} of them")

    tapes = S.TickTapes(store, workers=args.workers_io) if ticks else S.Tape(
        store, bar_min=args.bar, days=60)
    graded = dict(horizons=horizons, workers=args.workers_io, latency_s=args.latency)
    summaries: list[S.ArmSummary] = []
    per_pair: dict[str, list[S.Outcome]] = {}
    sweep: list[S.ArmSummary] = []

    def sample_of(code: str) -> str:
        date = code.rsplit(":", 1)[-1]
        return "in-sample" if date >= X.FIRST_PRINTED_MEDIAN else "out-of-sample"

    for pair in pairs:
        pooled = X.signals(rows, pair)
        outcomes = S.measure(tapes, pooled, **graded)
        nulls = S.measure(tapes, S.null_signals(tapes, outcomes, per=args.null_per,
                                                horizons=horizons, latency_s=args.latency,
                                                workers=args.workers_io), **graded)
        per_pair[pair] = outcomes
        for label in ("in-sample", "out-of-sample", "pooled"):
            chosen = [o for o in outcomes
                      if label == "pooled" or sample_of(o.signal.code) == label]
            chosen_signals = [s for s in pooled
                              if label == "pooled" or sample_of(s.code) == label]
            note = "" if label != "pooled" else "both halves, one convention"
            summaries.append(S.summarize(f"{pair} {label}", chosen_signals, chosen, nulls,
                                         horizons=horizons, note=note,
                                         latency_s=args.latency if ticks else 0.0))

    print("\nSigned log return per signal, bps, on the dots rule"
          + (f", entering on the first tick at\nor after the release + {args.latency:g}s, "
             "paying the ask to go long and the bid to go short."
             if ticks else f", at the open of the first {args.bar}-minute bar after it.")
          + "\nz is against the same pair and side at random moments within 5 days.")
    print(f"\n{X.RULE_SENTENCE}")
    _arms_table(summaries, horizons, ticks=ticks)
    print("The null is drawn once per pair, from the pooled trades, and the two "
          "halves share it.")

    horizon = 15 if 15 in horizons else horizons[-1]
    print(f"\nthe rule per pair and sample, at +{horizon}m: "
          f"hit rate, mean, median and the two-sided sign test")
    print(f"{'arm':<22}{'traded':>8}{'hit':>7}{'mean':>8}{'s.e.':>7}{'median':>8}{'p':>8}")
    for summary, (pair, label) in zip(
            summaries, [(p, s) for p in pairs
                        for s in ("in-sample", "out-of-sample", "pooled")]):
        values = [o.fwd_bps[horizon] for o in per_pair[pair]
                  if label == "pooled" or sample_of(o.signal.code) == label]
        wins = sum(1 for v in values if v > 0)
        if not values:
            print(f"{pair + ' ' + label:<22}{0:>8}" + "".join(f"{'-':>7}" for _ in range(2))
                  + "".join(f"{'-':>8}" for _ in range(3)))
            continue
        print(f"{pair + ' ' + label:<22}{len(values):>8}{summary.hit.get(horizon, 0.0):>7.0%}"
              f"{summary.fwd[horizon].mean:>+8.0f}{summary.fwd[horizon].se:>7.0f}"
              f"{statistics.median(values):>+8.0f}"
              f"{X.sign_test(wins, len(values)):>8.3f}")

    # The variants, on the original pair, labelled as variants and never chosen.
    variant_rows: list[tuple[str, S.ArmSummary, list[float]]] = []
    if pairs:
        pair = pairs[0]
        print(f"\nvariants on {pair} -- reported, not chosen. The rule above is the rule; "
              f"these\nare here so a reader can see whether it is one pick out of five.")
        print(f"{'variant':<22}{'signals':>8}{'traded':>7}{'hit':>7}{'mean':>8}"
              f"{'s.e.':>7}{'p':>8}")
        for arm in X.VARIANTS:
            arm_signals = X.signals(rows, pair, arm=arm)
            arm_outcomes = S.measure(tapes, arm_signals, **graded)
            arm_nulls = S.measure(
                tapes, S.null_signals(tapes, arm_outcomes, per=args.null_per,
                                      horizons=horizons, latency_s=args.latency,
                                      workers=args.workers_io), **graded)
            summary = S.summarize(f"{pair} {arm}", arm_signals, arm_outcomes, arm_nulls,
                                  horizons=horizons, note="variant",
                                  latency_s=args.latency if ticks else 0.0)
            values = [o.fwd_bps[horizon] for o in arm_outcomes]
            wins = sum(1 for v in values if v > 0)
            variant_rows.append((arm, summary, values))
            print(f"{arm:<22}{summary.signals:>8}{summary.measured:>7}"
                  f"{summary.hit.get(horizon, 0.0):>7.0%}{summary.fwd[horizon].mean:>+8.0f}"
                  f"{summary.fwd[horizon].se:>7.0f}{X.sign_test(wins, len(values)):>8.3f}")

    if args.latency_sweep and not ticks:
        print("\n--latency-sweep needs --tape dukascopy; 5-minute bars have one latency",
              file=sys.stderr)
    elif args.latency_sweep and pairs:
        swept = X.signals(rows, pairs[0])
        sweep_h = tuple(h for h in (15, 60) if h in horizons) or (horizons[-1],)
        sweep = S.latency_sweep(tapes, swept, latencies=X.LATENCIES, horizons=sweep_h,
                                per=args.null_per, workers=args.workers_io)
        print(f"\nlatency sweep ({pairs[0]}, {len(swept)} signals, net of the half spread).")
        print("Five minutes is the question: can somebody who reads the table by hand "
              "still catch it?")
        cols = "".join(f"{'+' + str(h) + 'm':>8}{'s.e.':>7}{'null':>8}{'z':>6}" for h in sweep_h)
        print(f"{'entry':>8}{'traded':>8}{'rush':>7}{'sprd':>6}{cols}")
        for row in sweep:
            cells = "".join(
                f"{row.fwd[h].mean:>+8.0f}{row.fwd[h].se:>7.0f}"
                f"{row.null[h].mean:>+8.0f}{row.z[h]:>+6.1f}" for h in sweep_h)
            print(f"{row.name:>8}{row.measured:>8}{row.rush.mean:>+7.0f}"
                  f"{row.spread.mean:>6.1f}{cells}")

    by_code: dict[str, dict[str, float]] = {}
    for pair, outcomes in per_pair.items():
        for outcome in outcomes:
            date = outcome.signal.code.rsplit(":", 1)[-1]
            by_code.setdefault(date, {})[pair] = outcome.fwd_bps.get(horizon, 0.0)
    for row in rows:
        row.outcomes = by_code.get(row.date, {})

    print(f"\nper meeting: next year's median now and at the previous SEP, the rule's "
          f"sign, and\neach pair's +{horizon}m in bps (blank where the tape could not "
          f"answer).")
    head = "".join(f"{p:>9}" for p in pairs)
    print(f"{'meeting':>11}{'prev SEP':>11}{'year':>7}{'now':>7}{'was':>7}{'sign':>6}{head}")
    for row in rows:
        rule = row.rule
        cells = "".join(
            f"{row.outcomes[p]:>+9.0f}" if p in row.outcomes else f"{'-':>9}" for p in pairs)
        if rule is None:
            print(f"{X.pretty_date(row.date):>11}{X.pretty_date(row.previous_date):>11}"
                  f"{'-':>7}{'-':>7}{'-':>7}{'-':>6}{cells}")
            continue
        print(f"{X.pretty_date(row.date):>11}{X.pretty_date(row.previous_date):>11}"
              f"{rule.year:>7}{rule.now:>7.2f}{rule.previous:>7.2f}{rule.sign:>+6d}{cells}")

    schedule = X.calendar(store)
    ahead = X.next_projection_meetings(schedule, now)
    print(f"\n{X.RULE_SENTENCE}")
    if ahead:
        print("The next projection meetings, from the FOMC calendar -- running this "
              "command\nagain after each one is the forward test:")
        for meeting in ahead:
            print(f"    {meeting.date}")
    else:
        print("The calendar lists no projection meeting after today.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        stored: list[dict[str, Any]] = []
        if out.exists():
            try:
                stored = json.loads(out.read_text()).get("meetings", [])
            except (ValueError, OSError):
                stored = []
        payload = {
            "mode": "dots", "rule": X.RULE, "variants": list(X.VARIANTS),
            "rule_sentence": X.RULE_SENTENCE,
            "since": since, "until": until, "tape": tape_name,
            "latency_s": args.latency if ticks else None,
            "horizons": list(horizons), "pairs": pairs,
            "usd_side": X.USD_SIDE,
            "first_printed_median": X.FIRST_PRINTED_MEDIAN,
            "projections": len(projections),
            "unreadable": [{"date": d, "why": w} for d, w in unreadable],
            "printed_agreement": {"agreed": agreed, "compared": compared,
                                  "meetings": with_printed, "notes": notes},
            "arms": [
                {"name": s.name, "signals": s.signals, "measured": s.measured,
                 "note": s.note, "pre": asdict(s.pre), "spread": asdict(s.spread),
                 "rush": asdict(s.rush),
                 "fwd": {h: asdict(v) for h, v in s.fwd.items()},
                 "fwd_mid": {h: asdict(v) for h, v in s.fwd_mid.items()},
                 "null": {h: asdict(v) for h, v in s.null.items()},
                 "hit": s.hit, "z": s.z}
                for s in summaries
            ],
            "variant_arms": [
                {"name": arm, "signals": s.signals, "measured": s.measured,
                 "hit": s.hit, "fwd": {h: asdict(v) for h, v in s.fwd.items()},
                 "sign_test": X.sign_test(sum(1 for v in values if v > 0), len(values))}
                for arm, s, values in variant_rows
            ],
            "latency_sweep": [
                {"latency_s": r.latency_s, "signals": r.signals, "measured": r.measured,
                 "spread": asdict(r.spread), "rush": asdict(r.rush),
                 "fwd": {h: asdict(v) for h, v in r.fwd.items()},
                 "null": {h: asdict(v) for h, v in r.null.items()}, "z": r.z}
                for r in sweep
            ],
            "next_projection_meetings": [m.date for m in ahead],
            "meetings": X.merge_records(stored, [X.record_to_dict(r) for r in rows]),
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        print(f"\nwrote {out} ({len(payload['meetings'])} meetings in the register)")
    return 0


def cmd_fx_presser(args: argparse.Namespace) -> int:
    """Read the press conference that starts half an hour after the statement.

    2022-11-02 is the case: the statement read one way and the press conference
    the other, and the reader that had only the statement was short EURUSD 66 bp
    into the wrong side of it. This mode gives the tree the statement, the dots
    and the transcript split into the Chair's opening remarks and the Q&A, and
    grades from the moment the Chair started speaking.

    A transcript is published after the conference, so every arm here measures
    whether the words were worth hearing and not whether they could have been
    traded; a live speech-to-text feed is what would change that.
    """
    import concurrent.futures
    import os
    import statistics
    import time
    from pathlib import Path

    from .fx import context as C
    from .fx import documents as D
    from .fx import dots as X
    from .fx import presser as P
    from .fx import reader as R
    from .fx import study as S
    from .fx.reader import Reader
    from .listing.store import Store

    store = Store(args.cache)
    horizons = X.horizon_default(args.horizons)
    # Both of these modes reach years back, where there are no 5-minute bars, so
    # ticks are the default judge and ``--tape yahoo`` is the deliberate opt-out.
    tape_name = args.tape or "dukascopy"
    ticks = tape_name == "dukascopy"
    now = time.time()
    since = X.day_epoch(args.since) if args.since else X.day_epoch("2011-01-01")
    until = X.day_epoch(args.until) + 86400 if args.until else now
    if until <= since:
        print("--until is not after --since", file=sys.stderr)
        return 1
    # The arms here are all on EURUSD, which is the pair the statement study and
    # the dots rule were measured on; ``--pairs`` belongs to ``--dots``. Keeping
    # one pair also keeps the statement reader's sign convention -- a hawkish
    # dollar is short EURUSD -- exactly as the absolute tree wrote it.
    pair = "EURUSD"

    found = D.collect(store, since=since, until=until, issuers=("fed",),
                      kinds=("monetary_policy",), limit=args.limit,
                      workers=args.workers_io)
    statements = S.statements(found.documents)
    if not statements:
        print("no FOMC statements in range", file=sys.stderr)
        return 1
    by_date = {X.eastern_date(d.ts): d for d in statements}
    print(f"{len(statements)} FOMC statements, {statements[0].when:%Y-%m-%d} to "
          f"{statements[-1].when:%Y-%m-%d}")

    dates = P.presser_dates(store, since=since, until=until)
    dates = [d for d in dates if d in by_date]
    starts = {d: P.presser_start(by_date[d].ts) for d in dates}
    per_year: dict[str, int] = {}
    for date in dates:
        per_year[date[:4]] = per_year.get(date[:4], 0) + 1
    print(f"press conferences on the calendar: {len(dates)}"
          + (f" ({', '.join(f'{y} {n}' for y, n in sorted(per_year.items()))})"
             if per_year else ""))

    # The dots sentences for the day, from the same reader the context mode
    # uses, so the two modes cannot disagree about what the dots said.
    def day_dots(date: str) -> tuple[str, list[str]]:
        try:
            parsed = C.dots(store, by_date[date].ts, local_date=date)
        except Exception:  # noqa: BLE001 -- no projections is the common answer
            return date, []
        return date, list(parsed.sentences) if parsed is not None else []

    with concurrent.futures.ThreadPoolExecutor(max(1, args.workers_io)) as pool:
        dots_by_date = dict(pool.map(day_dots, dates))

    pressers, coverage = P.collect(store, dates, {d: by_date[d].ts for d in dates},
                                   dots=dots_by_date, workers=args.workers_io)
    if coverage.dark:
        print(f"\npress-conference reader: DARK -- {coverage.dark}")
        print("  the transcript arms are not run; the arms that need only the "
              "start time still are")
    else:
        remarks = sorted(len(p.transcript.remarks) for p in pressers.values()) or [0]
        qa = sorted(len(p.transcript.qa) for p in pressers.values()) or [0]
        styles: dict[str, int] = {}
        for presser in pressers.values():
            styles[presser.transcript.style] = styles.get(presser.transcript.style, 0) + 1
        print(f"transcripts read: {coverage.parsed}/{len(dates)}; median remarks "
              f"{remarks[len(remarks) // 2]:,} chars, median Q&A {qa[len(qa) // 2]:,}; "
              f"Q&A marked by " + ", ".join(f"{k} {v}" for k, v in sorted(styles.items())))

    def make_reader(mode: str):
        def factory() -> Reader:
            if args.provider == "mock" or (
                args.provider == "auto" and not os.environ.get("TYPESAFE_API_KEY")
            ):
                from .fx.mock import MockFxClient

                return Reader(MockFxClient(), mode=mode)
            return Reader(resolve_client(args.provider, timeout_s=20.0), mode=mode)
        return factory

    provider = make_reader(R.PRESSER)().client.provider
    read_started = time.perf_counter()

    def progress(tag: str):
        def report(done: int, total: int, round_two: int, cost: float) -> None:
            print(f"  {tag} {done}/{total}, {round_two} to round two, ${cost:.3f}, "
                  f"{time.perf_counter() - read_started:.0f}s", file=sys.stderr, flush=True)
        return report

    # Only the statements that had a press conference are read; the rest are not
    # part of this question and would cost a request each.
    conference_days = [by_date[d] for d in dates]
    absolute = S.read_all(
        conference_days, make_reader(R.ABSOLUTE),
        store=None if args.no_reading_cache else store,
        cache_tag=f"{provider}:{R.TREE_VERSION}", workers=args.workers,
        progress=progress("statement"), progress_every=25,
    )
    readings: list = []
    if coverage.lit:
        blocks = {by_date[d].id: pressers[d].state() for d in dates if d in pressers}
        readable = [by_date[d] for d in dates if d in pressers]
        readings = S.read_all(
            readable, make_reader(R.PRESSER),
            store=None if args.no_reading_cache else store,
            cache_tag=f"{provider}:{R.PRESSER_VERSION}", pressers=blocks,
            workers=args.workers, progress=progress("presser"), progress_every=25,
        )
        widths = sorted(r.questions_asked for r in readings) or [0]
        walls = sorted(r.wall_ms for r in readings) or [0.0]
        print(f"reader-presser ({provider}): {len(readings)} conferences, "
              f"{sum(r.rounds == 2 for r in readings)} went to round two, "
              f"{widths[len(widths) // 2]} questions in the median conference, "
              f"median {statistics.median(walls):.0f} ms, "
              f"${S.cost_usd(readings):.3f} of input tokens, "
              f"{time.perf_counter() - read_started:.0f}s wall")

    tapes = S.TickTapes(store, workers=args.workers_io) if ticks else S.Tape(
        store, bar_min=args.bar, days=60)
    graded = dict(horizons=horizons, workers=args.workers_io, latency_s=args.latency)
    start_by_id = {by_date[d].id: starts[d] for d in dates}

    # The dots rule, on the projection meetings among these days, entered when
    # the Chair started rather than at the release: does the dots move continue?
    # Every projection meeting since 2012 has had a press conference, so walking
    # the conference days finds them all and ``records`` pairs consecutive ones
    # without a gap to jump.
    with concurrent.futures.ThreadPoolExecutor(max(1, args.workers_io)) as pool:
        projections = [p for p in pool.map(
            lambda date: X.fetch_projection(store, date, ts=by_date[date].ts), dates)
            if p is not None]
    dots_rows = X.records(sorted(projections, key=lambda p: p.date))
    dots_signals = []
    for signal in X.signals(dots_rows, pair):
        date = signal.code.rsplit(":", 1)[-1]
        if date in starts:
            dots_signals.append(replace(signal, ts=starts[date]))

    arms: list[tuple[str, list[S.Signal], str]] = [
        ("presser-reader",
         P.reader_signals(readings, args.threshold, start_by_id, pair=pair),
         "the pc1 tree on the transcript" if coverage.lit else "dark: no pypdf"),
        ("statement-reader",
         [replace(s, ts=start_by_id[s.code], pair=R.TICK_SYMBOLS.get(s.pair, s.pair))
          for s in S.reader_signals(absolute, args.threshold)
          if s.code in start_by_id],
         "the v1 statement read, carried into the press conference"),
        ("dots-rule", dots_signals, "the dots rule, entered when the Chair started"),
        ("all pressers", P.keyword_signals(list(pressers.values()), pair=pair),
         "keyword sign on the transcript" if coverage.lit else "dark: no pypdf"),
    ]
    summaries: list[S.ArmSummary] = []
    per_arm: list[list[S.Outcome]] = []
    for name, signals, note in arms:
        outcomes = S.measure(tapes, signals, **graded)
        nulls = S.measure(tapes, S.null_signals(tapes, outcomes, per=args.null_per,
                                                horizons=horizons, latency_s=args.latency,
                                                workers=args.workers_io), **graded)
        summaries.append(S.summarize(name, signals, outcomes, nulls, horizons=horizons,
                                     note=note, latency_s=args.latency if ticks else 0.0))
        per_arm.append(outcomes)

    print(f"\nSigned log return per signal, bps, on {pair}, entering on the first tick at "
          f"or after\nthe press conference started + {args.latency:g}s. The start is "
          f"the statement plus thirty\nminutes from 2013, and 2:15 p.m. ET in 2011 and "
          f"2012; see ``presser_start``.")
    _arms_table(summaries, horizons, ticks=ticks)

    sweep: list[S.ArmSummary] = []
    swept_arm, swept = (arms[0] if len(arms[0][1]) >= 5 else arms[1])[:2]
    if args.latency_sweep and not ticks:
        print("\n--latency-sweep needs --tape dukascopy; 5-minute bars have one latency",
              file=sys.stderr)
    elif args.latency_sweep and swept:
        sweep_h = tuple(h for h in (15, 60) if h in horizons) or (horizons[-1],)
        sweep = S.latency_sweep(tapes, swept, latencies=X.LATENCIES, horizons=sweep_h,
                                per=args.null_per, workers=args.workers_io)
        print(f"\nlatency sweep ({swept_arm}, {len(swept)} signals, net of the "
              f"half spread):")
        cols = "".join(f"{'+' + str(h) + 'm':>8}{'s.e.':>7}{'null':>8}{'z':>6}"
                       for h in sweep_h)
        print(f"{'entry':>8}{'traded':>8}{'rush':>7}{'sprd':>6}{cols}")
        for row in sweep:
            cells = "".join(
                f"{row.fwd[h].mean:>+8.0f}{row.fwd[h].se:>7.0f}"
                f"{row.null[h].mean:>+8.0f}{row.z[h]:>+6.1f}" for h in sweep_h)
            print(f"{row.name:>8}{row.measured:>8}{row.rush.mean:>+7.0f}"
                  f"{row.spread.mean:>6.1f}{cells}")

    tape = tapes.get(pair) if ticks else None
    flipped = P.reversals(readings, {r.document.id: pressers[X.eastern_date(r.document.ts)]
                                     for r in readings
                                     if X.eastern_date(r.document.ts) in pressers}, tape)
    print(f"\nwhere the press conference did not say what the statement said "
          f"({len(flipped)} of {len(readings)}):")
    print(f"{'date':>12}{'remarks vs stmt':>18}{'Q&A vs remarks':>17}{'stance':>9}"
          f"{'stmt->pc':>10}{'pc->+60m':>10}")
    shown = max(args.show, 12)
    for row in flipped[:shown]:
        before = f"{row.statement_bps:+.0f}" if row.statement_bps is not None else "-"
        after = f"{row.presser_bps:+.0f}" if row.presser_bps is not None else "-"
        print(f"{X.pretty_date(row.date):>12}{row.remarks_vs_statement:>18}"
              f"{row.qa_vs_remarks:>17}{row.presser_stance or '-':>9}{before:>10}{after:>10}")
    if len(flipped) > shown:
        print(f"  ... and {len(flipped) - shown} more; every one of them is in --out")

    topics: dict[str, int] = {}
    stances: dict[str, int] = {}
    for reading in readings:
        topics[reading.dominant_topic] = topics.get(reading.dominant_topic, 0) + 1
        stances[reading.presser_stance] = stances.get(reading.presser_stance, 0) + 1
    if readings:
        print("\nwhat the conferences were about: " + ", ".join(
            f"{k or 'none'} {v}" for k, v in sorted(topics.items(), key=lambda kv: -kv[1])))
        print("how they read for the dollar: " + ", ".join(
            f"{k or 'none'} {v}" for k, v in sorted(stances.items(), key=lambda kv: -kv[1])))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)

        def outcome_row(o: S.Outcome) -> dict[str, Any]:
            return {
                "ts": o.signal.ts, "code": o.signal.code, "pair": o.signal.pair,
                "side": o.signal.sign, "strength": o.signal.strength,
                "pre_bps": o.pre_bps, "rush_bps": o.rush_bps,
                "spread_bps": o.spread_bps, "latency_s": o.latency_s,
                "fwd_bps": {h: o.fwd_bps.get(h) for h in horizons},
                "fwd_mid_bps": {h: o.fwd_mid_bps.get(h) for h in horizons},
            }

        out.write_text(json.dumps({
            "mode": "presser", "tree": R.PRESSER_VERSION, "provider": provider,
            "since": since, "until": until, "tape": tape_name, "pair": pair,
            "latency_s": args.latency if ticks else None,
            "threshold": args.threshold, "horizons": list(horizons),
            "dark": coverage.dark,
            "conferences": len(dates), "transcripts": coverage.parsed,
            "per_year": per_year,
            "arms": [
                {"name": s.name, "signals": s.signals, "measured": s.measured,
                 "note": s.note, "pre": asdict(s.pre), "spread": asdict(s.spread),
                 "rush": asdict(s.rush),
                 "fwd": {h: asdict(v) for h, v in s.fwd.items()},
                 "null": {h: asdict(v) for h, v in s.null.items()},
                 "hit": s.hit, "z": s.z,
                 "outcomes": [outcome_row(o) for o in outcomes]}
                for s, outcomes in zip(summaries, per_arm)
            ],
            "latency_sweep": [
                {"latency_s": r.latency_s, "signals": r.signals, "measured": r.measured,
                 "fwd": {h: asdict(v) for h, v in r.fwd.items()},
                 "null": {h: asdict(v) for h, v in r.null.items()}, "z": r.z}
                for r in sweep
            ],
            "reversals": [asdict(row) for row in flipped],
            "days": [
                {
                    "date": X.eastern_date(r.document.ts),
                    "statement_ts": r.document.ts,
                    "presser_ts": start_by_id.get(r.document.id),
                    "remarks_chars": len(
                        pressers[X.eastern_date(r.document.ts)].transcript.remarks)
                    if X.eastern_date(r.document.ts) in pressers else 0,
                    "qa_chars": len(pressers[X.eastern_date(r.document.ts)].transcript.qa)
                    if X.eastern_date(r.document.ts) in pressers else 0,
                    "qa_marker_style": pressers[X.eastern_date(r.document.ts)].transcript.style
                    if X.eastern_date(r.document.ts) in pressers else "",
                    "presser_reading": S.reading_to_dict(r),
                }
                for r in readings
            ],
        }, ensure_ascii=False, indent=1))
        print(f"\nwrote {out}")
    return 0


def cmd_fx_wire(args: argparse.Namespace) -> int:
    """Read the retail FX wire -- every headline a scalper sees -- on minute candles.

    The central-bank study graded one issuer's scheduled text because the Fed
    archive was the only free source with depth. This mode points the same kind
    of tree at the stream a retail scalper actually reads: data prints, every
    central bank's speakers, intervention talk, tariffs, geopolitics and order
    flow, twenty thousand posts a year.

    Two things are checked before anything is fetched. ``robots.txt``, which is
    the wire's own statement of who may read it in bulk; and the caller's
    intent, because ``--collect-only`` and ``--warm-candles`` are the two halves
    of a run that takes tens of minutes each and are meant to be run apart.
    """
    import os
    import statistics
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from .fx import candles as C
    from .fx import posts as PS
    from .fx import reader as R
    from .fx import study as S
    from .fx import wire as W
    from .fx import wirestudy as WS
    from .fx.reader import WireReader
    from .listing.store import Store

    store = Store(args.cache)
    horizons = tuple(int(h) for h in args.horizons.split(",") if h.strip())
    report = tuple(h for h in WS.REPORT_HORIZONS if h in horizons) or horizons[:3]
    now = time.time()
    since = W.day_epoch(args.since) if args.since else now - args.days * 86400
    until = W.day_epoch(args.until) + 86400 - 1 if args.until else now
    if until <= since:
        print("--until is not after --since", file=sys.stderr)
        return 1

    # The candle feed is a different host with a different robots.txt (it has
    # none: 404, nothing disallowed), so warming it is never gated on the wire's.
    if args.warm_candles:
        print(f"warming {len(C.SYMBOLS)} pairs x 2 sides x "
              f"{len(C.days_covering(since, until))} days of 1-minute candles")
        warmth = C.warm(store, C.SYMBOLS, since, until, workers=max(1, min(args.workers_io, 2)))
        print(warmth.summary())
        for symbol, (have, want) in C.coverage(store, C.SYMBOLS, since, until).items():
            print(f"  {symbol}: {have}/{want} files on disk")
        return 0

    robots: dict[str, Any] = {"consulted": False}
    if args.posts:
        # A file the user brought. Nothing is fetched, so ``robots.txt`` is not
        # consulted at all: it governs crawling a site, and there is no site
        # here. Reading somebody's own corpus is not a request to anybody.
        print(f"corpus: {len(args.posts)} local file(s); no wire request is made, "
              f"so robots.txt is not consulted")
        articles, post_coverage = PS.import_posts(store, args.posts, since, until,
                                                  body_chars=args.body_chars)
        print(post_coverage.summary())
        where = PS.per_weekday(articles)
        print(f"  in the window: {len(articles)} posts, {where['weekday']} on a "
              f"weekday, {where['weekend']} at a weekend (UTC)")
        if args.collect_only:
            return 0
        if not articles:
            print("no posts in range", file=sys.stderr)
            return 1
    else:
        # ---- who may read this wire, asked before anything is fetched
        try:
            verdict = W.check(store, "/news/")
            index_verdict = W.check(store, "/articles-sitemap-index.xml")
        except Exception as exc:  # noqa: BLE001 -- unreachable robots is not permission
            print(f"{W.ROBOTS_URL}: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("robots.txt could not be read, which is not permission. Nothing fetched.")
            return 3
        allowed, blocked = verdict.allowed, verdict.blocked_by
        robots = {"consulted": True, "articles_allowed": allowed,
                  "blocked_by": blocked, "rule": verdict.rule}
        print(f"robots.txt: articles {'allowed' if allowed else 'DISALLOWED'}"
              f"{'' if allowed else ' for ' + blocked + ' (' + verdict.rule + ')'}"
              f"; sitemaps {'allowed' if index_verdict.allowed else 'DISALLOWED'}")
        offline = not (allowed and index_verdict.allowed)
        if offline:
            print("\nThe wire names AI agents in robots.txt and disallows them. This is an AI")
            print("agent doing a bulk fetch, so the rule binds whatever User-Agent header it")
            print("would send, and nothing will be fetched from it. --warm-candles still")
            print("works: the price feed is a different host with no such rule.")
            if not store.get("wire:index")[0]:
                print("No cached corpus either, so there is nothing to read. Stopping.")
                return 3
            print("A cached corpus is present, so the run continues against it and sends no")
            print("requests to the wire at all.")

        articles, coverage = W.collect(store, since, until,
                                       workers=min(args.workers_io, 4),
                                       limit=args.limit, offline=offline)
        print(coverage.summary())
        if args.collect_only:
            weeks = W.per_week(articles)
            if weeks:
                counts = sorted(weeks.values())
                print(f"  {len(weeks)} ISO weeks, {counts[0]} to {counts[-1]} posts a "
                      f"week, median {counts[len(counts) // 2]}")
            return 0
    if not articles:
        print("no articles in range", file=sys.stderr)
        return 1
    print(f"{len(articles)} posts, {articles[0].when:%Y-%m-%d %H:%M} to "
          f"{articles[-1].when:%Y-%m-%d %H:%M} UTC")

    # ---- the reader
    def make_reader() -> WireReader:
        if args.provider == "mock" or (
            args.provider == "auto" and not os.environ.get("TYPESAFE_API_KEY")
        ):
            from .fx.mock import MockWireClient

            return WireReader(MockWireClient(), body_chars=args.body_chars)
        return WireReader(resolve_client(args.provider, timeout_s=20.0),
                          body_chars=args.body_chars)

    provider = make_reader().client.provider
    started = time.perf_counter()

    def progress(done: int, total: int, round_two: int, cost: float) -> None:
        print(f"  read {done}/{total}, {round_two} to round two, ${cost:.3f}, "
              f"{time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)

    readings = WS.read_all(
        articles, make_reader, store=None if args.no_reading_cache else store,
        cache_tag=f"{provider}:{R.WIRE_VERSION}", workers=args.workers,
        body_chars=args.body_chars, progress=progress,
    )
    walls = sorted(r.wall_ms for r in readings)
    print(
        f"reader ({provider}, {R.WIRE_VERSION}): {len(readings)} posts, "
        f"{sum(r.rounds == 2 for r in readings)} went to round two, "
        f"median {statistics.median(walls):.0f} ms, "
        f"${WS.cost_usd(readings):.3f} of input tokens, "
        f"{time.perf_counter() - started:.0f}s wall"
    )

    # ---- the arms
    tapes = C.MinuteTapes(store)
    graded = dict(horizons=horizons, workers=args.workers_io, latency_s=args.latency)
    arms: list[tuple[str, list[S.Signal], str]] = [
        (f"reader >={args.threshold:.2f}", WS.reader_signals(readings, args.threshold),
         "the w1 tree"),
        ("keyword-bot", WS.keyword_signals(articles), "the wire's own lexicon, counted"),
        ("wire-sample", WS.sample_signals(articles, args.sample),
         f"a seeded sample of {args.sample}, keyword sign"),
    ]
    summaries: list[Any] = []
    per_arm: list[list[Any]] = []
    for name, signals, note in arms:
        outcomes = WS.measure(tapes, signals, **graded)
        nulls = WS.measure(tapes, WS.null_signals(tapes, outcomes, per=args.null_per,
                                                  horizons=horizons, latency_s=args.latency,
                                                  workers=args.workers_io), **graded)
        summaries.append(S.summarize(name, signals, outcomes, nulls, horizons=horizons,
                                    note=note, latency_s=args.latency))
        per_arm.append(outcomes)

    print(
        f"\nSigned log return per signal, bps, entering at the OPEN of the first 1-minute"
        f"\nbar starting at or after the post + {args.latency:g}s -- so the latency rounds up"
        "\nto the next minute boundary. A long pays the ASK open, a short is filled at the"
        "\nBID open, and the exit is the mid close. 'pre' is the 15 minutes BEFORE the post,"
        "\nsigned by the side taken: that is the wire's lateness, measured. 'sprd' is the"
        "\nspread at entry in bps. z is against the same pair and side at random moments"
        "\nwithin 5 days where the tape has bars, at the same latency."
    )
    hz = "".join(f"{'+' + str(h) + 'm':>9}" for h in horizons)
    print(f"\n{'arm':<22}{'signals':>8}{'traded':>7}{'pre':>7}{'sprd':>6}{hz}"
          f"{'hit15':>7}{'z15':>7}")
    print("-" * (22 + 8 + 7 + 13 + 9 * len(horizons) + 14))
    for s_row in summaries:
        cells = "".join(f"{s_row.fwd[h].mean:>+9.0f}" for h in horizons)
        tail = (f"{s_row.hit.get(15, 0.0):>7.0%}{s_row.z.get(15, 0.0):>+7.1f}"
                if s_row.measured else f"{'-':>7}{'-':>7}")
        print(f"{s_row.name:<22}{s_row.signals:>8}{s_row.measured:>7}"
              f"{s_row.pre.mean:>+7.0f}{s_row.spread.mean:>6.1f}{cells}{tail}")
        blank = " " * 13
        errs = "".join(f"{'+-' + format(s_row.fwd[h].se, '.0f'):>9}" for h in horizons)
        null = "".join(f"{s_row.null[h].mean:>+9.0f}" for h in horizons)
        mid = "".join(f"{s_row.fwd_mid[h].mean:>+9.0f}" for h in horizons)
        print(f"{'  s.e.':<22}{'':>8}{'':>7}{blank}{errs}")
        print(f"{'  null':<22}{'':>8}{'':>7}{blank}{null}")
        print(f"{'  no spread':<22}{'':>8}{'':>7}{blank}{mid}")
        if s_row.note:
            print(f"  ({s_row.note})")

    reader_outcomes = per_arm[0]
    shut = [WS.shut_out(signals, outcomes)
            for (_n, signals, _t), outcomes in zip(arms, per_arm)]
    print("\nwhat the tape could and could not price (spot FX shuts Fri ~21:00 - "
          "Sun ~21:00 UTC):")
    for (name, _signals, _note), row in zip(arms, shut):
        print(f"  {name:<22}{row.summary()}")

    print("\nwhat the reader made of the wire:")
    for field_name in ("category", "currency", "direction"):
        print(f"  {field_name}: " + ", ".join(
            f"{k} {v}" for k, v in WS.answer_counts(readings, field_name).items()))

    print(f"\nwhere the reader stayed out (>= {args.threshold:.2f} to trade):")
    print(f"{'category':<27}{'read':>7}{'traded':>8}{'abstained':>11}")
    for row in WS.abstentions(readings, args.threshold):
        print(f"{row.category[:26]:<27}{row.read:>7}{row.traded:>8}{row.rate:>11.0%}")

    tables: list[tuple[str, list[Any]]] = []
    for title, key in (
        ("reader, by category", WS.by_reading(reader_outcomes, readings, "category")),
        ("reader, by currency", WS.by_reading(reader_outcomes, readings, "currency")),
        ("reader, scheduled", WS.by_reading(reader_outcomes, readings, "scheduled")),
        ("reader, is_number", WS.by_reading(reader_outcomes, readings, "is_number")),
        ("reader, by session (UTC)", WS.by_session),
        ("reader, weekday or weekend", WS.by_weekday),
        ("reader, by pair", WS.by_pair),
    ):
        rows = WS.breakdown(reader_outcomes, key, horizons=report)
        tables.append((title, rows))
    for title, key in (
        ("keyword-bot, by session (UTC)", WS.by_session),
        ("keyword-bot, by pair", WS.by_pair),
    ):
        tables.append((title, WS.breakdown(per_arm[1], key, horizons=report)))
    tables.append(("wire-sample, by session (UTC)",
                   WS.breakdown(per_arm[2], WS.by_session, horizons=report)))
    for title, rows in tables:
        print(f"\n{title}:")
        print(WS.header(report))
        for row in rows[: max(args.show, 10)]:
            print(row.row(report))

    sweep: list[Any] = []
    swept = arms[0][1] if len(arms[0][1]) >= 5 else arms[1][1]
    sweep_h = tuple(h for h in (15, 60) if h in horizons) or (horizons[-1],)
    sweep = WS.latency_sweep(tapes, swept, horizons=sweep_h, per=args.null_per,
                             workers=args.workers_io)
    print(f"\nlatency sweep ({len(swept)} signals). On minute bars 0s and 60s are the same")
    print("fill except for a post stamped exactly on a boundary; only the 300s row can differ:")
    cols = "".join(f"{'+' + str(h) + 'm':>8}{'s.e.':>7}{'null':>8}{'z':>6}" for h in sweep_h)
    print(f"{'entry':>8}{'traded':>8}{'sprd':>6}{cols}")
    for row in sweep:
        cells = "".join(
            f"{row.fwd[h].mean:>+8.0f}{row.fwd[h].se:>7.0f}"
            f"{row.null[h].mean:>+8.0f}{row.z[h]:>+6.1f}" for h in sweep_h)
        print(f"{format(row.latency_s, 'g') + 's':>8}{row.measured:>8}"
              f"{row.spread.mean:>6.1f}{cells}")

    # Left becomes ``.bot`` and right becomes ``.reader`` on a ``Disagreement``,
    # so the keyword bot goes first here exactly as it does in --context.
    diffs = S.signal_disagreements(arms[1][1], arms[0][1],
                                   reader_outcomes + per_arm[1],
                                   horizon=report[min(1, len(report) - 1)])
    diffs.sort(key=lambda d: -max([abs(v) for v in d.outcomes.values()] or [0.0]))
    print(f"\nwhere counting words and reading them traded differently "
          f"({len(diffs)} of {len(readings)}):")
    for d in diffs[: args.show]:
        when = datetime.fromtimestamp(d.when, timezone.utc)
        fmt = lambda rows: ", ".join(  # noqa: E731
            f"{'long' if s > 0 else 'short'} {p}" for p, s in rows) or "nothing"
        moves = ", ".join(f"{p} {v:+.0f}" for p, v in d.outcomes.items()) or "no tape"
        print(f"  {when:%Y-%m-%d %H:%M} {d.title[:66]}")
        print(f"      bot: {fmt(d.bot)[:70]}")
        print(f"   reader: {fmt(d.reader)[:70]}")
        print(f"     tape: {moves[:70]}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)

        def outcome_row(o: Any) -> dict[str, Any]:
            return {
                "ts": o.signal.ts, "code": o.signal.code, "title": o.signal.title,
                "pair": o.signal.pair, "side": o.signal.sign,
                "strength": o.signal.strength, "pre_bps": o.pre_bps,
                "spread_bps": o.spread_bps, "latency_s": o.latency_s,
                "entry_ts": o.entry_ts, "session": W.sessions(o.signal.ts),
                "fwd_bps": {h: o.fwd_bps.get(h) for h in horizons},
                "fwd_mid_bps": {h: o.fwd_mid_bps.get(h) for h in horizons},
            }

        def cell_row(c: Any) -> dict[str, Any]:
            return {"key": c.key, "n": c.n, "pre": asdict(c.pre), "hit": c.hit,
                    "fwd": {h: asdict(v) for h, v in c.fwd.items()}}

        out.write_text(json.dumps({
            "mode": "wire", "tree": R.WIRE_VERSION, "provider": provider,
            "since": since, "until": until, "tape": "dukascopy-1m",
            "latency_s": args.latency, "threshold": args.threshold,
            "horizons": list(horizons), "body_chars": args.body_chars,
            "sample": args.sample,
            "robots": robots,
            "corpus": ("posts-file" if args.posts else "wire"),
            "posts_files": [str(f) for f in (args.posts or [])],
            "coverage": asdict(post_coverage if args.posts else coverage),
            "shut": [{"arm": name, **asdict(row)}
                     for (name, _s, _t), row in zip(arms, shut)],
            "arms": [
                {"name": s.name, "signals": s.signals, "measured": s.measured,
                 "note": s.note, "pre": asdict(s.pre), "spread": asdict(s.spread),
                 "fwd": {h: asdict(v) for h, v in s.fwd.items()},
                 "fwd_mid": {h: asdict(v) for h, v in s.fwd_mid.items()},
                 "null": {h: asdict(v) for h, v in s.null.items()},
                 "hit": s.hit, "z": s.z,
                 "outcomes": [outcome_row(o) for o in outcomes]}
                for s, outcomes in zip(summaries, per_arm)
            ],
            "breakdowns": [
                {"table": title, "horizons": list(report),
                 "rows": [cell_row(c) for c in rows]}
                for title, rows in tables
            ],
            "abstention": [
                {"category": a.category, "read": a.read, "traded": a.traded,
                 "rate": a.rate}
                for a in WS.abstentions(readings, args.threshold)
            ],
            "latency_sweep": [
                {"latency_s": r.latency_s, "measured": r.measured,
                 "spread": asdict(r.spread),
                 "fwd": {h: asdict(v) for h, v in r.fwd.items()},
                 "null": {h: asdict(v) for h, v in r.null.items()}, "z": r.z}
                for r in sweep
            ],
            "readings": [
                {"id": r.id, "ts": r.ts, "url": r.article.url,
                 "headline": r.article.headline, "section": r.article.section,
                 **WS.reading_to_dict(r)}
                for r in readings
            ],
        }, ensure_ascii=False, indent=1))
        print(f"\nwrote {out}")
    return 0


def cmd_fx(args: argparse.Namespace) -> int:
    """Read central-bank text in one second; let the spot tape grade every arm."""
    if args.wire:
        return cmd_fx_wire(args)
    if args.dots:
        return cmd_fx_dots(args)
    if args.presser:
        return cmd_fx_presser(args)
    if args.context:
        return cmd_fx_context(args)
    import os
    import statistics
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from .fx import documents as D
    from .fx import study as S
    from .fx.reader import TREE_VERSION, Reader
    from .listing.store import Store

    def day_epoch(text: str) -> float:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()

    store = Store(args.cache)
    issuers = tuple(i.strip().lower() for i in args.issuers.split(",") if i.strip())
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip()) if args.kinds else None
    horizons = tuple(int(h) for h in args.horizons.split(","))
    tape_name = args.tape or "yahoo"
    ticks = tape_name == "dukascopy"
    now = time.time()
    # --since/--until are whole UTC days and win over --days; --until includes
    # the day it names, so "--until 2026-09-18" keeps that evening's speeches.
    since = day_epoch(args.since) if args.since else now - args.days * 86400
    until = day_epoch(args.until) + 86400 if args.until else now
    if until <= since:
        print("--until is not after --since", file=sys.stderr)
        return 1
    if not ticks:
        short = [h for h in horizons if h < args.bar]
        if short:
            print(f"warning: horizon(s) {short} are shorter than one {args.bar}-minute bar "
                  f"and will round up to one; --tape dukascopy measures them properly",
                  file=sys.stderr)

    if args.snapshot_calendar:
        key, rows = D.calendar_snapshot(store)
        print(f"stored {len(rows)} calendar rows under {key}")

    found = D.collect(store, since=since, until=until, issuers=issuers, kinds=kinds,
                      limit=args.limit, workers=args.workers_io)
    documents = found.documents
    if not documents:
        print("no documents in range", file=sys.stderr)
        return 1
    first, last = documents[0].when, documents[-1].when
    print(
        f"{len(documents)} documents, {first:%Y-%m-%d} to {last:%Y-%m-%d}, "
        f"issuers: {', '.join(f'{k} {v}' for k, v in sorted(found.per_issuer.items()))}"
        f"; {found.skipped_no_time} rows dropped for having no time of day"
        f"; {found.short_bodies} bodies under {D.MIN_BODY} chars"
    )
    print("  kinds: " + ", ".join(
        f"{k} {v}" for k, v in sorted(found.per_kind.items(), key=lambda kv: -kv[1])))
    calendar = D.calendar_rows(store)
    weeks = sorted({D.week_key(r['ts'])[-8:] for r in calendar})
    print(
        f"calendar snapshots: {len(calendar)} rows over {len(weeks)} week(s)"
        + (f" ({', '.join(weeks)})" if weeks else " -- the surprise arm is not available")
    )

    def make_reader() -> Reader:
        if args.provider == "mock" or (
            args.provider == "auto" and not os.environ.get("TYPESAFE_API_KEY")
        ):
            from .fx.mock import MockFxClient

            return Reader(MockFxClient())
        return Reader(resolve_client(args.provider, timeout_s=20.0))

    provider = make_reader().client.provider
    started = time.perf_counter()

    def progress(done: int, total: int, round_two: int, cost: float) -> None:
        print(f"  read {done}/{total}, {round_two} to round two, ${cost:.3f}, "
              f"{time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)

    readings = S.read_all(
        documents, make_reader, store=None if args.no_reading_cache else store,
        cache_tag=f"{provider}:{TREE_VERSION}", calendar=calendar, workers=args.workers,
        progress=progress,
    )
    walls = sorted(r.wall_ms for r in readings)
    widths = sorted(r.questions_asked for r in readings)
    print(
        f"reader ({provider}): {len(readings)} documents, "
        f"{sum(r.rounds == 2 for r in readings)} went to round two, "
        f"{widths[len(widths) // 2]} questions in the median document, "
        f"median {statistics.median(walls):.0f} ms "
        f"(p90 {walls[int(0.9 * (len(walls) - 1))]:.0f} ms), "
        f"${S.cost_usd(readings):.3f} of input tokens, "
        f"{(time.perf_counter() - started):.0f}s wall for this run"
    )

    if ticks:
        tape = S.TickTapes(store, workers=args.workers_io)
    else:
        # Yahoo serves 60 days of 5-minute bars and 7 of 1-minute ones; asking
        # for more silently returns less, so the ceiling is applied here.
        span = max(1.0, (until - since) / 86400.0)
        tape = S.Tape(store, bar_min=args.bar, days=min(int(max(span, 7)),
                                                        60 if args.bar >= 5 else 7))
    graded = dict(horizons=horizons, workers=args.workers_io, latency_s=args.latency)
    arms: list[tuple[str, list[S.Signal], str]] = [
        ("keyword-bot", S.bot_signals(documents, S.keyword_bot), ""),
        ("surprise-bot", S.surprise_signals(documents, calendar),
         "" if calendar else "no calendar snapshot: not available"),
        (f"reader >={args.threshold:.2f}", S.reader_signals(readings, args.threshold), ""),
        ("all text", S.all_text_signals(documents), ""),
    ]
    summaries: list[S.ArmSummary] = []
    per_arm: list[list[S.Outcome]] = []
    outcome_pool: list[S.Outcome] = []
    for name, signals, note in arms:
        outcomes = S.measure(tape, signals, **graded)
        nulls = S.measure(tape, S.null_signals(tape, outcomes, per=args.null_per,
                                               horizons=horizons, latency_s=args.latency,
                                               workers=args.workers_io),
                          **graded)
        summaries.append(S.summarize(name, signals, outcomes, nulls, horizons=horizons,
                                     note=note, latency_s=args.latency if ticks else 0.0))
        per_arm.append(outcomes)
        outcome_pool.extend(outcomes)

    hz = "".join(f"{'+' + str(h) + 'm':>9}" for h in horizons)
    lead = f"{'pre':>7}{'rush':>7}{'sprd':>6}" if ticks else f"{'pre':>7}{'bar':>7}"
    if ticks:
        print(
            f"\nSigned log return per signal, bps, entering on the first tick at or after the"
            f"\nrelease + {args.latency:g}s, paying the ask to go long and the bid to go short,"
            "\nexiting at the mid. 'pre' is the 15 min before the release; 'rush' is how far"
            "\nthe mid moved between the release and the entry, signed by the side taken;"
            "\n'sprd' is the spread at entry in bps. z is against the same pair and side at"
            "\nrandom moments within 5 days where the tape has ticks, at the same latency."
        )
    else:
        print(
            f"\nSigned log return per signal, bps, entering at the open of the first {args.bar}-minute"
            "\nbar AFTER the release. 'pre' is the 15 min before it; 'bar' is the release bar"
            "\nitself. z is against the same pair and side at random moments within 5 days,"
            "\ndrawn only where the tape has bars (spot FX is shut all weekend)."
        )
    print(f"\n{'arm':<16}{'signals':>8}{'traded':>7}{lead}{hz}{'hit15':>7}{'z15':>7}")
    print("-" * (16 + 8 + 7 + len(lead) + 9 * len(horizons) + 14))
    for s in summaries:
        cells = "".join(f"{s.fwd[h].mean:>+9.0f}" for h in horizons)
        values = (f"{s.pre.mean:>+7.0f}{s.rush.mean:>+7.0f}{s.spread.mean:>6.1f}" if ticks
                  else f"{s.pre.mean:>+7.0f}{s.release_bar.mean:>+7.0f}")
        blank = " " * len(lead)
        print(
            f"{s.name:<16}{s.signals:>8}{s.measured:>7}{values}{cells}"
            f"{s.hit.get(15, 0.0):>7.0%}{s.z.get(15, 0.0):>+7.1f}"
        )
        errs = "".join(f"{'+-' + format(s.fwd[h].se, '.0f'):>9}" for h in horizons)
        null = "".join(f"{s.null[h].mean:>+9.0f}" for h in horizons)
        print(f"{'  s.e.':<16}{'':>8}{'':>7}{blank}{errs}")
        print(f"{'  null':<16}{'':>8}{'':>7}{blank}{null}")
        if ticks:
            mid = "".join(f"{s.fwd_mid[h].mean:>+9.0f}" for h in horizons)
            print(f"{'  no spread':<16}{'':>8}{'':>7}{blank}{mid}")
        if s.note:
            print(f"{'  (' + s.note + ')':<16}")

    sweep: list[S.ArmSummary] = []
    if args.latency_sweep and not ticks:
        print("\n--latency-sweep needs --tape dukascopy; 5-minute bars have one latency",
              file=sys.stderr)
    elif args.latency_sweep:
        reader_arm = arms[2]
        swept, swept_name = (reader_arm[1], reader_arm[0]) if len(reader_arm[1]) >= 5 \
            else (arms[3][1], arms[3][0])
        sweep_h = tuple(h for h in (15, 60) if h in horizons) or (horizons[-1],)
        sweep = S.latency_sweep(tape, swept, horizons=sweep_h, per=args.null_per,
                                workers=args.workers_io)
        print(f"\nlatency sweep ({swept_name}, {len(swept)} signals, net of the half spread):")
        cols = "".join(f"{'+' + str(h) + 'm':>8}{'s.e.':>7}{'null':>8}{'z':>6}" for h in sweep_h)
        print(f"{'entry':>8}{'traded':>8}{'rush':>7}{'sprd':>6}{cols}")
        for row in sweep:
            cells = "".join(
                f"{row.fwd[h].mean:>+8.0f}{row.fwd[h].se:>7.0f}"
                f"{row.null[h].mean:>+8.0f}{row.z[h]:>+6.1f}" for h in sweep_h)
            print(f"{row.name:>8}{row.measured:>8}{row.rush.mean:>+7.0f}"
                  f"{row.spread.mean:>6.1f}{cells}")

    sweep_h = horizons[min(1, len(horizons) - 1)]
    print(f"\nreader threshold sweep (+{sweep_h}m, same cached answers):")
    print(f"{'thr':>6}{'signals':>9}{'traded':>8}{'+' + str(sweep_h) + 'm':>8}"
          f"{'s.e.':>7}{'null':>8}{'z':>7}")
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.40):
        signals = S.reader_signals(readings, thr)
        one = dict(horizons=(sweep_h,), workers=args.workers_io, latency_s=args.latency)
        outcomes = S.measure(tape, signals, **one)
        nulls = S.measure(tape, S.null_signals(tape, outcomes, per=args.null_per,
                                               horizons=(sweep_h,), latency_s=args.latency,
                                               workers=args.workers_io),
                          **one)
        s = S.summarize(f"{thr}", signals, outcomes, nulls, horizons=(sweep_h,))
        print(
            f"{thr:>6.2f}{s.signals:>9}{s.measured:>8}{s.fwd[sweep_h].mean:>+8.0f}"
            f"{s.fwd[sweep_h].se:>7.0f}{s.null[sweep_h].mean:>+8.0f}{s.z[sweep_h]:>+7.1f}"
        )

    kinds_seen: dict[str, int] = {}
    stances: dict[str, int] = {}
    for r in readings:
        kinds_seen[r.kind] = kinds_seen.get(r.kind, 0) + 1
        stances[r.stance] = stances.get(r.stance, 0) + 1
    print("\nwhat the reader thinks the feed is made of: " + ", ".join(
        f"{k} {v}" for k, v in sorted(kinds_seen.items(), key=lambda kv: -kv[1])))
    print("how it read them: " + ", ".join(
        f"{k} {v}" for k, v in sorted(stances.items(), key=lambda kv: -kv[1])))
    loud = sorted(readings, key=lambda r: -r.intervention)[:3]
    if loud and loud[0].intervention > 0:
        print("furthest up the intervention ladder: " + "; ".join(
            f"{r.intervention:.2f} {r.document.title[:50]}" for r in loud))

    diffs = S.disagreements(readings, args.threshold, outcome_pool)
    diffs.sort(key=lambda d: -max([abs(v) for v in d.outcomes.values()] or [0.0]))
    print(f"\nwhere counting words and reading them traded differently "
          f"({len(diffs)} of {len(readings)}); +15m bps per pair:")
    for d in diffs[: args.show]:
        when = datetime.fromtimestamp(d.when, timezone.utc)
        fmt = lambda rows: ", ".join(  # noqa: E731
            f"{'long' if s > 0 else 'short'} {p}" for p, s in rows) or "nothing"
        moves = ", ".join(f"{p} {v:+.0f}" for p, v in d.outcomes.items()) or "no tape"
        print(f"  {when:%Y-%m-%d %H:%M} [{d.issuer}] {d.title[:62]}")
        print(f"      bot: {fmt(d.bot)[:70]}")
        print(f"   reader: {fmt(d.reader)[:70]}")
        print(f"     tape: {moves[:70]}")

    if args.out:
        def outcome_row(o: S.Outcome) -> dict[str, Any]:
            return {
                "ts": o.signal.ts, "title": o.signal.title, "pair": o.signal.pair,
                "side": o.signal.sign, "strength": o.signal.strength,
                "pre_bps": o.pre_bps, "rush_bps": o.rush_bps,
                "spread_bps": o.spread_bps, "latency_s": o.latency_s,
                "entry_ts": o.entry_ts,
                "fwd_bps": {h: o.fwd_bps.get(h) for h in horizons},
                "fwd_mid_bps": {h: o.fwd_mid_bps.get(h) for h in horizons},
            }

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "provider": provider, "tree": TREE_VERSION, "days": args.days,
            "since": since, "until": until,
            "tape": tape_name, "latency_s": args.latency if ticks else None,
            "issuers": issuers, "kinds": kinds, "threshold": args.threshold,
            "horizons": horizons, "bar_min": None if ticks else args.bar,
            "documents": len(documents), "skipped_no_time": found.skipped_no_time,
            "short_bodies": found.short_bodies, "per_kind": found.per_kind,
            "calendar_weeks": weeks,
            "arms": [
                {"name": s.name, "signals": s.signals, "measured": s.measured,
                 "note": s.note, "pre": asdict(s.pre), "release_bar": asdict(s.release_bar),
                 "spread": asdict(s.spread), "rush": asdict(s.rush),
                 "fwd": {h: asdict(v) for h, v in s.fwd.items()},
                 "fwd_mid": {h: asdict(v) for h, v in s.fwd_mid.items()},
                 "null": {h: asdict(v) for h, v in s.null.items()},
                 "hit": s.hit, "z": s.z,
                 # Every graded signal, so nobody has to recompute the run to ask
                 # it a question it was not asked here.
                 "outcomes": [outcome_row(o) for o in outcomes]}
                for s, outcomes in zip(summaries, per_arm)
            ],
            "latency_sweep": [
                {"latency_s": r.latency_s, "signals": r.signals, "measured": r.measured,
                 "spread": asdict(r.spread), "rush": asdict(r.rush),
                 "fwd": {h: asdict(v) for h, v in r.fwd.items()},
                 "null": {h: asdict(v) for h, v in r.null.items()}, "z": r.z}
                for r in sweep
            ],
            "readings": [
                {"id": r.document.id, "ts": r.document.ts, "issuer": r.document.issuer,
                 "title": r.document.title, **S.reading_to_dict(r)}
                for r in readings
            ],
        }, ensure_ascii=False, indent=1))
        print(f"\nwrote {out}")
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

    listing = sub.add_parser("listing", help="read exchange announcements in one second; the tape grades it")
    listing.add_argument("--provider", default="auto", choices=("auto", "jev", "mock"))
    listing.add_argument("--days", type=float, default=180.0)
    listing.add_argument("--catalogs", default="48,161,49",
                         help="Binance CMS catalogues: 48 listings, 161 delistings, 49 general news")
    listing.add_argument("--threshold", type=float, default=0.25, help="reader signal strength to trade")
    listing.add_argument("--horizons", default="1,5,15,60", help="minutes after entry")
    listing.add_argument("--limit", type=int, default=0, help="only the most recent N announcements")
    listing.add_argument("--cache", default=".cache/listing")
    listing.add_argument("--no-reading-cache", action="store_true", help="ask the model again even if cached")
    listing.add_argument("--workers", type=int, default=3, help="parallel model readers")
    listing.add_argument("--workers-io", type=int, default=8, help="parallel candle fetches")
    listing.add_argument("--null-per", type=int, default=2, help="random controls per measured signal")
    listing.add_argument("--show", type=int, default=8, help="disagreements to print")
    listing.add_argument("--out", default="", help="write a JSON record of the run")
    listing.set_defaults(func=cmd_listing)

    fx = sub.add_parser("fx", help="read central-bank text in one second; the spot tape grades it")
    fx.add_argument("--provider", default="auto", choices=("auto", "jev", "mock"))
    fx.add_argument("--days", type=float, default=60.0, help="how far back; 60 = Yahoo 5m depth")
    fx.add_argument("--since", default="", help="YYYY-MM-DD; overrides --days")
    fx.add_argument("--until", default="", help="YYYY-MM-DD, inclusive; overrides --days")
    fx.add_argument("--tape", default="", choices=("", "yahoo", "dukascopy"),
                    help="yahoo = 5m bars, 60 days deep; dukascopy = ticks, back to 2003; "
                         "the default is bars, or ticks for --dots and --presser")
    fx.add_argument("--latency", type=float, default=1.0,
                    help="seconds between the timestamp and the entry tick (dukascopy only)")
    fx.add_argument("--latency-sweep", action="store_true",
                    help="same signals entered at 0, 1, 5, 30 and 120 seconds")
    fx.add_argument("--issuers", default="fed,ecb,boj,boe",
                    help="fed has a deep archive; the others are RSS-shallow")
    fx.add_argument("--kinds", default="monetary_policy,speech,testimony",
                    help="document kinds to keep; empty string keeps everything")
    fx.add_argument("--threshold", type=float, default=0.15, help="reader strength to trade")
    fx.add_argument("--horizons", default="5,15,30,60",
                    help="minutes after entry; 1 is only meaningful on ticks")
    fx.add_argument("--bar", type=int, default=5, help="bar size in minutes (5 or 1); yahoo only")
    fx.add_argument("--limit", type=int, default=0, help="only the most recent N documents")
    fx.add_argument("--cache", default=".cache/fx")
    fx.add_argument("--no-reading-cache", action="store_true", help="ask the model again even if cached")
    fx.add_argument("--workers", type=int, default=3, help="parallel model readers")
    fx.add_argument("--workers-io", type=int, default=8, help="parallel fetches")
    fx.add_argument("--null-per", type=int, default=2, help="random controls per measured signal")
    fx.add_argument("--show", type=int, default=8, help="disagreements to print")
    fx.add_argument("--dots", action="store_true",
                    help="validate the dot-plot rule out of sample and forward; no model")
    fx.add_argument("--presser", action="store_true",
                    help="read the press conference that starts 30 minutes after the statement")
    fx.add_argument("--pairs", default="EURUSD,USDJPY,GBPUSD",
                    help="--dots only: the pairs to grade the rule on")
    fx.add_argument("--context", action="store_true",
                    help="FOMC statements only, read against the pre-release context "
                         "(implies --issuers fed)")
    fx.add_argument("--context-chars", type=int, default=12000,
                    help="character budget for the assembled context block")
    fx.add_argument("--snapshot-calendar", action="store_true",
                    help="store this week's calendar so the surprise arm can run later")
    fx.add_argument("--wire", action="store_true",
                    help="read investinglive.com's FX wire, graded on 1-minute candles")
    fx.add_argument("--posts", nargs="+", default=[], metavar="FILE",
                    help="--wire: read a local post archive (jsonl or csv) instead of "
                         "fetching; no request is made, so robots.txt is not consulted")
    fx.add_argument("--collect-only", action="store_true",
                    help="--wire: scrape and cache the articles, print the summary, stop")
    fx.add_argument("--warm-candles", action="store_true",
                    help="--wire: pre-fetch the seven pairs' BID/ASK day files, stop")
    fx.add_argument("--sample", type=int, default=3000,
                    help="--wire: posts in the seeded base-rate sample")
    fx.add_argument("--body-chars", type=int, default=3000,
                    help="--wire: article body characters the reader is shown")
    fx.add_argument("--out", default="", help="write a JSON record of the run")
    fx.set_defaults(func=cmd_fx)

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
