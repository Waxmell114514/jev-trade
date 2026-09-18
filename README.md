# jev-trade

A simulated high-frequency crypto trading loop that makes its buy/sell calls with
[**Jev**](https://typesafe.ai/blog/introducing-system-one-models-and-jev), TypeSafe AI's
System One model.

Recent prices and volumes for BTC/ETH go in; a typed, confidence-aware decision
comes out; a cost-aware execution simulator turns it into P&L you can actually
argue with.

---

## One correction to the brief

The task description said Jev's output latency is *extremely slow*. It is the
opposite, and the difference matters for this design.

TypeSafe reports **70–500 ms end-to-end**, against 3–329 s for frontier
models, because Jev does not generate tokens at all — a parallel sampler emits
every answer in one shot. That speed is precisely *why* it suits
["System One"](https://docs.typesafe.ai/concepts/system-one) work: fast,
intuitive, low-deliberation judgment, the kind a human scalper does at a glance.
A genuinely slow model would be unusable here, for reasons this repo measures
rather than asserts (see [The latency experiment](#the-latency-experiment)).

The rest of the brief stands, and the design follows it: a fast, natively
structured decision model is a good fit for reading a tape.

Everything below still holds if you disagree about the latency. The engine
treats latency as a tunable, not an assumption — `--extra-latency-ms` lets you
simulate a model of any speed and watch what it does to the strategy.

---

## What it does

```
 Kraken / synthetic feed          all arithmetic lives here
        │                    ┌──────────────────────────────┐
        ▼                    │  features.py   returns, vol, │
   ┌─────────┐               │                z-scores,     │
   │  ticks  │──────────────▶│                imbalance     │
   └─────────┘               └──────────────┬───────────────┘
                                            ▼
                             ┌──────────────────────────────┐
                             │ discretize.py  numbers→words │
                             └──────────────┬───────────────┘
                                            ▼  state (words only)
                             ┌──────────────────────────────┐
                             │  POST /v1/systemone          │
                             │  6 typed questions, 1 call   │◀── jev-latest
                             └──────────────┬───────────────┘
                                            ▼  typed answers + confidence
                             ┌──────────────────────────────┐
                             │  policy.py    answers→units  │
                             └──────────────┬───────────────┘
                                            ▼
                             ┌──────────────────────────────┐
                             │  engine.py    latency, gates │
                             │  execution.py fills, fees    │
                             └──────────────────────────────┘
```

The governing rule comes from TypeSafe's own
[jaggedness notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13):

> *"Jev is not a calculator. We strongly recommend implementing any mathematical
> logic in code."*

So **code owns every number and Jev owns every judgment.** The model is never
asked what 76,412.30 minus 76,398.10 is, or whether that gap is large. It is
asked whether the tape leans one way, and it answers with a probability
distribution over options we defined.

## Quickstart

No dependencies, no build step, Python 3.10+.

```bash
python -m jevtrade.cli backtest --baselines     # run and score the loop
python -m jevtrade.cli decide                   # one decision, fully unpacked
python -m jevtrade.cli sweep                    # what latency costs you
python -m jevtrade.cli fetch --symbol ETH       # real bars from Kraken
```

Without `TYPESAFE_API_KEY` the loop runs against an offline stub and says so,
loudly, on every report. With a key it calls the real thing:

```bash
export TYPESAFE_API_KEY=sk-...
python -m jevtrade.cli models
python -m jevtrade.cli backtest --provider jev
```

## The Jev integration

Three files carry it. They are worth reading in this order.

### 1. `discretize.py` — numbers become words

Jev is documented as weaker on numeric representations than semantic ones, so
no float ever reaches it. The raw features stay on the decision record for the
audit trail; the model sees this:

```json
{
  "instrument": "BTC-USD spot, 250 ms snapshots",
  "price_action": {
    "last_snapshot": "drifting up",
    "last_5_snapshots": "rising",
    "last_15_snapshots": "rising hard",
    "streak": "7 consecutive up ticks",
    "recent_balance": "8 of the last 10 snapshots closed higher",
    "character": "a clean one-way run",
    "versus_session": "far above the session's average traded price"
  },
  "activity": { "volatility": "normal", "traded_volume": "light" },
  "order_book": {
    "resting_size": "heavily weighted to buyers",
    "depth": "thinner than usual",
    "cost_to_cross_the_spread": "small next to a typical move"
  },
  "our_book": { "exposure": "slightly long", "open_position_pnl": "a small gain" }
}
```

Every bucketed value comes from a closed vocabulary, and a test asserts it.

One choice worth calling out: the spread is described *relative to the typical
tick move*, not in basis points. "Costs about as much as a typical move to
cross" is the fact that decides whether a scalp is viable. "3.4 bps" is not a
judgment, it is a number, and numbers are our job.

### 2. `questions.py` — six judgments, one request

Jev ingests the state once and evaluates every question against it in parallel,
so extra questions cost a few tokens and almost no latency
([speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)). One broad
"should I trade this?" is therefore split into six atomic ones:

| id | type | asks |
|---|---|---|
| `direction` | Choice | up / down / unclear |
| `follow_through` | Choice | continuation / reversal / no pattern |
| `setup_quality` | Score | how much the parts of the tape agree, on a 4-level rubric |
| `liquidity_ok` | Noul | can we open normal size right now? |
| `disorderly` | Noul | is this a dislocation to stay out of? |
| `cut_position` | Noul | should the open position be reduced? |

Choice is *relative* (which option wins) and Noul is *absolute* (whether a
condition holds). TypeSafe warns these are not comparable to each other, so the
policy never mixes their thresholds.

### 3. `policy.py` — answers become a quantity

The model never sees a position size. Code combines the six answers with
weights that live in a dataclass, so tuning the strategy means changing a
number in `PolicyConfig`, not rewriting a prompt:

```
edge        = P(up) - P(down)                    # from the Choice distribution
conviction  = setup_quality rescaled to 0..1
multiplier  = follow-through and confidence adjustments
target      = clamp(edge × conviction × multiplier × gain) × max_units
```

then five hard gates in order — `disorderly`, `cut_position`, `low_confidence`,
`weak_setup`, `no_edge` — plus an `illiquid` gate that blocks *increases* but
never blocks an exit. A thin book must not be able to trap us in a position.
Every gate has a test.

## Reading the output honestly

A backtest is the easiest thing in finance to lie with. Four things to know
before believing any number this repo prints.

**1. The default provider is not Jev.** Jev is a closed API behind an
early-access waitlist. Without a key you get `MockJevClient`, a hand-written
scoring function over the same vocabulary the model sees. It reads only the
state dict, so it has no oracle access to the simulator's hidden state — but it
is not a model, and every report it produces is stamped `provider=mock`.
Numbers from it describe this repo's plumbing and nothing else.

**2. The synthetic tape has a known, planted edge.** `SyntheticFeed` runs a
hidden AR(1) process that biases the next return *and* leaks into the
observable book imbalance. So book imbalance genuinely predicts the next move,
by an amount you set with `--alpha`. This is deliberate: it makes the harness
interpretable. With `--alpha 0` there is nothing to find, and a test asserts
that a book-reading rule earns nothing on such a tape. Finding an edge in a
market that was built to contain one is not evidence about a model.

**3. The default run loses money, and that is the correct result.** Here is a
default backtest:

```
P&L
  gross              28.90
  fees              -91.60
  net               -62.69   (-0.825% of capital at risk)

Risk
  turnover      915,968  over 250 fills
  break-even fee  0.316 bps   (the strategy is viable only below this taker fee)

Decisions
  hit rate      67.8% over 339 calls   (scored where the decision was formed)
```

A 67.8% hit rate and still a loss. The edge is worth about 1 bp over an
eight-tick hold, while a round trip costs the 0.25 bp spread plus 2 × 1 bp of
taker fees. **No amount of predictive accuracy survives costs that exceed the
edge.** That is the central fact of high-frequency trading, and a harness that
hid it would be worthless.

The `break-even fee` line names the number that actually decides viability:
at 0.316 bps this strategy needs a top-tier fee schedule or maker rebates, not
a retail account. Run `--fee-bps 0.2` and it turns a profit; `--fee-bps 2` and
it is hopeless. Set `--fee-bps 0` to see the raw signal with no costs at all.

**4. Compare against the baselines, not against zero.** `--baselines` runs
rule-based strategies over the identical ticks, identical costs and identical
one-tick execution delay:

```
strategy                 net       gross       fees   fills      maxDD     hit
------------------------------------------------------------------------------
jev (mock)            -62.69       28.90      91.60     250      71.40   67.8%
flat                    0.00        0.00       0.00       0      -0.00       -
buy_and_hold          101.75      102.51       0.76       1      36.27       -
momentum             -233.70      -73.69     160.01     300     249.84       -
mean_reversion       -321.45     -161.43     160.01     300     323.76       -
imbalance            -214.34       -6.73     207.61     264     237.93       -
random               -486.34     -230.20     256.14     241     492.39       -
```

The decision loop is the only strategy with positive gross P&L — its gating
keeps it out of trades the raw `imbalance` rule takes and loses on. And
`buy_and_hold` wins outright, because this particular seed drifted up. That is
a coin flip on the seed, not a benchmark anyone beat.

There is also a **calibration table**, because calibrated probabilities are
Jev's central claim and the policy's confidence thresholds are only as sound as
that calibration:

```
Calibration of P(up | directional)
  bucket      n    predicted   realised
  0.0-0.2    291     0.05       0.34
  0.8-1.0    290     0.95       0.65
```

That is the mock being badly overconfident, which is exactly what the table is
for. Run it against the real model and see.

## The latency experiment

`sweep` holds everything else fixed and varies only how long the decision takes
to come back, averaged over several paths:

```
 added latency  decisions  dropped  skipped   formed  executed      gross        net
------------------------------------------------------------------------------------
          0 ms        742        0        0   63.2%     61.8%      88.44    -107.19
        300 ms        742        0        0   63.2%     61.3%      72.05    -125.66
       1200 ms        742        0        0   63.0%     60.1%      38.38    -156.52
       3000 ms        373        0      369   63.0%     54.3%      12.29    -100.33
```

`formed` scores each call from the tick it was made on; `executed` scores the
*same call* from the tick it actually reached the market. They separate two
things that get confused:

- **The judgment never degrades.** `formed` is flat at ~63% at every latency.
  A slow model is not a worse analyst.
- **The market moves on.** `executed` decays to 54.3% — near a coin flip — and
  gross P&L falls 86%. The answer was right about a tape that no longer exists.

At 3000 ms the loop also *skips* 369 decision points, because only one request
is ever in flight. A slow provider does not merely act late; it gets fewer
looks at the market.

Set `--deadline-ms 400` and late decisions are dropped outright rather than
acted on, which is the only safe policy: a stale read of the tape is worse than
no read. Jev's documented 70–500 ms sits inside that budget. A model in the
3–329 s range does not, which is the quantitative version of the correction at
the top of this file.

Two safeguards run in code on every tick, independent of the model — a stop
loss and a drawdown kill switch. Nothing that requires a network call should be
the only thing between a position and a loss.

## Real market data

```bash
python -m jevtrade.cli fetch --symbol BTC --out data/btc_1m.csv
python -m jevtrade.cli backtest --source csv --csv data/btc_1m.csv
python -m jevtrade.cli backtest --source live --symbol ETH   # fetch and run
```

Kraken's public API gives 720 recent OHLCV bars, no key needed. There is no
order book in a bar, so `bid`/`ask` and resting sizes are **reconstructed** from
the bar's range and close — an approximation, flagged in the state that Jev
sees. On real ETH 1-minute bars the mock scores a 50.2% hit rate: a coin flip,
which is what an untuned heuristic on real data should look like, and a useful
check that nothing here is rigged.

## Testing

```bash
python -m pytest -q      # 76 tests
```

They cover the documented request/response schema, each policy gate, position
accounting through a flip, the latency and deadline behaviour, the stop and
kill switch, and two honesty checks on the simulator itself: no edge when
`alpha=0`, and an edge when it is switched on.

## Layout

| file | role |
|---|---|
| `feed.py` | synthetic microstructure sim, CSV replay, Kraken fetch |
| `features.py` | every number in the system |
| `discretize.py` | numbers → closed-vocabulary words |
| `questions.py` | the six typed questions |
| `jev/client.py` | `POST /v1/systemone`, stdlib only |
| `jev/mock.py` | offline stub, clearly labelled |
| `policy.py` | answers + confidence → target position |
| `engine.py` | the loop: latency, deadlines, safeguards |
| `execution.py` | fills, spread, impact, fees |
| `metrics.py` | P&L, hit rate, calibration, break-even fee |
| `baselines.py` | rule strategies for comparison |

TypeSafe also ships first-party SDKs (`pip install typesafe-sdk`,
`@typesafe-ai/sdk`). This repo speaks HTTP directly so the wire format stays
visible, since with a System One model the wire format *is* the interesting
part: there is no parsing step and no schema-repair retry, because the model
cannot emit a value outside the schema you declared.

## Sources

- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [API reference](https://docs.typesafe.ai/api) · [Primitives](https://docs.typesafe.ai/primitives) · [Confidence](https://docs.typesafe.ai/confidence) · [Models & pricing](https://docs.typesafe.ai/models)
- [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13) — the failure modes this design is built around
- [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out) · [Composite scoring](https://docs.typesafe.ai/patterns/composite-scoring) · [Confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing)
