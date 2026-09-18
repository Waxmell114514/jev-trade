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

## The live demo

```bash
export TYPESAFE_API_KEY=sk-...
python -m jevtrade.server            # http://127.0.0.1:8787
```

A page that watches Jev trade a live BTC curve, one decision per second.

![the live demo](docs/demo.png)

The point of the page is the **latency budget**. Every snapshot is a 1000 ms
window, and the hero tile shows how much of it the decision consumed —
typically a third. The bar chart puts one bar per call under a rule line at the
deadline: a decision slower than one snapshot describes a book that no longer
exists, so it is dropped rather than traded. Measured over a live session:

```
p50 386 ms   ·   p95 490 ms   ·   0 dropped out of 366 decisions
1093 input tokens per decision = $0.000046
```

The right-hand column is the part worth watching: `What Jev was shown` is the
state in words, and `What Jev answered` is the six typed answers with their
probability distributions, updating live. You can watch a Choice swing from
`up` to `down` and see the gate that follows from it.

This path uses **Kraken's ticker, which carries the real best bid and ask with
their resting sizes** — unlike the 1-minute OHLCV replay above, where the book
has to be reconstructed. The feature window is primed from the public trade
tape, so the demo starts trading immediately instead of waiting a minute.

Notes on what the P&L means: trades cross the real spread, but no exchange fee
is modelled by default (`--fee-bps` adds one), and it is paper trading against
observed prices, not order placement. It is a demo of a decision loop, not a
trading result.

```
--symbol BTC|ETH|SOL   --port 8787        --interval-ms 1000
--max-units 0.05       --fee-bps 0        --provider auto|jev|mock
```

The API key stays in the server process; the browser never sees it. Without a
key the page runs the offline simulator and says so. `?static=1` renders one
snapshot instead of holding the stream open, and `?theme=light` forces a mode.

`tools/record_session.py` captures a run's event stream to JSON, which is how
the shareable replay of a real session was built — 199 live decisions, three of
which came back past the deadline and were dropped.

## Market making: where this model actually fits

The directional strategy above is the wrong job for Jev, and the numbers say so
— a taker needs a sub-0.32 bp fee to survive, and predicting direction is the
one thing a System One model is weakest at.

Market making is the better fit, for a reason worth stating precisely. The
mechanical part — quote both sides, earn the spread, lean against inventory —
is arithmetic, and a traditional program does it better than any model. What
kills a market maker is **adverse selection**: getting picked off by someone
who knows something. Deciding *"is now a bad time to be showing this side"* is
a defensive, low-cardinality judgment over text, and it has to be made in
hundreds of milliseconds. That intersection is the niche: too semantic for a
rule, too fast for a frontier model.

```bash
python -m jevtrade.cli mm --seeds 20                  # the comparison
python -m jevtrade.cli mm --event-impact-bps 0        # the falsifiability run
```

### Jev goes off the quote path

```
quoting engine (pure code, every tick) ──> bid / ask
        ▲ reads, never waits
   risk posture  {per-side spread, per-side size, ttl}
        ▲ lands ~400 ms later
   Jev ◀── a headline arrives, or the tape turns one-sided
```

The quoter never blocks on a network call. Jev sets a *posture*; code sets
quotes. This is what makes the latency budget survivable: a stale posture is
merely conservative, while a stale directional bet is fatal. If the model is
slow or down, the stance defaults to defensive.

It also buys something a volatility trigger structurally cannot do. Knowing the
direction lets you **pull the side about to be picked off and keep quoting the
other** — protected and positioned at once. A vol spike has no direction, so it
can only widen both sides.

And uncertainty has a safe direction here, which it does not in directional
trading. There, low confidence means stand aside and earn nothing. Here it
means quote wider and keep earning. Calibrated confidence maps onto a continuum
of profitable states rather than an on/off switch.

### The experiment

Four arms over identical markets — the price path and event schedule run on
their own RNG stream, so every arm sees the same tape:

| arm | what it does |
|---|---|
| `naive` | quotes through everything |
| `vol` | widens both sides when realised vol spikes — the traditional defence |
| `keyword` | pauses quoting on a news word — what desks actually run |
| `jev` | reads the headline and the tape, sets a stance per side |

Counterparties are a mix of **noise flow** (fill probability decays with quote
distance) and **informed flow** (arrives after a material event, trades the way
the price is about to move, and crosses only while the quote is cheap relative
to the move it already knows about). Without that asymmetry a market-making
backtest is meaningless — quoting a random walk always "wins".

Scoring is by **markout**, the only measure that tells a market maker anything:
each fill splits into the half-spread captured at the fill and what the mid did
afterwards. The second term is adverse selection.

The `keyword` arm is built to be a *strong* competitor: its dictionary catches
all six obviously material headlines in both directions. Its only failures are
the four deliberately subtle ones and the seven it cannot read — denials and
re-reports. Beating a strawman would prove nothing.

### Results

20 independent markets, 6,000 ticks each, real `jev-latest`:

```
arm           mean net   std error    vs naive
----------------------------------------------
naive            1,755         767          +0
vol              1,500         928        -255
keyword          8,820         900      +7,066
jev              5,829         752      +4,074

paired jev - keyword: -2,991 +- 654  (outside the noise)
jev precision 97%   recall 62%
```

**The keyword rule wins, and the gap is real.** Jev's reading is close to
perfect — it flags 97% of its stances on genuinely material news, and in a
separate check it scored `p(denied) = 0.99` on every denial and freshness
`0.11` on every re-report — but it acts on only 62% of material events, and in
this world a miss costs more than a false alarm.

The falsifiability run makes the trade-off visible. Set `--event-impact-bps 0`
so headlines print but the price never moves, and every defensive stance is
pure cost:

```
naive           20,479          +0
keyword         13,993      -6,486   (-32%)
jev             19,689        -791   ( -4%)
```

Jev's false alarms cost **eight times less**. That is precision, measured.

So there is a crossover, and it sits at how much of the flow is informed:

```
 toxicity    naive  keyword      jev   jev-kw   +-se  winner
     0.00   20,333   19,092   20,828   +1,736    717  jev
     0.25   15,183   16,876   16,725     -152    916  tie
     0.50   11,303   13,462   12,083   -1,379    470  keyword
     1.00    3,391   10,127    5,266   -4,861    778  keyword
     1.50   -4,757    5,478     -885   -6,363  1,145  keyword
```

**Precision pays when false alarms dominate; recall pays once adverse selection
does.** If your news feed is mostly noise, reading it is worth a lot. If every
event is a real 20 bp move, pausing on everything is hard to beat.

Two things that did *not* work, recorded because they were predictions:
acting harder once the floor is cleared made things worse (the lost spread
exceeds the protection), and raising the event rate did not tip the balance
toward precision — more news means more *material* news too.

### What this does not show

- **The world is built so that semantics matter.** Denials and re-reports are
  55% of the event mix by construction. Measured against real feeds (see the
  next section) the real figure is closer to 95% non-material — noisier than
  assumed, which favours precision, but on a feed with no forward signal left
  in it.
- **The "subtle" labels are fiat.** Four headlines are written to read as
  immaterial while the simulator moves the price 22 bp anyway. Jev reads them
  as immaterial — as would a person. Most of the recall gap is these, so it may
  be an artifact of my labelling rather than a limit of the model.
- **Jev is a sampler.** Re-drawing its answers moved a single configuration by
  ~30%. Every number here fixes one set of answers across all arms and reports
  a standard error across markets; treat differences smaller than ~2 standard
  errors as nothing.
- Fills, impact and flow are a model, not an exchange. No queue position, no
  cancel latency, no fee tiers or rebates — and rebates are most of why real
  market making works.

## Does real news flow look like that? (measured)

The market-making experiment rests on one assumption: that a useful share of
real headlines are the kind a keyword rule misreads. That is testable, and
unlike the synthetic world it has a ground truth nobody has to label — **what
the price actually did afterwards**.

```bash
python -m jevtrade.cli news
```

209 de-duplicated headlines from 11 public crypto RSS feeds over 58 hours,
aligned to 5-minute BTC bars. A headline "moved" if BTC made a ≥2σ excursion in
the next 15 minutes *and* was not already moving in the 15 before — because a
move already underway is one you are too late to trade. The control throughout
is **randomly timed fake headlines** over the same window.

```
arm                   alerts   fire  moved    null      z
all headlines            209  100%   5.7%    5.0%   +0.47
keyword rule              67   32%   9.0%    5.1%   +1.37
jev top-40                40   19%   2.5%    5.0%   -0.74
```

Nothing clears the noise. Crypto headlines as a class are followed by a
material move about as often as a randomly chosen moment is.

### The crux

That comparison is not yet decisive, because news clusters around busy moments
and a busy moment stays busy on its own. So the real test holds recent
volatility fixed: each flagged headline is compared against random moments with
a comparable move already behind them.

```
group         condition            n   after    null      z
jev top-40    quiet before        27    0.51    0.53   -0.22
jev top-40    already moving      13    1.71    0.79   +4.12
keyword       quiet before        41    0.55    0.53   +0.30
keyword       already moving      26    1.28    0.78   +3.21
```

**A headline arriving into a quiet tape predicts nothing** — z ≈ 0 for both
arms, and this is the well-controlled half of the table. The "already moving"
rows look strong, but those are headlines *reporting* a move in progress, and
the volatility matching there is coarse enough that the number is not evidence
of anything.

The qualitative read tells the same story. Jev's highest-risk headlines are
exactly the ones a person would pick out:

```
0.268   SEC releases long-awaited innovation exemption
0.259   Fed Hikes Rates for the First Time Since 2023, Bitcoin Spikes
0.196   Binance Alert: System Upgrade to Suspend Deposits and Withdrawals
0.176   Bank of Japan Follows Fed and ECB With Rate Hike to 1.25%
```

The second one says it outright. By the time an article about a Fed decision is
written, edited and published to RSS, the spike it describes is history.

### What this settles, and what it does not

**The distribution assumption is half right, in the direction that matters.**
Real flow is far noisier than the simulation assumed — roughly 95% of headlines
are followed by nothing, against 55% non-material in the synthetic mix. That
pushes the operating point deep into the regime where precision beats recall,
which is exactly where Jev won and the keyword rule bled (−4% versus −32%). On
this feed a keyword rule fires on **32%** of everything, so at a ~5% base rate
at least six of every seven alerts are false by arithmetic alone.

**But the premise underneath it does not survive as stated.** There is no
tradeable forward signal here to be precise about. Jev reads the headlines
correctly — its risk scores top out at 0.27, i.e. it judges almost nothing on
this feed to be materially BTC-moving, which the tape agrees with — and reading
them correctly is worth nothing when the information has already been priced.

The likely culprit is the data source, not the model. RSS publication
timestamps trail the underlying event by minutes; news desks that trade on
headlines use millisecond-stamped wire feeds. So the honest scope of this
result is: **consumer crypto RSS carries no tradeable forward information at
5-minute resolution.** It is not a finding about news in general.

Other limits worth holding onto: 209 headlines over one quiet 58-hour window is
a small sample and underpowered for rare events; 5-minute bars are coarse for a
reaction that happens in seconds; and only BTC was tested. Settling this
properly needs a low-latency wire feed with millisecond stamps and an L2 book —
which is the experiment to run before building anything on this idea.

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
python -m pytest -q      # 133 tests
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
| `live.py` | Kraken live top-of-book, real bid/ask and sizes |
| `server.py` | the demo server: trading loop + SSE |
| `web/index.html` | the demo page |
| `mm/market.py` | quoting sim with informed (toxic) flow |
| `mm/events.py` | synthetic headlines + the keyword competitor |
| `mm/questions.py` | the six headline questions |
| `mm/strategies.py` | the four arms |
| `mm/metrics.py` | markout decomposition |
| `news/feeds.py` | real headlines from 11 public RSS feeds |
| `news/label.py` | labels each headline by what the tape did next |
| `news/study.py` | the null controls and the crux test |

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
