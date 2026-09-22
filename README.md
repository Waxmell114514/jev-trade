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

No required dependencies, no build step, Python 3.10+.

```bash
python -m jevtrade.cli backtest --baselines     # run and score the loop
python -m jevtrade.cli decide                   # one decision, fully unpacked
python -m jevtrade.cli sweep                    # what latency costs you
python -m jevtrade.cli fetch --symbol ETH       # real bars from Kraken
python -m jevtrade.cli listing --provider mock  # read exchange announcements
python -m jevtrade.cli fx --provider mock       # read central banks, graded on spot FX
python -m jevtrade.cli fx --context --provider mock  # statements read against what was priced
python -m jevtrade.cli fx --dots                    # the dot-plot rule, no model at all
python -m jevtrade.cli fx --presser --provider mock # the press conference, 30 minutes later
```

`--presser` reads PDF transcripts, which is the one thing the standard library
cannot do: `pip install 'jev-trade[pdf]'` adds `pypdf`, and without it that arm
reports itself dark and everything else runs unchanged.

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

## Composed judgment: reading an announcement in one second

The news study above put Jev on the wrong end of a trade-off. A headline that
moves the market is rare and worth a lot, so whoever trades it can afford
five seconds and five cents of a frontier model; a 400 ms answer buys nothing
there. Jev's advantage — frontier-grade judgment, cheap, fast — pays where
there are *many* judgments to make and each one is worth little. And, as it
turns out, where several of them have to be composed.

### Width is free, depth costs a round trip (measured)

Against the real API, from this sandbox, with a ~1k-token document:

```
questions per request      median latency
   1                          412 ms
   6                          420 ms
  24                          392 ms

5 sequential rounds × 6 questions, each round given the previous answers:
  2.0–2.4 s, about 400 ms per round
```

Ten judgments cost the same as one if they share a request. The budget is
therefore *rounds* — conditional layers, "ask A, then depending on A ask B" —
not judgments. One second buys two rounds from here and perhaps three to five
from a colocated box, each carrying dozens of questions. So a judgment tree
built on this model should be **wide and shallow**, which is also what keeps
errors from compounding: ten sequential 90%-accurate steps are 35% accurate,
two wide ones are not.

### The opportunity: exchange listing announcements

Binance publishes listings, delistings and every other notice through one
public CMS endpoint with a millisecond `releaseDate` — the same endpoint the
public "listing sniper" bots poll. Those bots are fast enough. What they cannot
do is read: they match the title (`Will List`, `Removal`) and trade every
ticker in it. That fails on exactly the announcements that carry the most
money — three tickers where one is the subject, a pair removal that is not a
delisting, a "Will Remove the Seed Tag" that the word `remove` turns into a
short, a Seagate `(STX)` that buys Stacks.

`jevtrade/listing/` reads each announcement with a two-round tree:

* **Round one, wide and speculative:** what kind of notice is this (listing,
  more markets for an existing token, delisting, warning tag, housekeeping,
  tokenized stock), which way it cuts, how big, is it conditional — and, for
  every ticker the *code* found in the text, whether that ticker is what the
  announcement is about rather than a quote currency, collateral or a passing
  mention. Jev never returns a string, so extraction is code and judgment is
  the model.
* **Round two, only if a ticker cleared that bar:** the direction for each
  such token specifically, a reversed-framing check ("would a holder have no
  reason to act?") whose errors are partly independent of the first framing,
  and whether the substance was already public. No second round, no trade.

Every number is arithmetic on the model's probabilities; the model computes
nothing. On six recent, deliberately varied announcements against the real
API the reader took **0.89–0.97 s** for two rounds (0.45 s when it stopped
after one), on 2–4k input tokens — about $0.0001 per announcement.

### How it will be graded

```bash
python -m jevtrade.cli listing --days 180            # real key: reads with Jev
python -m jevtrade.cli listing --days 30 --provider mock
```

Every arm turns an announcement into (token, side) signals and every signal is
scored the same way: enter at the open of the minute *after* the release —
deliberately up to 59 seconds late for a reader that answers in one — and
take the signed log return at 1, 5, 15 and 60 minutes, on Binance if the
token still trades there, else OKX, else Coinbase, in that fixed order for
every arm. The null is the same tokens and sides at random moments within
five days, so whatever drift those tokens had that week the null has too. The
release minute itself is reported but never credited to anyone.

Three controls sit beside the reader: the title-matching bot (the incumbent),
the same rules over tickers from the body as well (so that *seeing* the body
and *judging* it are separable), and every mentioned ticker bought (the base
rate for being mentioned at all). Model answers are cached on disk per
announcement, so the threshold sweep and every re-run score one fixed set of
answers rather than re-sampling the model. The run has not been done yet; the
numbers will go here when it has.

## The same reader, pointed at central banks

The listing study aims the reader at a feed whose incumbent is fast but cannot
read. This one aims it at a feed whose incumbent *can* read and is slow — a
human on a desk with two statements side by side — and where the part a machine
can already handle is gone before anyone blinks.

**The thesis, stated so it can fail.** In FX the numeric releases are priced
within five minutes: NFP, CPI and the headline rate itself are numbers, every
machine on the tape has them at the same millisecond, and the trade is
subtraction. The *text* events are not. A rate-decision statement, a set of
minutes, a press conference, a speech, a line about the exchange rate being
"excessive and one-sided" — these keep moving price for fifteen to sixty minutes
because somebody has to read them first. A model that answers thirty questions
about a statement in one 400 ms round is early relative to a fifteen-minute
digestion. That window, and nothing wider, is what `jevtrade/fx/` tests.

### Why this window and not another (measured, one week, small)

Median |move| in bps on the spot pair of the event's currency, over the
ForexFactory calendar for 2026-09-14 to 09-18, Yahoo 5-minute bars:

| events | n | +5m | +15m | +30m | +60m |
|---|---:|---:|---:|---:|---:|
| High impact, numeric | 10 | 2.2 | 5.7 | 4.2 | 6.6 |
| High impact, text | 4 | 3.3 | **12.1** | **19.5** | **20.0** |
| Medium impact, numeric | 7 | 1.6 | 3.6 | 1.2 | 3.5 |
| Medium impact, text | 4 | 2.3 | 5.2 | 4.6 | 8.2 |

The caveats are larger than the table: **one calendar week, four High-impact
text events, and 5-minute bars**, which is an anecdote with a standard error,
not a result. It is here because it is what motivated building the thing, and
because the shape is what the thesis predicts — numeric events flat after five
minutes, text events still going at sixty.

The individual cases say the same thing more legibly. The BoJ press conference
on 2026-09-18 moved USDJPY **0.6 bp in five minutes and 23.9 bp in fifteen**:
nothing happens while it is being read, and then it happens. The 2026-09-16 FOMC
statement differs from the 2026-07-29 one in **7 of its 9 sentence slots**, and
the hike itself was on the calendar — 4.00% against 3.75% previous — so the
number was not the news. The reading was in the changed sentences: the inflation
paragraph went from "elevated relative to the Committee's 2 percent goal, in
part reflecting supply shocks" to a flat "Inflation remains elevated," and the
vote went from **9–3 with three dissents for a hike** to **12–0**.

(With the site chrome left in the page, as a first pass did, the same pair of
statements reads as 11 changed slots out of 27; the counts above are after the
body extraction in `documents.py` throws the navigation away. Same story, fewer
sentences.)

### The feeds

| source | reachable | depth | timestamp |
|---|---|---|---|
| `federalreserve.gov/json/ne-press.json` | yes | 4633 rows to 2006 | US Eastern local, minute; 468 old rows have none |
| `…/ne-speeches.json` | yes | 1336 rows | same |
| `…/ne-testimony.json` | yes | 280 rows | same |
| Fed statement / speech pages | yes | immutable | — (body text) |
| ECB `rss/press.html` | yes | last 15 items | RFC 2822, `+0200` |
| BoJ `en/rss/whatsnew.xml` | yes | last 44 items | `+0900`; links are usually PDFs |
| BoE `rss/news` | yes | last 50 items | `+0100` |
| ForexFactory `ff_calendar_thisweek.json` | yes | **this week only** | ISO with offset |
| Yahoo `v8/finance/chart/{sym}` | yes | 60d of 5m, 7d of 1m | epoch seconds, UTC |
| Dukascopy `datafeed/{SYM}/…/{HH}h_ticks.bi5` | yes | ticks back to 2003 | ms into the UTC hour |
| Reuters, Bloomberg, X API | no | — | — |

Only the Fed has an archive. Everything else is a window onto the last week or
two, which is why the Fed is the primary source and the other three are there to
show the tree is not Fed-shaped. `lastweek` and `nextweek` both 404 on the
calendar, so a `collect` step stores the current week under an ISO-week key and
the numeric-surprise baseline is live only for the weeks somebody ran it. No
calendar is ever reconstructed for a past week: a consensus invented after the
fact is not a consensus, and that arm would win for the wrong reason.

### The sign convention

Stated once, in one table, and tested:

| issuer | currency | hawkish means | pair traded | hawkish side |
|---|---|---|---|---|
| Fed | USD | USD strengthens | `EURUSD=X` | short |
| ECB | EUR | EUR strengthens | `EURUSD=X` | long |
| BoE | GBP | GBP strengthens | `GBPUSD=X` | long |
| BoJ | JPY | JPY strengthens | `JPY=X` (USDJPY) | short |

A hawkish Fed sends EURUSD down and a hawkish BoJ sends USDJPY down, because the
currency in question is the *quote* side of those pairs. Getting this backwards
inverts the entire study while leaving every number plausible, so it lives in
one dict and has its own test.

### The tree

**Round one, one request, everything speculative** — what kind of text this is
(rate decision, minutes, speech or testimony, press conference or interview, FX
or intervention comment, data or survey, operational, other), whether it is
policy at all, which way it leans for the issuer's own currency, whether there
is anything new in it, how big, whether the guidance moved, whether it is a
surprise against what the text implies was expected (and against the calendar
forecast, when a snapshot covers it), and how far it goes up a five-level
intervention ladder: *no mention of the exchange rate → officials are watching →
moves are "excessive or one-sided" → "ready to take decisive action", or a rate
check → intervention announced or confirmed.*

Then, for each of up to twelve sentences that changed since the previous edition
— found by `difflib` on sentence lists, not by the model — two more questions:
which way *that sentence* cuts, and whether the change is substance or
rephrasing. With a full diff that is **32 questions in one round**, and width is
free.

**Round two, only if round one found something** — a decisive stance, a material
sentence change, or intervention language near the top of the ladder. It asks
the direction again with the framing reversed ("if you had to take a position
for the next hour, which side?"), a holder check ("would a trader long this
currency be unaffected?"), and the horizon. Two rephrasings of one question make
partly independent errors; that is the cheapest redundancy there is. **No second
round, no trade** — a document whose confirmation never ran scores zero.

Every number is arithmetic on the probabilities: strength is stance × the
reversed-framing confirmation × magnitude × policy relevance, lifted a little by
surprise and halved by the complement of "new information". The model multiplies
nothing.

### The baselines and the grading

```bash
python -m jevtrade.cli fx --days 60                      # real key: reads with Jev
python -m jevtrade.cli fx --days 60 --provider mock      # offline, keyword stub
python -m jevtrade.cli fx --snapshot-calendar            # store this week's calendar
```

Four arms are scored the same way. **`keyword-bot`** counts hawkish words
against dovish ones and trades the difference — the incumbent, and the thing to
beat; it reads "the Committee no longer expects to raise rates and will not
tighten further" as hawkish. **`surprise-bot`** takes the printed rate minus the
snapshotted forecast, which is the incumbent that actually wins on numbers, and
reports *not available* for every week without a snapshot. **`all text`** trades
every document, so the table shows whether central-bank text moves the tape at
all relative to nothing happening — if that arm is flat, nothing downstream
matters. **`reader`** is the tree at a strength threshold.

Every signal enters at the open of the first bar *after* the published
timestamp — up to five minutes late on 5-minute bars, deliberately — and is
measured by signed log return at 5, 15, 30 and 60 minutes. The null is the same
pair and the same side at random moments within five days, drawn *only where the
tape has bars*: spot FX is shut from about Friday 21:00 to Sunday 21:00 UTC, so
a third of naive draws land in a hole, and a 60-minute return computed by bar
index across a Friday close would be a 51-hour return in disguise. Horizons are
therefore checked against the bars' own timestamps and dropped when the window
is not contiguous. Model answers are cached per document and tree version, so
the threshold sweep scores one fixed set of answers rather than re-sampling.

### Grading on ticks

Five-minute bars were the binding constraint, not the feed. Yahoo serves sixty
days of them, which is thirty documents and five reader trades; and a bar has no
bid and no ask, so every arm was being graded at a mid price nobody is quoted.
Dukascopy publishes free tick files back to 2003 for the majors and fixes both.

**The format** (probed from this environment, not assumed):

```
https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
```

`MM` is **zero-based** — January is `00` — while `DD` and `HH` are ordinary
two-digit fields and `HH` is the UTC hour. The body is LZMA (`lzma.decompress`,
standard library). Decompressed it is consecutive 20-byte big-endian records,
`struct.unpack(">IIIff")` = *(milliseconds since the start of the hour, ask, bid,
ask volume, bid volume)*. Prices are integers scaled by **1e5**, or by **1e3**
when the quote currency is JPY: `115510 → 1.15510` on EURUSD, `156225 → 156.225`
on USDJPY. An hour with no ticks — every weekend hour of seventeen years, most
holidays — answers `200` with a zero-byte body, and a date the feed does not
have answers `404`; both are cached as "no ticks", because a run that re-asks for
every Saturday since 2009 spends its afternoon on them. A `5xx` or a dropped
connection is the opposite: transient, retried with backoff, and never cached,
since caching one would turn a bad minute of network into a permanent hole.

**The entry rule, stated so the cost is visible.** A signal enters on the first
tick at or after `published + latency`, pays the **ask** to go long and hits the
**bid** to go short, and exits at the **mid** at the horizon. So the round trip
pays half the spread, which is the friendly end of the honest range — a trade
that had to hit the other side on the way out would pay `spread` more. Two new
columns come with it:

- **`rush`** — how far the mid moved between the publication timestamp and the
  entry tick, signed by the side taken. Positive means the move had already gone
  the reader's way before it could act: that part of the edge belongs to whoever
  was faster.
- **`sprd`** — the bid/ask spread at entry, in bps. EURUSD is under a bp in
  London hours and several times that in the minute after a statement, which is
  exactly the minute every arm here trades in.

A `no spread` row under each arm gives the same trade measured mid-to-mid — the
bar study's number — so the difference between the two lines is what the book
costs.

**The latency sweep is the whole question in one table.** `--latency-sweep` runs
the *same* signals at 0, 1, 5, 30 and 120 seconds and prints one row each. Zero
is the counterfactual nobody has: filled on the first tick after the timestamp
itself. One second is a model that answered and hit the button. One hundred and
twenty is a person who read the statement. If the rows are flat, speed was not
what was being paid for and the reader is buying comprehension, not latency. If
they decay, the slope *is* the price of being slow, in basis points, and it is
the number a scalper actually wants. The table sweeps the reader's own signals,
falling back to `all text` when the threshold leaves the reader with fewer than
five; whichever it used is named in the caption.

**The sample this buys** (counted, 2009-01-01 to 2026-09-19, not estimated).
The three Fed archives hold **5,237 rows with a minute timestamp** in that
window; exactly one row in it carries a date and no time of day, and is dropped
rather than guessed at. Filtered to the three kinds that can move a currency
they are **858 monetary-policy releases, 1,116 speeches and 213 testimonies —
2,187 documents**, 150 of them titled "FOMC statement", running 85 to 191 a
year. The rest is noise the `pt` field separates out without a judgment call:
1,185 enforcement actions, 968 banking and consumer regulatory policy items,
548 other announcements, 349 orders on banking applications. So the tick judge
turns thirty documents into two thousand, and coverage on the tick side is
complete: EURUSD, GBPUSD and USDJPY all go back to 2003.

**The body extractor was re-checked on the old pages rather than assumed.** The
2015 site rebuild re-templated the whole archive, so a 2009 statement, a 2010
speech and a 2011 testimony all carry the same `col-xs-12 col-sm-8 col-md-8`
wrapper as a 2026 one. Over all 5,237 documents, **one** comes back under 200
characters — the 187-character September 2026 Implementation Note, which really
is that short. `collect` reports that count, because "the reader was
unconvinced" and "the reader saw nothing" are different results.

```bash
python -m jevtrade.cli fx --tape dukascopy --since 2009-01-01 --issuers fed \
    --horizons 1,5,15,30,60 --latency 1.0 --latency-sweep
```

Everything is cached so the run is resumable: readings per document (keyed on a
hash of the round-one state, not just the id — a statement whose previous
edition becomes visible in a longer window is a different question and is read
again), bodies per URL, tick hours per file as the compressed bytes they arrived
as. An interrupted run re-reads what it has and fetches only what is missing.

Fetching was the slow part until the connection was kept open. Measured here: a
fresh TLS connection to the feed costs nine to sixteen seconds before the first
byte, and the feed answers 503 when several are opened at once (six of eight
concurrent requests failed; the same eight one at a time all succeeded), which
capped the fetch at about five hour-files a minute. On a connection kept open
the next request costs 0.2 s. So the fetch holds one persistent connection per
worker (`--workers-io`; two to four is plenty), tunnelled through the proxy when
one is configured, drops it on any transport error and reconnects on the next
try. Of the roughly 15,000 hour files the 2009–2026 run needed, 20 stayed
unreachable after five tries; a failure is reported and never cached, so the
next invocation fetches only those.

### The run (real model, 60 days)

```bash
python -m jevtrade.cli fx --provider jev --days 60 --out runs/fx-jev-60d.json
```

Run on 2026-09-19: 30 documents from 2026-07-29 to 09-18 (Fed 11, BoE 11, ECB
5, BoJ 3), 14 went to round two, 8 questions in the median document and 32 with
a full statement diff, **median 693 ms per document, p90 928 ms, $0.004 of input
tokens** for the lot. Keyword and reader answers are the same cached set across
every row.

```
arm              signals traded    pre    bar      +5m     +15m     +30m     +60m  hit15    z15
keyword-bot           18     17     -1     +1       -0       -1       +2       +1    50%   -0.5
  s.e.                                             +-1      +-2      +-2      +-3
surprise-bot           0      0     (both snapshotted decisions printed in line with the forecast)
reader >=0.15          6      5     -1     +0       -1       -3       +5       +6    40%   -1.0
  s.e.                                             +-1      +-4      +-6      +-7
all text              30     29     -1     -0       +0       +0       +2       +3    54%   -0.0
  s.e.                                             +-0      +-1      +-2      +-3
```

Nothing clears the noise, and with five reader trades nothing could: −3 ± 4 bp
at fifteen minutes and +6 ± 7 at sixty is a sample size, not a verdict. What
the run does settle is that the tree reads the way it was meant to, which with
thirty documents is the part worth looking at:

- **The 2026-09-16 FOMC hike** came back `rate_decision`, hawkish at p = 1.00,
  magnitude 0.86, round two confirmed at 0.71, strength 0.47 — the highest of
  the run. Short EURUSD. The release bar had already moved −13 bp (the fastest
  actors' five minutes); the trade then made +10, +29 and +31 bp at 15, 30 and
  60 minutes. The number was on the calendar; the text was still moving price
  an hour later.
- **The 2026-07-29 hold with three hawkish dissents** came back hawkish at
  p = 0.51 with a round-two confirmation of 0.17: strength 0.03, no trade. The
  keyword bot shorted EURUSD on the dissent language and was down 8 bp at
  fifteen minutes. The reversed-framing round is what kept the reader out.
- **The ECB's 2026-09-10 decision** was read hawkish at p = 1.00, magnitude
  0.84 — confident, long EURUSD, and wrong: −13 bp at fifteen minutes, −12 at
  sixty. The press-conference statement 45 minutes later, also read long, made
  +9 at sixty. Confidence is not accuracy; one document says nothing about the
  rate, and this is the one to remember when the sample is larger.
- **Noise was filtered.** Court of Directors minutes, an AI-consortium minute,
  a chair appointment and a 2027 meeting calendar all came back neutral at
  p ≈ 1.00 with magnitude between 0.01 and 0.05 and were never traded. The
  keyword bot traded several of them.
- **The BoJ is blind.** Its RSS links are PDFs, the body extractor is standard
  library only, so those documents arrived with empty bodies and the reader
  classified the 2026-09-18 hike from its title alone (hawkish 0.82, strength
  0.03). The issuer with the largest text-driven moves in the motivating table
  contributed nothing. A PDF reader is the cheapest improvement on this list.
- **The surprise arm is empty because the surprises were.** The FOMC printed
  4.00% against a 4.00% forecast and the BoE 3.75% against 3.75%. A first cut
  of this arm shorted GBPUSD on the BoE decision: the calendar match had picked
  the "MPC Official Bank Rate Votes" row, whose forecast is "3-0-6", and the
  rate parser had read the 2% inflation target as the rate. Both are fixed and
  tested; the table above is from after the fix.

The binding constraint is the judge, not the feed. Sixty days of five-minute
bars gave thirty documents; the tick judge above lifts the same study to 2,187,
on the order of $0.30 of input tokens, and puts the bid/ask and the entry
latency into the number instead of leaving them out of it.

### The run (real model, Fed 2009–2026, ticks)

```bash
python -m jevtrade.cli fx --provider jev --tape dukascopy \
    --since 2009-01-01 --issuers fed --horizons 1,5,15,30,60 \
    --latency 1.0 --latency-sweep --out runs/fx-jev-fed-ticks.json
```

Run on 2026-09-19: **2,187 documents** from 2009-01-06 to 2026-09-18 (858
monetary-policy releases, of which 150 FOMC statements; 1,116 speeches; 213
testimonies), 1,142 went to round two, 11 questions in the median document,
**median 741 ms per document, p90 929 ms, $0.363 of input tokens, 476 s of wall
time** for the reading with three workers. Graded on EURUSD ticks: entry on the
first tick at or after the release plus one second, paying the ask to go long
and the bid to go short, exit at the mid; null of the same side at random
moments within five days where the tape has ticks. Events whose hours the feed
never served are absent from the *traded* counts.

```
arm              signals traded    pre   rush  sprd      +1m      +5m     +15m     +30m     +60m  hit15    z15
keyword-bot         1219   1149     -0     +0   0.6       +0       -0       -1       -1       -1    45%   -0.8
  s.e.                                                   +-0      +-0      +-0      +-0      +-1
reader >=0.15        240    225     +1     +0   0.8       +2       +1       +2       +2       +1    51%   +1.3
  s.e.                                                   +-1      +-1      +-2      +-2      +-2
  null                                                    -0       -0       -0       -1       -1
all text            2187   2101     -0     +0   0.6       -0       -0       -1       -1       -1    46%   -1.7
  s.e.                                                   +-0      +-0      +-0      +-0      +-0

latency sweep (reader >=0.15, net of the half spread):
   entry  traded   rush  sprd    +15m   s.e.    null     z    +60m   s.e.    null     z
      0s     225     +0   0.8      +2      2      -0  +1.5      +1      2      -1  +1.0
      1s     225     +0   0.8      +2      2      -0  +1.3      +1      2      -1  +1.0
      5s     225     +1   0.8      +1      1      -0  +0.9      +0      2      -1  +0.6
     30s     225     +2   0.6      +0      1      -0  +0.2      -1      2      -1  +0.2
    120s     225     +2   0.5      +0      1      -0  +0.5      -0      2      -2  +0.5

threshold sweep (+5m):  thr 0.05: 447 traded, +1 +-1   0.15: 225, +1 +-1   0.25: 109, +3 +-2   0.40: 39, +3 +-4
```

What seventeen years say, in the order they matter:

- **Reading beats counting words, and neither is a trade.** The keyword bot's
  1,149 trades come out at −1 ± 0 bp fifteen minutes on; its *hawkish* calls
  lose outright, −1.5 ± 1.1 bp at fifteen minutes and −4.5 ± 1.7 at sixty over
  207 trades, because "elevated" and "tightening" in a sentence are not a
  hawkish sentence. The reader's 225 trades come out at **+2 ± 2 bp** with a hit
  rate of 51% and z = +1.3 against its null. Two basis points with a standard
  error of two is not a result anyone should size a book on. The all-text arm
  at −1 bp (z −1.7) is the base rate: Fed text at large does not move EURUSD in
  the direction the words point.
- **Whatever is there is gone in thirty seconds.** The latency sweep is the
  scalping question in one table: +2 bp entering at zero or one second, +1 at
  five, **0 at thirty and at 120**, while the *rush* column — the mid's move
  between the release and the entry, signed by the reader's side — climbs from
  0 to +2 over the same range. The direction the model reads is the direction
  the first half-minute goes, slightly more often than not, and after that
  there is nothing left to collect. A 400 ms reader is inside that window; a
  human is not. The spread at entry averages 0.8 bp, so the half-spread paid is
  a fifth of the gross.
- **It lives in the statements.** FOMC statements: 66 of 150 traded, +3.1 ± 4.2
  bp at fifteen minutes and −3.5 ± 6.2 at sixty — the move mean-reverts. Speeches
  and testimony: 119 traded, −0.1 ± 1.4 — nothing, and for a speech the
  published timestamp is the scheduled start, not the moment a wire ran the
  line, so nothing is the honest expectation. Dovish readings did better than
  hawkish ones (+3.1 ± 2.2 over 130 against +0.2 ± 1.8 over 95). Split by era,
  2009–2015 is −1.3 ± 4.4 (51 trades), **2016–2021 +6.1 ± 2.0 (82)**, 2022–2026
  −0.2 ± 2.1 (92): one cell of three at three standard errors is what a post-hoc
  split produces, not a regime, and it is reported so nobody has to find it.
- **The big calls.** Right: 2009-03-18 (long, +136 bp at fifteen minutes, the
  Treasury-purchase statement), 2020-03-23 (long, +61, unlimited purchases),
  2016-12-14 (short, +55, the hike with the dots), 2019-01-30 (long, +46, the
  "patient" pivot). Wrong: 2009-01-28 (long, −75), 2024-12-18 (long, −70, a cut
  the market took as hawkish), 2022-11-02 (short, −66 at fifteen minutes and −2
  at sixty, the statement read one way and the press conference the other). On
  the largest statements the reader is right more often than not and wrong by
  as much when it is wrong.
- **What it read.** Neutral 1,349, dovish 550, hawkish 288. Speech or testimony
  1,328, operational 305, minutes 287 (286 of them neutral), rate decision 195.
  The intervention ladder never went above 0.47, on three speeches about
  Treasury-market liquidity — the Fed does not talk about the dollar, and the
  tree noticed.

For the scalper who asked: reading central-bank text in under a second is worth
about two basis points of EURUSD on the documents the reader chooses, captured
inside thirty seconds of a scheduled release, with a standard error the same
size as the edge, and mostly on FOMC statements. That is a measurement, not a
strategy. What it rules out is the keyword bot; what it rules in is only that a
400 ms reading points the right way for the first half-minute a little more
often than not, at $0.36 for seventeen years of it.

### Statements in context

The run above buries a negative result. On the **150 FOMC statements** the tape
moves a lot — mean |move| 15 bp at one minute, 25 at fifteen, 37 at sixty — and
the reader's direction is a coin flip: 66 traded, **50% at every horizon**, and
the highest-strength bucket (14 trades) hits **43%**. The three worst calls all
have the same shape:

| statement | what the text said | what the market did |
|---|---|---|
| 2024-12-18 | a cut, and it reads dovish | took it as **hawkish** — the dots moved 2025 from three cuts to two |
| 2022-11-02 | read one way | the press conference an hour later read the other |
| 2009-01-28 | promised purchases | had already priced them |

The diagnosis is not that the model reads badly. It is that **a statement is
read against what the market expected, and the expectation is not in the
statement.** The absolute tree asks "is this a surprise?" with nothing to be
surprised against. So `fx/context.py` assembles the expectation and the tree
gets a second mode that asks the relative question instead.

**What goes in, where it comes from, and what its timestamp has to satisfy.**
Every item is stamped with the moment it became public, and `Context` refuses to
exist if any of them is at or after the release:

| piece | source | timestamp rule |
|---|---|---|
| rates and what they price | H.15 daily package (Treasury constant maturities) plus the federal funds effective rate | the last row **strictly before the release's UTC date** — the row named after the meeting day prints at 4:15 p.m. ET, two hours after a 2 p.m. statement |
| the dots | `fomcprojtabl{YYYYMMDD}.htm` (and `fomcprojtable…`, which March 2022 uses) | **concurrent**: published at the release's own minute, the one allowed exception, and labelled as such |
| the previous statement | `diff.py`'s `Editions`, with a fallback to the last statement when the Fed renamed the release | strictly earlier |
| the minutes | the last FOMC minutes press release before the meeting, followed to its HTML page | released three weeks after the previous meeting, so strictly earlier |
| intermeeting communication | the Fed speech and testimony archives | strictly inside `(previous meeting, this release)`, open at both ends |
| the pair into the release | the cached Dukascopy ticks | the last quote strictly before the release |

Numbers become words on the way in, as everywhere else in this repo: the model
is told *"the six-month bill yields 4.30 percent against an overnight rate of
4.58: the market prices roughly one quarter-point cut within six months"*, never
handed a bill yield and a funds rate to subtract. The raw values stay on the
dataclasses. Two sanity rules are code's job and are stated because they bite:
`baseline.announced_rate` reads "at 0 to 1/4 percent" as **1.00**, which is the
wording of every statement from 2008 to 2015, so a target the effective funds
rate contradicts is refused and the sentence falls back to the effective rate;
and the longer-run dot is used to check the alignment of the previous SEP's row,
because a central tendency printed as a bare number reads as one more median.

**The relative questions** ride in round one next to the whole absolute tree,
because width is free:

| id | type | asks |
|---|---|---|
| `expected_action` | Choice | hike / hold / cut — what *the context* says was expected |
| `actual_action` | Choice | hike / hold / cut — what the statement did (code cross-checks this against the rate it parsed) |
| `relative_stance` | Choice | more hawkish / in line / more dovish **than the context implies** |
| `surprise_channel` | Choice | rate decision, guidance, balance sheet, dots, vote, assessment, nothing |
| `surprise_size` | Score | nothing they already had → a nuance → a clear surprise → the kind that sets the day |
| `versus_minutes` | Choice | more hawkish / consistent / more dovish than the minutes and the speeches |

Round two runs only on a decisive relative call or a large surprise, and asks it
reversed — *"a desk that had read this context, which side of EURUSD for the
next hour?"* — plus the holder and horizon questions. **The signal's sign comes
from `relative_stance` and never from `stance`**: on 2024-12-18 a reader that
gets the statement right has to answer *dovish* to the first question and
*more hawkish than expected* to the second, and that split is the whole
point of the mode. Strength is arithmetic:
`p(relative) × confirm × surprise_size × (1 − priced_in)`, with no half-measure
damper — the point of handing the reader the expectation is that "the market
already had this" is a complete answer. The mode has its own cache tag (`ctx1`),
so the absolute reader's `v1` answers are reused as they are and never re-asked.

**Two numeric baselines, both crude, both labelled.** The instrument that
actually prices a meeting is the fed funds future, and that is not free. What is
free is the six-month bill:

- `bill-surprise` — the decision the statement took, minus the one the bill
  implied. The bill prices the *path* over six months, so the spread over the
  effective rate is divided by the four meetings six months holds to get a
  per-meeting expectation. That division is an assumption and the arm inherits
  its error.
- `dots-surprise` — the change in next year's median dot against the previous
  SEP, on projection meetings only. The SEP has published a funds-rate median
  since September 2015 and printed ranges and a histogram before that, so the
  arm is dark for the first third of the archive — 44 of the 150 statements
  have readable dots.

```bash
python -m jevtrade.cli fx --context --provider jev --tape dukascopy \
    --since 2009-01-01 --horizons 1,5,15,30,60 --latency 1.0 --latency-sweep \
    --context-chars 12000 --out runs/fx-jev-fed-context.json
```

`--context` implies `--issuers fed` and reads statements only. It prints the
arms table, the latency sweep for the context reader, the confusion between
`expected_action`/`actual_action` and the rate the code parsed, the distribution
of `surprise_channel`, and the statements where the absolute and the context
readers traded against each other with the tape's verdict next to both. The
`--out` JSON carries every graded signal and, per statement, the timestamp of
every context item and both readers' answers.

Offline, `--provider mock` assembles the same context and answers it with a
rule: the pricing sentence for what was expected, the decision verb inside
"decided to …" for what happened, the gap between the two for the relative call.
That is a rule and not a reading, and it should score like one.

#### The run (real model, 150 statements, ticks)

Run on 2026-09-20. Context assembled for all 150 (rates 150, minutes 150,
previous statement 149, Chair speech 128, dots 44; median 10,210 characters).
The absolute reader's answers came from the 2009–2026 run's cache unchanged; the
context reader asked 32 questions in the median statement, 43 went to round two,
**median 537 ms per statement, $0.052** for the lot.

```
arm              signals traded    pre   rush  sprd      +1m      +5m     +15m     +30m     +60m  hit15    z15
reader-absolute       67     66     +3     +1   1.5       +1       +1       +3       +4       -3    50%   +0.5
  s.e.                                                   +-2      +-3      +-4      +-5      +-6
reader-context        14     13     +5     -1   1.4       +2       -1       -6       -3       -1    38%   -0.6
  s.e.                                                   +-6      +-7      +-8     +-10     +-12
bill-surprise         90     89     -0     +0   1.8       -1       -0       +2       +1       -1    51%   +0.5
  s.e.                                                   +-2      +-3      +-4      +-4      +-5
dots-surprise         31     31     +2     +3   1.8       +8      +14      +19      +18      +14    77%   +3.5
  s.e.                                                   +-4      +-5      +-6      +-6      +-8
all statements       150    149     -2     +1   1.6       -0       -1       -1       -0       -3    49%   -0.3
  s.e.                                                   +-2      +-2      +-3      +-3      +-4

what the model said the statement did, against the rate the code parsed (141 parseable):
  actual_action agrees with the parsed decision 132/141 (94%); expected_action matches what happened 133/141 (94%)
relative stance: in line 98, more dovish than expected 32, more hawkish than expected 20
where it put the surprise: economic assessment 45, vote 32, balance sheet 28, rate decision 19,
  forward guidance 15, nothing 11, projections or dots 0
```

Three things came out, in order of weight:

- **The context did not fix the coin flip; it turned the reader into an
  abstainer.** Given what the market already had, the model called 98 of 150
  statements *in line with expectations* and cleared the trading threshold on
  14. It read the facts right — 94% agreement with the parsed decision, 94% on
  what was expected — and stayed out of the largest moves in both directions:
  the ones the absolute reader had got right (2009-03-18, 2020-03-23,
  2019-01-30) and the ones it had got wrong (2024-12-18, 2022-11-02,
  2009-01-28). The 13 it did trade came out at −6 ± 8 bp at fifteen minutes,
  five of thirteen right. Thirteen is not a sample; abstaining on 136 is the
  result.
- **The direction was in the dots, and it is a number.** The change in next
  year's median projection against the previous SEP, on the 31 projection
  meetings since September 2015 with a readable median: **+19 ± 6 bp at fifteen
  minutes, 24 of 31 right** (sign test p = 0.003), +14 ± 5 at five, +14 ± 8 at
  sixty; without the three largest moves, +14 and 24 of 28. Eighteen shorts,
  thirteen longs. The move builds over the first quarter-hour (+8 at one minute,
  +19 at fifteen) rather than printing at the release. This is one arm of five
  and thirty-one observations, so it is a finding to test forward, not a
  strategy; but it is the only signed number in this whole section that is not
  within two standard errors of zero.
- **The reader had the dots and did not use them.** The context carried the
  dots as sentences ("Median projection for end-2025 moved to 3.9% from 3.4%")
  for 44 statements, and `surprise_channel` was *projections or dots* for none
  of them. It put the surprise in the economic assessment (45), the vote (32)
  and the balance sheet (28), which are the parts of a statement people argue
  about and not the parts the tape moved on. Whether that is the question's
  framing (the dots sat under a "context" heading, the question asked about
  "the statement") or the model, the next version of the tree should ask about
  the projections by name.

The bill-implied surprise scored nothing (89 trades, +2 ± 4), which is what a
six-month proxy for a one-meeting expectation deserves. The absolute reader on
statements alone reproduced its earlier +3 ± 4 at 50%.

For the scalper, this closes the loop from the other side: the semantic reader
is right about the words and no better than a coin on the direction, because
the direction on statement days is set by a table of numbers released with
them, and a table of numbers is not a reading job. Where Jev earned its keep
here was in saying *nothing surprising* 98 times — the abstention is correct
far more often than the keyword bot's trades are — and that is a risk filter,
not a scalp.

### Validating the dots

The run above leaves one signed number outside two standard errors of zero, and
it is not the reader: it is a table of numbers released with the statement.
**The rule, written down before it is tested again and not rewritten
afterwards:**

> On a projection meeting, take the sign of the change in the median
> federal-funds projection for **next year** against the previous SEP. Positive
> is fewer cuts, a stronger dollar: short EURUSD at the statement, out at the
> horizon.

That is what `--context` found — +19 ± 6 bp at fifteen minutes, 24 of 31 right,
sign test p = 0.003 — over 31 meetings on one pair, chosen from five arms. It is
the kind of number that is usually a coincidence, so `fx/dots.py` and
`jevtrade fx --dots` exist to give it four ways to fail.

```bash
python -m jevtrade.cli fx --dots --since 2012-01-01 \
    --pairs EURUSD,USDJPY,GBPUSD --latency-sweep --workers-io 2 \
    --out runs/fx-dots.json
```

**No model is called.** The dots are arithmetic, the sign is arithmetic, and the
tape does the grading; there is no provider argument and nothing to pay for.

**Out of sample means the years before the median was printed.** The SEP has
carried a funds-rate *median* row only since September 2015, which is where the
rule was found. From January 2012 — the first dot plot — the same page carries a
**histogram**: rate levels down the rows, years across the columns, the number
of participants at each level in the cells. The median is arithmetic on that,
so the fifteen meetings from 2012-01 to 2015-06 are a genuine hold-out for a
rule fitted on the printed ones. Three things about those pages had to be found
rather than assumed:

- **Blank cells vanish when the tags are stripped.** "0.50 1 2" could be one
  participant in the first year and two in the third, or one in the second and
  two in the fourth. So the table is read by walking `<tr>` and `<td>` and
  keeping the column positions, not by flattening the page.
- **The headings change three times.** 2012 writes them in Title Case
  ("Appropriate Pace of Policy Firming"), 2013–2015 in sentence case, and from
  March 2016 the stub column is "Midpoint of target range or target level"
  because the target became a range. The parser finds the table by its stub
  header, which names the funds rate in all three.
- **December 2012 has no projection page at all.** Its SEP was published inside
  the minutes, and its histogram is on `fomcminutes20121212epa.htm`, where the
  rows are *buckets* ("0.38 - 0.62") rather than levels. A bucket is read as the
  one quarter-point value it contains.

**The parser is validated where both exist.** On the 44 pages from September
2015 on that print a median row *and* the histogram, the histogram-derived
median reproduces the printed one in **198 of 199 year-cells, 43 of 44 meetings
exactly**. The single disagreement is the 2026-09-16 longer-run dot: eighteen
participants, the ninth and tenth both at 3.25, so the median is exactly 3.25,
and the page prints 3.2 — while the same page rounds 0.875 up to 0.9. The two
cannot both be right; the histogram is what this study uses, and the
disagreement is printed rather than reconciled. Finding that check working also
found a real bug: the September 2015 table has a **`-0.125` row**, one
participant projecting a negative funds rate, and a parser that only reads
unsigned numbers moves that page's 2016 median a whole eighth away from the
median printed six inches above it.

**The hold-out is thin, and that is the first thing the run will say.** Of the
fourteen out-of-sample meetings with a predecessor, **eight print no change at
all**: from January 2012 to December 2013 the median projection for next year
sat at 0.25 and never moved, because the funds rate was at its floor and the
first hike was still two years away. Only six of the fourteen carry a sign, all
of them between March 2014 and June 2015. Six trades will not confirm or refute
anything; what they can do is fail loudly, and the in-sample count they are
being compared against — 31 signed meetings out of 44 — is reproduced exactly by
this parser, which is the part that had to be got right first.

**Four variants are reported and none of them is ever chosen.** The current-year
median, the two-years-out median, the longer-run dot, and the sum of the year
medians are computed and printed next to the rule, labelled as variants. The
point is not to find the best one — that is how a coincidence becomes a
strategy — it is to let a reader see whether the rule is one lucky pick out of
five.

**Three pairs, one convention, in one dict with its own test.** A hawkish dot
plot is a stronger dollar. The dollar is the *quote* side of EURUSD and GBPUSD
and the *base* side of USDJPY, so the rule is short EURUSD, short GBPUSD and
long USDJPY. Getting that backwards inverts the study while leaving every number
plausible, which is why it is `USD_SIDE = {"EURUSD": -1, "GBPUSD": -1,
"USDJPY": +1}` and not a derivation.

**The latency sweep asks one question.** The move builds over the first quarter
hour rather than printing at the release — +8 bp at one minute against +19 at
fifteen — so unlike the reader's edge it might survive a human. The sweep runs
the same signals at **0, 1, 5, 30, 120 and 300 seconds**, and the five-minute row
is the question in the form somebody would actually ask it: *can a person who
opens the projection table and reads one number by hand still catch this?*

**The forward register.** Each run ends by parsing `fomccalendars.htm` — where
an asterisk on the date marks a projection meeting — and printing the rule in one
sentence and the projection meetings still ahead. As of 2026-09-20 those are
**2026-12-09, 2027-03-17, 2027-06-09, 2027-09-15 and 2027-12-08**. The per-meeting
records go into the `--out` JSON keyed by date, and a later run merges into that
file rather than recomputing it, so running the same command after each of those
dates *is* the forward test and the file accumulates it.

#### The run (no model, 58 projection meetings, three pairs, ticks)

Run on 2026-09-20 with `--since 2012-01-01 --pairs EURUSD,USDJPY,GBPUSD
--latency-sweep`. 59 projection tables parsed, 15 from the histogram years and
44 with a printed median; the two 2011 pages carry no funds-rate histogram at
all. **Histogram against printed median: 198 of 199 year-cells agree** over the
44 meetings that have both (the one miss is 2026-09-16's longer-run dot, which
the page rounds to 3.2 and the histogram puts at exactly 3.25). Of 58 meetings
the rule signs 37; the other 21 printed no change in next year's median, twelve
of them at the zero bound in 2012–2013 and 2020–2021.

```
arm                    signals traded    pre   rush  sprd      +1m      +5m     +15m     +30m     +60m  hit15    z15
EURUSD in-sample            31     31     +2     +3   1.8       +8      +14      +19      +18      +14    77%   +3.4
EURUSD out-of-sample         6      6     +2     +7   2.2      +12      +13      +29      +25      +21    83%   +3.1
EURUSD pooled               37     37     +2     +3   1.9       +9      +14      +20      +19      +15    78%   +4.2
  s.e.                                                         +-4      +-4      +-5      +-5      +-8
USDJPY pooled               37     36     -1     +5   3.0       +8      +12      +22      +21      +14    78%   +5.0
  s.e.                                                         +-4      +-4      +-4      +-5      +-8
GBPUSD pooled               37     36     +0     +3   3.7       +7      +12      +18      +18      +14    75%   +4.2
  s.e.                                                         +-3      +-4      +-4      +-5      +-7

at +15m: hit, mean, s.e., median, two-sided sign test
EURUSD in-sample     31   77%   +19   6   +26   0.003     USDJPY in-sample   30   80%   +21   5   +21   0.001
EURUSD out-of-sample  6   83%   +29   9   +32   0.219     GBPUSD in-sample   30   73%   +17   5   +21   0.016
EURUSD pooled        37   78%   +20   5   +29   0.001     USDJPY pooled      36   78%   +22   4   +23   0.001
                                                          GBPUSD pooled      36   75%   +18   4   +22   0.004

variants on EURUSD, reported and not chosen:
current-year   20   90%   +28   6   p 0.000      two-years-out   43   70%   +16   5   p 0.014
longer-run     21   57%    +6   7   p 0.664      sum-of-years    47   68%   +15   5   p 0.019

latency sweep, EURUSD, 37 signals, net of the half spread:
   entry   rush  sprd    +15m   s.e.     z    +60m   s.e.     z
      0s     +0   2.4     +23      5  +4.8     +18      8  +2.2
      1s     +3   1.9     +20      5  +4.2     +15      8  +1.8
      5s     +9   1.9     +15      4  +3.4      +9      7  +1.1
     30s     +9   0.7     +16      3  +4.4      +9      6  +1.4
    120s    +16   0.4      +8      3  +2.2      +4      6  +0.6
    300s    +18   0.4      +6      3  +1.7      +2      6  +0.3
```

What the validation says, and what it cannot:

- **The out-of-sample half agrees and is too small to prove anything.** Six
  signed meetings in 2014–2015 (the other nine printed no change, because next
  year's median sat at 0.25 through the zero bound), five of six right, +29 ± 9
  bp. The sign test on six is p = 0.22. It does not contradict the in-sample
  number; it cannot confirm it either. The pooled 37 is the honest headline:
  **+20 ± 5 bp at fifteen minutes, 29 of 37 right, p = 0.001.**
- **The other two pairs agree, and they are one factor.** USDJPY 78% and
  GBPUSD 75% over 36 each, both p ≤ 0.004. That is the dollar moving, seen
  three times, not three independent tests; what it rules out is a EURUSD-only
  artefact.
- **The rule is not one pick out of five.** Current-year's median does better
  (90% over 20), two-years-out and the sum do worse but hold (70%, 68%), the
  longer-run dot does nothing (57%, p = 0.66). The finding is "the near dots
  moved", and it degrades gracefully away from that.
- **A person can catch it.** Entering thirty seconds after the release, time
  to read a table by hand, still returns +16 ± 3 at fifteen minutes; two
  minutes late, +8 ± 3; five minutes late, +6 ± 3 with the rush column at +18,
  meaning most of the move has gone by. So this is a number that a machine
  reads in a second and a desk reads in half a minute, and the market takes a
  quarter of an hour to finish pricing. It is, once more, not a reading job.
- **The misses are informative.** Eight of 37 on EURUSD: 2018-12-19 (dots
  down, "autopilot" press conference, dollar up), 2023-03-22 (the SVB meeting),
  2021-09-22 (first hike pulled into 2022, dollar down), 2019-09-18, 2018-03-21,
  2025-06-18, 2024-03-20, 2014-06-18. The days the rule lost are the days the
  press conference or the moment mattered more than the table.

**The forward register.** The rule, the 58 records and the sign convention are
in `runs/fx-dots-2012-2026.json`; the next projection meetings on the FOMC
calendar are **2026-12-09**, 2027-03-17, 2027-06-09, 2027-09-15 and 2027-12-08.
Running the same command after each is the forward test, and the first check is
scheduled for the morning after 2026-12-09. A forward tally will be appended
here, one row per meeting, with no other change to the rule.


### The press conference, thirty minutes later

On 2022-11-02 the statement read one way and the press conference read the other,
and the tape followed the press conference: the reader was short EURUSD and −66
bp at fifteen minutes. That day is in the run's list of worst calls, and the
statement tree cannot see it, because the thing that moved the price is a
different document published half an hour later. `fx/presser.py` and
`jevtrade fx --presser` are that document.

```bash
python -m jevtrade.cli fx --presser --provider jev --tape dukascopy \
    --since 2011-01-01 --latency-sweep --workers-io 2 \
    --out runs/fx-presser.json
```

**The sources.** Every press conference since April 2011 has a transcript at
`…/mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf`. Which meetings had one comes
off the calendar pages — `fomccalendars.htm` for the current six years and
`fomchistorical{YYYY}.htm` for one year each before that — both of which link
the conference by its date. Counted over 2011 to 2026-09 that is **95
conferences**: three in 2011, five in 2012, four a year through 2018, then every
meeting from 2019 (nine in 2020, including the two intermeeting briefings). The
ninety-fifth is only there because the match is deliberately loose: the Fed's own
January 2026 row spells the link `fomcpressconf`, with two s's.

**A transcript is published after the conference it records.** So, exactly as
the speeches arm already says, this measures *"was it worth listening to"* and
not *"could you have traded it"*. What would make it tradeable is a live
speech-to-text feed off the video, and that is not free.

**`pypdf` is an optional extra** (`pip install 'jev-trade[pdf]'`). The rest of
this repository is standard library only and stays that way. When `pypdf` is not
importable the module says so once and the transcript arms are reported **dark**
— which is a result, in the same sense as the surprise bot's "not available" —
while the arms that need only the start time still run. That path has its own
test.

**When it starts is a rule, and only half of it is measured.** From 2013 the
statement is released at 2:00 p.m. ET and the Chair starts at 2:30, so the start
is the statement plus thirty minutes; every FOMC statement row in the archive
from 2013-03-20 on carries 14:00, and the Fed announced the change on
2013-03-13, but the half hour itself is the published schedule taken on trust.
**In 2011 and 2012 the statement went out at 12:30 p.m. and the Chair began at
2:15**, which is checked: the April 2011 press-conference page prints "FOMC
Meeting Statement (Released April 27, 2011 at 12:30 p.m.)" beside "Projections
Materials … (Released April 27, 2011 at 2:15 p.m.)", and the projections were
released as the conference opened. The archive's own minute for those statements
is 12:35 or 12:40 — when the release was *posted*, not when it was released —
which is why the early rule is an absolute time of day and not an offset.

**Splitting the transcript** is the part most likely to rot, so the marker style
is recorded per day. The running header (`Page 3 of 26`, or a bare `3 of 26` in
2011, plus the dated "… Press Conference FINAL/PRELIMINARY" line) is stripped,
and the opening remarks end at the first speaker who is not the Chair. Counted
over all 95 transcripts, that speaker is marked three ways and all three are
handled and tested: **`QUESTION.`** — exactly once, on 2011-04-27, and never
again; **a reporter's name in capitals** (`JON HILSENRATH.`) from 2011-06-22,
47 times; and **a press officer handing over** (`MICHELLE SMITH.  Steve.`) from
2020-04-29, 47 times. Two smaller things had to be found the same way: the 2018
transcripts punctuate the Chair's own marker with a **colon**, and the June 2024
one leaves **one space** after the stop rather than two, either of which makes a
parser start the "opening remarks" somewhere in the middle of the Q&A. With both
handled the remarks run 2,400 to 14,100 characters, median 7,700, on all 95.

Each half is capped at 9,000 characters for the state. On the remarks that is
almost always the whole thing; on the Q&A, whose median is 41,300, it is a real
cut, and what the reader gets is the **opening exchanges** — where the questions
about the path get asked — rather than the hour.

**The tree** is a third mode (`presser`, version `pc1`, its own cache tag, so
the statement reader's `v1` answers are untouched). Its state is the statement,
the dots sentences for the day, the opening remarks and the Q&A. Round one
carries the whole absolute tree and adds seven questions:

| id | type | asks |
|---|---|---|
| `remarks_vs_statement` | Choice | more hawkish / consistent / more dovish **than the statement** |
| `qa_vs_remarks` | Choice | the same, for the answers against the Chair's own remarks |
| `pushback_on_pricing` | Noul | did the Chair push back against how the market was pricing the path |
| `presser_stance` | Choice | hawkish / dovish / neutral for the dollar, the conference as a whole |
| `new_information` | Noul | relative to the statement |
| `surprise_size` | Score | nothing the statement did not have → the kind that sets the day |
| `dominant_topic` | Choice | inflation / labor / growth / financial conditions / balance sheet / path of rates / other |

Round two runs on a decisive `presser_stance` **or** on either half not being
*consistent* — that second door is the whole point of the mode — and asks the
direction again reversed ("a desk that had read the statement and the dots and
then listened to this: which side for the next hour?"), the holder check and the
horizon. **The sign comes from `presser_stance` and never from the statement's
`stance`**, which is what lets a 2022-11-02 point the other way. Strength is the
context tree's arithmetic: `p(stance) × confirm × surprise_size × (1 −
priced_in)`.

**Four arms on press-conference days**, all on EURUSD ticks and all entered from
the moment the Chair started rather than from the release: **`presser-reader`**,
the `pc1` tree; **`statement-reader`**, that day's cached absolute reading of the
statement carried into the conference — "what if you held the statement's read
through it"; **`dots-rule`** from the same moment, which asks whether the dots
move is still going half an hour on; and **`all pressers`**, the keyword bot
pointed at the transcript. Each against the same session-matched null, with a
latency sweep on the reader.

**And a reversal table.** Every day where `remarks_vs_statement` or
`qa_vs_remarks` was *not* consistent, printed with the tape's move from the
release to the Chair's first word and from there to an hour later, side by side.
That is the shape of 2022-11-02, and a table is the only way to find out whether
that day was one of a kind or one of twenty.

#### The run (real model, 95 press conferences, ticks)

Run on 2026-09-20. All 95 transcripts since April 2011 fetched and split
(median remarks 7,698 characters, median Q&A 41,308, of which the reader saw
the first 9,000); 85 went to round two, 33 questions in the median conference,
**median 1,029 ms per conference, $0.056** for the lot, plus $0.024 for the 95
statement readings this mode asked again. Entry on
the first EURUSD tick after the Chair's first word plus one second.

```
arm                    signals traded    pre   rush  sprd      +1m      +5m     +15m     +30m     +60m  hit15    z15
presser-reader              24     24     +4     +0   0.4       +1       -1       -3       -3       -3    42%   -0.6
  s.e.                                                         +-1      +-2      +-5      +-6      +-8
statement-reader            52     52     +1     +0   0.4       +0       +2      -11      -11       -9    31%   -3.5
  s.e.                                                         +-1      +-2      +-3      +-4      +-5
dots-rule                   37     37     -1     +0   0.3       +1       +1       -5       -4       -1    41%   -0.8
  s.e.                                                         +-1      +-2      +-4      +-5      +-6
all pressers                95     95     +1     +0   0.4       -0       -0       -7       -8       -9    38%   -2.8
  s.e.                                                         +-0      +-1      +-2      +-3      +-3

how they read for the dollar: dovish 52, hawkish 38, neutral 5
what they were about: path of rates 48, inflation 21, balance sheet 8, financial conditions 6
where the conference did not say what the statement said: 12 of 95
```

- **The reader read the press conference and got nothing from it.** 24 trades,
  −3 ± 5 bp at fifteen minutes, ten of twenty-four right, flat at every latency
  from zero to five minutes. A transcript is the slowest possible feed for a
  spoken event, and the reader saw the first fifth of the Q&A; both are
  reasons, neither is an excuse. The number is nothing.
- **Holding the statement's read into the conference loses, and loses more
  than chance.** The absolute reading of the day's statement, carried from
  14:30, comes out at **−11 ± 3 bp at fifteen minutes, 16 of 52 right** (sign
  test p = 0.008), −11 ± 4 at thirty, −9 ± 5 at sixty. Split by era it is
  −3 (47%) over the quarterly conferences of 2011–2018 and **−15 (21%) over the
  every-meeting conferences since 2019**. The keyword bot on the transcript
  itself loses too: −7 ± 2 at fifteen minutes (p = 0.02), −9 ± 3 at sixty
  (p = 0.007), in both eras. The dots move from the release does not continue
  through the conference either (−5 ± 4). Whatever the text of the day says,
  the hour after the Chair starts talking tends to go the other way. Fading
  the statement's read at 14:30 would have made +11 bp at fifteen minutes on
  these 52 days; that is the mirror of one arm on one pair, reported here as a
  pattern registered for the forward test and not as a trade.
- **The reversal table is short.** Twelve days of ninety-five where the reader
  said the remarks or the Q&A did not say what the statement said, 2024-12-18
  among them (statement to Chair −69 bp, Chair to an hour later −57). Nine of
  the twelve are from 2021 on. 2022-11-02 is not in it: the reader called that
  conference consistent with its statement, which is the day the mode was
  built for, and it missed it.

For the question that started this section: the words of the press conference
are not where the direction is. The direction is in the reversal of whatever
the statement's words said — a pattern, not a reading — and the thing a reader
can still do here is what it did on the statements: read the facts right and
abstain.

### The wire: every headline a retail scalper sees

Everything above is graded on **one issuer's scheduled text**, because the Fed
archive was the only free source with minute timestamps and depth. That is not
what a scalper reads. A scalper reads a wire: data prints from every country,
every central bank's speakers, intervention talk, tariffs and politics,
geopolitics, order-flow notes — eighty to five hundred and sixty items a week,
all day. Asking whether reading central-bank text is worth two basis points
answers a question about Fed statements, not a question about FX.

**The source.** [investinglive.com](https://investinglive.com), formerly
ForexLive, is the retail FX wire, and it publishes its whole archive as
sitemaps. Probed 2026-09-21:

| thing | what it is |
|---|---|
| `articles-sitemap-index.xml` | 946 weekly sitemaps, `/sitemaps/news/articles/{YYYY}-W{WW}.xml`, 2008-W33 to now |
| a weekly sitemap | `<url><loc>…</loc><lastmod>2017-08-28T00:28:59+00:00</lastmod></url>` |
| an article page | `<script type="application/ld+json">` with `@type: NewsArticle` |
| its fields | `headline`, `alternativeHeadline`, `datePublished`, `dateModified`, `articleSection`, `keywords`, `genre`, `articleBody`, `text` |
| `datePublished` | `2025-04-06T23:51:08.4325180Z` — UTC, to the second |
| 2025 alone | 20,898 articles, 81 to 560 a week |

`www.forexlive.com/sitemap.xml` redirects to the same index. URLs come in two
shapes — `/news/!/japan-monetary-base-…-20170903` on the old site,
`/news/…-20250406/` and `/central-banks/…`, `/commodities/…`, `/forex/…` on the
new one — and both are just links to follow. `datetime.fromisoformat` refuses
that timestamp (seven fractional digits), so `wire.py` parses it out rather
than borrowing.

That table is from a probe made **before** the `robots.txt` check below came
back no. Nothing in it has been re-checked since, and nothing in it will be
from here: the parsers are written to what it says, tested against fixtures,
and left. The one line anybody should re-run is the robots check itself.

#### The robots check came back no, and nothing was scraped

`fx/wire.py` reads `robots.txt` before it fetches anything, and the answer
decided this section. The file has a wildcard group that allows the articles
and the sitemaps to any crawler — and then this:

```
User-agent: ClaudeBot
Disallow: /

User-agent: anthropic-ai
Disallow: /

User-agent: Claude-Web
Disallow: /
```

…along with `AI2Bot`, `Bytespider`, `CCBot`, `DeepSeek`, `Baiduspider`,
`Qwen-Agent`, `Amazonbot`, `meta-externalagent` and `Diffbot`. The site has
said, by name, that it does not want AI agents crawling it.

**So the 21,000-page collect was not run.** A bulk fetch driven by an agent,
whose output is fed to a model, is the thing those three lines are about;
sending a browser's `User-Agent` would evade the rule rather than satisfy it.
The check is therefore written against *the work being done* and not against
the header that would be sent: `wire.permission` takes the wildcard group **and**
the AI-agent groups, and every one of them has to say yes.

```
robots.txt: articles DISALLOWED for ClaudeBot (Disallow: /); sitemaps DISALLOWED
```

`collect` raises `WireForbidden` rather than fetching, the CLI prints that line
and exits 3, and a test pins the verdict to the real file so that a change on
the site's side shows up as a failing test rather than as a silent scrape. The
adapter is finished and exercised end to end offline; what it will not do is
send the requests.

There is one door and it does not go round the rule. If a cached corpus is
already on disk — filled by somebody the wire *does* permit — the run continues
in an **offline mode that fetches nothing at all**: cached weeks and cached
articles are read, a link the cache has never held is counted in `uncached` and
skipped, and no request reaches the host. A run that sends no requests needs no
permission; a run that would send them does not get to borrow one.

#### The judge: 1-minute candles, and the weekend that is not empty

The tick judge is right and too expensive here: twenty-one thousand events on
seven pairs, each needing a quarter of an hour before and an hour after, is
dozens of hour-files apiece. The same feed publishes one file per pair per day
per side of **1-minute candles** — 1,440 records, ~12 kB compressed — so a year
of seven pairs both sides is 5,110 files instead of hundreds of thousands.

```
https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/BID_candles_min_1.bi5
                                                                 ASK_candles_min_1.bi5
```

`MM` is **zero-based**, exactly like the tick files, so the April 2025 file is
`2025/03/02`. LZMA; decompressed it is consecutive 24-byte big-endian records,
`struct.unpack(">IIIIIf")` = *(seconds since 00:00 UTC, open, close, low, high,
volume)*, with prices scaled by 1e5 or by 1e3 for JPY. Verified on EURUSD
2025-04-02 BID, whose first record is `(0, 107947, 107931, 107931, 107950,
90.6)` → open 1.07947, and on USDJPY ASK → 149.757.

**One thing about the format had to be found rather than assumed, and it is the
kind that corrupts a study silently.** The hourly tick files answer an empty
body for a closed hour. **The daily candle files do not: they pad.** Checked on
2025-04-05, EURUSD, NZDUSD and USDCHF: the Saturday file is 1,440 records at
Friday's closing price with **volume zero on both sides**. Friday's own file
carries volume to 20:59 UTC and pads 21:00–23:59; Sunday pads until 21:00 and
then has 180 real minutes. Taken at face value, a weekend "trade" returns a
guaranteed 0 bp — which would quietly flatter every null — and a Friday-evening
+60m becomes a 51-hour return in disguise. So **a zero-volume minute is treated
as no minute**: a bar the aggregated feed saw no trade in is not a price anyone
could have transacted at, and dropping it reproduces the tick tape's weekend
behaviour exactly. On real 2025 data that leaves Friday with 1,260 minutes,
Saturday with none and Sunday with 180, and a Friday 20:30 signal at +60m comes
back `None`. Over the whole year on EURUSD it leaves **372,088 tradeable
minutes of the 525,600 records the feed serves**, with 52 days carrying none —
so **29% of what those files contain is padding**, and a study that took them
at face value would have graded almost a third of its nulls on it.

**The entry rule, stated so the cost is visible.** A signal enters at the
**open of the first bar whose start is at or after `posted + latency`**, pays
the **ASK open to go long** and is filled at the **BID open to go short**, and
exits at the **mid close** of the bar `h` minutes later. The latency therefore
always rounds *up* to the next minute boundary, which means **0 s and 60 s are
the same fill** except for a post stamped exactly on a boundary. The sweep is
0 / 60 / 300 s for that reason and says so in its caption; only the five-minute
row can honestly differ, and it is the question in the form somebody would ask
it — *can a person who reads the headline and clicks still catch this?*

Seven pairs: `EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD`.

#### The lateness caveat, which is the scope of the whole section

This wire runs **seconds to minutes behind Reuters and Bloomberg**. Grading
from its post time measures what a reader *of this wire* could have done, not
what the event was worth. That is not a disclaimer to be taken on trust: every
table carries a **`pre`** column — how far the pair moved in the fifteen minutes
before the post, signed by the side the arm took — and that column *is* the
lateness, measured. A cell with a large positive `pre` and a small forward
return is the wire reporting a move that had already happened.

#### The tree (`w1`)

Its own version and its own cache tag, so no answer from `v1`, `ctx1` or `pc1`
is ever reused for it. The state is the headline, the published time in UTC and
in New York, London and Tokyo local, the section, the keywords and a body
excerpt (`--body-chars`, default 3,000).

**Round one, nine questions, one request, all speculative** — asked of every
post including the ones that are obviously not about FX, because width is free
and because the distribution of the answers is half the deliverable:

| id | type | asks |
|---|---|---|
| `about_fx` | Noul | is this about a currency market at all — not crypto, not equities, not a chart level |
| `currency` | Choice | `USD EUR JPY GBP AUD CAD CHF NZD CNY other none` — the one it most bears on |
| `direction` | Choice | stronger / weaker / none, for that currency |
| `category` | Choice | data release / central-bank decision / central-bank speaker / politics or trade / geopolitics / FX official or intervention / commentary or technical / order flow or positioning / other |
| `magnitude` | Score | the same four levels as the other modes |
| `new_information` | Noul | news, or a recap of something already out |
| `scheduled` | Noul | on the calendar, or not |
| `already_moved` | Noul | does the text itself say the market has already reacted |
| `is_number` | Noul | a number against a forecast, rather than words |

**Round two only if `about_fx` and `direction` are decisive** — which on a wire
is most of the filter, since a day's eighty posts are mostly recaps, chart
levels and crypto, and a second request on each of those would double the bill
for nothing. It asks the reversed framing — *"a trader reading this wire at this
moment: which side of {pair} for the next hour?"* — plus the holder check and the
horizon. The side is asked about the **pair** and not the currency, so the model
has to get from "the yen strengthens" to "short USDJPY" without being handed
the mapping.

The sign comes from `currency` + `direction` through one table with its own
test, because getting it backwards inverts the study while leaving every number
plausible:

| answer | pair | side |
|---|---|---|
| USD stronger | `EURUSD` | short |
| EUR stronger | `EURUSD` | long |
| JPY stronger | `USDJPY` | short |
| GBP stronger | `GBPUSD` | long |
| AUD stronger | `AUDUSD` | long |
| CAD stronger | `USDCAD` | short |
| CHF stronger | `USDCHF` | short |
| NZD stronger | `NZDUSD` | long |
| CNY / other / none | — | no trade |

Strength is arithmetic and the model multiplies nothing:
`p(direction) × confirm × magnitude × new_information × (1 − already_moved)`.
Unlike the absolute tree, `already_moved` is a **full** complement rather than a
half damper — on a wire that runs behind the primary feeds, "the text says the
market has already reacted" is a complete reason not to trade.

Offline, `--provider mock` answers the same tree with `wire.py`'s lexicon — the
*same* lexicon the `keyword-bot` arm uses — so the offline reader arm is that
bot wearing the tree's clothes and should score like it. The gap between that
and the real model on the same corpus is the result.

#### The arms, and the breakdowns that are the point

* **`reader >= thr`** — the `w1` tree at a strength threshold.
* **`keyword-bot`** — the incumbent: a currency is named or its central bank is,
  and hawkish/hike/beat/strong against dovish/cut/miss/weak sets the sign;
  nothing otherwise. It cannot read: *"the RBA will not hike"* counts as a hike.
* **`wire-sample`** — a seeded random sample (`--sample`, default 3,000) with the
  keyword sign where it has one. This is the base rate — *did a post happen* —
  without paying to grade all twenty-one thousand.
* **`null`** — per arm, the same pair and side at random moments within five days
  where the tape has bars, one per outcome.

**One mean over twenty-one thousand posts is a number about the wire's mixture,
not about anything tradeable.** So the deliverable is the split. Every
breakdown carries n, hit rate, mean and standard error at +5 / +15 / +60, and
the mean `pre`:

* per `category` — which kind of post moves the tape;
* per `scheduled` and per `is_number` — whether the thesis (numbers priced fast,
  words slow) survives outside the Fed;
* per currency and per pair;
* per session — Asia 00–07, London 07–13, New York 13–21, late 21–24, UTC;
* and **the reader's abstention rate per category**, which on a feed that is
  mostly recaps is a result in itself.

Plus the latency sweep at 0 / 60 / 300 s (with the note about minute bars
above) and the usual list of posts where counting words and reading them traded
differently, with the tape's own verdict next to both.

```bash
# scrape and cache the articles, print the summary, stop  (refused: see above)
python -m jevtrade.cli fx --wire --since 2025-01-01 --until 2025-12-31 --collect-only

# pre-fetch the seven pairs' BID/ASK day files for the window, stop
python -m jevtrade.cli fx --wire --warm-candles --since 2025-01-01 --until 2025-12-31

# the run
python -m jevtrade.cli fx --wire --provider jev --since 2025-01-01 --until 2025-12-31 \
    --horizons 1,5,15,30,60 --latency 1 --threshold 0.15 --sample 3000 \
    --body-chars 3000 --out runs/fx-wire-2025.json
```

Both halves are resumable: articles are cached **as the extracted fields and
never as the HTML** (21k pages of markup is about a gigabyte to keep a few
hundred characters of each), day-files as the compressed bytes they arrived as,
readings per post and tree version with the body budget in the key. An
interrupted run re-reads what it has and fetches only what it never reached.

**What a year would cost.** About 21,000 posts at roughly 700 input tokens each,
times ~1.3 rounds — round two runs only on the decisive ones — is on the order
of **$0.8 of input tokens**.

**The candle side has been done.** 2025 is 5,110 day-files and all 5,110 are on
disk: **53 minutes for 5,098 of them** with two keep-alive connections, then two
more passes of two minutes and twelve seconds for the twelve the feed refused
with a 503 or a read timeout — which is the resumability working, since a
transient failure is reported and never cached. The feed's throughput swung by
more than an order of magnitude inside that hour (0.1 to 2.5 files a second
with the same two connections), so the right posture towards it is a background
run and a second pass, not a longer timeout. 893 MB of compressed bytes on
disk, and none of it needs fetching again.

```
5110/5110 day-files on disk, 0 unreachable
  EURUSD 730/730   USDJPY 730/730   GBPUSD 730/730   AUDUSD 730/730
  USDCAD 730/730   USDCHF 730/730   NZDUSD 730/730
```

**The run on the wire itself has not been done, and will not be from here: its
robots.txt names AI agents and disallows them, so no article was fetched.** The
judge is warmed and the whole scoring path has been exercised against it — a
synthetic corpus of sixty posts over the real April 2025 candles measures,
places its nulls and fills every breakdown, at an average entry spread of 0.5
to 0.8 bp. What is missing is the corpus, and it is missing on purpose: it
cannot be collected from here without doing the thing the wire asked nobody to
do. So what is finished is the adapter, the judge, the tree, the arms and the
tests — and the honest note that the posts are waiting on permission rather
than on code.

#### A corpus the user brought: Truth Social, 2025

The paragraph above is the honest end of the wire story, and it is not the end
of the section, because **nothing downstream of `collect` cares where the posts
came from**. The `w1` tree, the minute-candle judge, the arms and the
breakdowns need `Article`s with a timestamp and nothing else. So a corpus
somebody collected under their own licence and handed over as a file drops
straight in, and `fx/posts.py` is that door.

The first one through it is the **Truth Social archive for 2025**, scraped from
trumpstruth.org by the user. It arrives as two JSON Lines captures of the same
thing plus a CSV of the first, and all three are read:

| shape | keys that matter |
|---|---|
| *posts* | `status_id`, `text`, `date_published` (ISO, UTC, to the second), `original_url`, `archive_url` |
| *statuses* | `status_id`, `article_body`, `headline` (`Donald J. Trump: "…"`), `date_published`, `url` |
| CSV | the posts shape, **with a BOM on the first header** — read `utf-8-sig` or the first column is named `﻿post_id` and every lookup of `post_id` silently misses |

**Three things about the files were counted rather than taken on trust, and
each of them contradicts what the import was specified from.** The posts file is
6,683 lines but only **6,120 distinct `status_id`** — 563 ids appear exactly
twice, with identical text, so the dedupe loses nothing. The statuses file is
6,120 lines covering *exactly* the same ids: it is a deduplicated second
capture, not extra coverage. And it **cannot fill a single missing text** —
both captures are missing the same 1,305 posts, the media-only ones and the
reposts, so the fill rule is implemented (it is the right rule for two captures
of one archive) and reports that it fired **zero** times rather than the code
pretending otherwise.

What is left is **4,815 posts with text**, 2025-01-01 to 2026-01-10, median 154
characters. A post with no text is dropped rather than read as an empty
document: a reader handed nothing will still answer something, and that answer
would be about the reader.

```
4815 posts with text from 12,803 lines in 2 file(s): 6120 distinct ids
(6683 rows merged into an id already seen), 0 texts filled from a second capture,
1305 dropped for having no text, 0 for having no timestamp, 0 unreadable rows;
median 154 characters
  per month: 2025-01 358, 2025-02 386, 2025-03 539, 2025-04 342, 2025-05 386,
  2025-06 415, 2025-07 430, 2025-08 464, 2025-09 275, 2025-10 340, 2025-11 493,
  2025-12 386, 2026-01 1
```

**The files stay where they are.** They are read by path, never copied into this
repository and never committed; the import writes the extracted fields into the
cache under `wire:posts:{status_id}`, which is deliberately *not* the wire's own
`wire:article:{url}` namespace — a cache filled from a file must not be
indistinguishable from one filled from the network.

**`--posts` does not consult `robots.txt` at all**, and that is the point rather
than a shortcut. That file governs crawling a site; there is no site here, no
request is made, and reading a corpus somebody already has is not a request to
anybody. The run is offline end to end, and a test proves it by making any wire
fetch raise.

**The weekend is the caveat this corpus carries and the wire does not.** A wire
posts when the market is open because that is when there is anything to report.
These posts land whenever they were written: **1,062 of the 4,815 are at a
weekend in UTC**, and spot FX is shut from about Friday 21:00 to Sunday 21:00.
Those cannot be graded — not scored zero, not carried forward to Monday — so the
run prints what the tape *could not* price beside what it could, per arm:

```
what the tape could and could not price (spot FX shuts Fri ~21:00 - Sun ~21:00 UTC):
  reader >=0.15   274 signals, 211 the tape could price, 63 it could not
                  (46 posted at a weekend, 17 on a weekday — a holiday, a Friday
                  evening, or an hour with no bars)
```

"The tape was shut" and "the reader stayed out" are different findings and only
one of them is about the reader, so they are never added together, and a
weekday/weekend breakdown sits beside the session one.

```bash
python -m jevtrade.cli fx --wire --posts trumpstruth_posts.jsonl trumpstruth_statuses.jsonl \
    --since 2025-01-01 --until 2025-12-31 --provider jev --sample 1000 \
    --horizons 1,5,15,30,60 --latency 1 --out runs/fx-posts-2025.json

python -m jevtrade.cli fx --wire --posts trumpstruth_posts.csv --collect-only   # import and stop
```

Offline (`--provider mock`) the whole thing runs in **1 m 44 s** over 4,814
posts on the warmed 2025 candles, which is the shape of the real run without
the reading: 274 reader signals, 211 of them priced, 61 posts where counting
words and reading them traded differently. The offline reader is the keyword
bot in the tree's clothes and its numbers mean nothing; what the smoke
establishes is that the corpus, the judge and every table line up.

**The run (real model, 4,814 posts, minute bars).** Run on 2026-09-22 with
`--provider jev --sample 1000`: 4,814 posts read (median 154 characters, up to
3,000 in the state), **31 went to round two, median 415 ms per post, $0.352**
for the lot — about 1,765 input tokens a post, the nine-question state being
most of it. Graded on the seven pairs' 2025 minute bars, entry at the next
minute's open one second after the post, paying the ask or the bid.

```
arm                    signals traded    pre  sprd      +5m     +15m     +30m     +60m  hit15    z15
reader >=0.15                0      0     +0   0.0       +0       +0       +0       +0      -      -
keyword-bot                335    260     +1   0.5       -0       -0       -1       -1    43%   -0.8
  s.e.                                                  +-0      +-0      +-1      +-1
wire-sample                 62     51     +2   0.5       -0       -0       +0       +0    35%   +0.2

what the reader made of it: politics_or_trade 3386, other 1025, geopolitics 368,
  market commentary 18, data release 15, central-bank speaker 2
currency: none 4125, USD 636, CAD 14, CNY 11, GBP 7, EUR 5, JPY 4;  direction: none 4231, stronger 338, weaker 245
about_fx >= 0.5 on 43 posts (0.9%);  abstention at the trading threshold: 100% in every category
shut out by the tape: 1,061 of 4,814 posts were written at a weekend; 54 of the keyword bot's 335 signals fell there
```

- **The reader traded nothing, and it was right not to.** It read 583 posts as
  directional and 324 of those as being about a tradable currency, but called
  the text *about a currency market* on 43 posts in the year and gave nearly
  all of them a magnitude near the floor, so nothing reached the second round
  with anything left to confirm. The keyword bot, which has no such qualms,
  made 260 trades and came out at −0 ± 0 bp at fifteen minutes with a 43% hit
  rate; the random sample of the corpus did the same.
- **Without the gate the directions are still noise.** Rescoring the 265
  directional reads the tape could price with no second round and no threshold:
  −0.2 ± 0.5 bp at fifteen minutes, 49% right. Tariff posts (58): −0.4 ± 0.9,
  43%. Posts about the Fed, Powell or rates (50): −0.0 ± 0.9, 46%. "USD
  stronger" reads (168): −0.3 ± 0.5. The one subset that looks like anything,
  magnitude ≥ 0.3 (22 posts, +4 ± 2 at fifteen minutes, 59%), is one cell of a
  dozen looked at after the fact and is reported so nobody has to find it.
- **The posts do not move the tape.** The unsigned EURUSD move in the fifteen
  minutes after a directional post has a median of 2.8 bp; after four hundred
  random non-directional posts, 2.5. The `pre` column is +0.6 bp: the market
  was not moving into these posts either. Whatever moved the dollar on tariff
  days in 2025 — the Rose Garden event, the executive orders, the wire reports
  of them — it was not the moment a post appeared on this account, or the
  post came after the tape already knew.

So the first unscheduled, high-impact, minute-stamped text corpus this study
has had says the same thing the scheduled one did, from the other side: on a
stream where the words carry no direction, the reader's value was abstaining
4,814 times for thirty-five cents, and the word-counter's cost was 260 trades
that went nowhere.

### What this does not show

**Feed latency is the real bottleneck, and this study cannot measure it.** A
scheduled statement is pollable to the second — the FOMC page goes live at
14:00:00 ET and anyone can be on it — so for rate decisions and minutes the
published timestamp is close to honest. For everything unscheduled it is not. A
speech is "published" when a web team gets to it; a press-conference remark
reaches the tape through a wire headline seconds after it is spoken and hours
before anything appears in an RSS feed, and wire feeds are not free. So a
positive result on speeches here would say "this text was worth reading", not
"you could have traded it", and the honest scope is the scheduled-text subset.

Three more limits worth holding onto. Only the Fed has an archive, so the
statistical weight will land on one central bank and one pair. Spot FX has no
weekend, which costs sample and biases the surviving events towards weekday
sessions. And 5-minute bars are coarse for a reaction whose first leg is
measured in seconds, which is why `--tape dukascopy` exists; the bar path is
kept because it is the cheaper check, and because two judges disagreeing about
an arm is worth knowing.

**Weekend FX perps were probed and left as future work.** Crypto venues list
24/7 FX perpetuals, which would cover the events the spot tape sleeps through.
Over 11 weekends, Gate's `EURUSD_USDT` drift across the closed period explains
some of Monday's spot gap and not much of it: beta 0.36, R² 0.22, mean |gap| 8.2
bp against mean |drift| 11.7 bp, residual s.d. 9.7 bp. Funding is effectively
zero (Bitget's USDJPY perp printed −3.95 bp once in ten days; EURUSD was flat
zero throughout). That is enough of a link to be interesting and too loose to
grade a signal on, so nothing in `jevtrade/fx/` depends on it.

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
python -m pytest -q      # 472 tests
```

They cover the documented request/response schema, each policy gate, position
accounting through a flip, the latency and deadline behaviour, the stop and
kill switch, and two honesty checks on the simulator itself: no edge when
`alpha=0`, and an edge when it is switched on. Every feed parser is tested
against a fixture rather than the network, including the wire's `robots.txt`
itself — the verdict that section is gated on is pinned to the real file, so a
change on the site's side fails a test instead of quietly starting a scrape.

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
| `listing/announcements.py` | Binance's CMS feed, millisecond-stamped |
| `listing/tickers.py` | candidate tickers, found by code |
| `listing/reader.py` | the two-round judgment tree |
| `listing/baseline.py` | the title-matching sniper bot |
| `listing/venues.py` | 1-minute candles from Binance, OKX or Coinbase |
| `listing/study.py` | entry rule, horizons, matched null, arms |
| `fx/documents.py` | Fed archives + ECB/BoJ/BoE RSS, timestamps and bodies |
| `fx/diff.py` | the previous statement, and the sentences that changed |
| `fx/reader.py` | the two-round tree and the sign convention |
| `fx/baseline.py` | the word-counting bot and the rate-surprise bot |
| `fx/tape.py` | spot FX bars, gap-aware entry and horizons |
| `fx/ticks.py` | Dukascopy ticks: the book, the entry latency, the spread |
| `fx/http.py` | one kept-open TLS connection per worker, shared by every FX fetch |
| `fx/candles.py` | Dukascopy 1-minute candles: the cheap judge, and the padded weekend |
| `fx/wire.py` | investinglive.com: the robots gate, the sitemaps, the JSON-LD |
| `fx/posts.py` | a post archive the user brought, read into the same `Article` |
| `fx/context.py` | the pre-release context, every item stamped and checked |
| `fx/dots.py` | the SEP histogram, the median, the rule and the forward register |
| `fx/presser.py` | the press-conference transcript, split at the first question |
| `fx/study.py` | the arms, the session-matched null, the audit |
| `fx/wirestudy.py` | the wire's arms and the breakdowns that are its deliverable |
| `fx/mock.py` | offline stub for the FX questions |

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
