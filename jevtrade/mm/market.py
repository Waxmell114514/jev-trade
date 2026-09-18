"""A market-making simulator with informed (toxic) flow.

Without toxic flow a market-making backtest is meaningless: quoting both sides
of a random walk always "wins", and every strategy looks brilliant. The whole
business is *spread captured minus adverse selection*, so the counterparty has
to sometimes know something.

Two kinds of counterparty arrive here:

* **Noise flow** — Poisson arrivals, random side. Its fill probability decays
  with how far our quote sits from the mid (``exp(-kappa * distance)``, the
  Avellaneda–Stoikov intensity). This is where the spread is earned, and
  quoting wider earns less of it.
* **Informed flow** — arrives in the window after a material event and trades
  the direction the tape is about to move. It crosses our quote only while the
  quote is cheap relative to the move it already knows about. This is where the
  money is lost.

That single asymmetry is what makes the posture decision worth anything:
widening sheds toxic fills faster than it sheds benign ones, and pulling sheds
both. Calibration note: a 2 bp half-spread is a mid-liquidity instrument, not
BTC/USD on a top venue, where the touch is a fraction of a basis point and no
taker strategy of this shape pays for itself.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .events import EventGenerator, NewsEvent


@dataclass
class MarketConfig:
    n_ticks: int = 6_000
    interval_ms: int = 250
    symbol: str = "BTC-USD spot"
    start_price: float = 77_000.0
    annual_vol: float = 0.70
    seed: int = 5

    # Noise flow
    noise_orders_per_tick: float = 0.55
    noise_kappa: float = 0.45  # fill intensity decay per bp of distance
    noise_size_mean: float = 1.0

    # Informed flow
    toxicity: float = 1.0  # 0 disables informed flow entirely
    informed_orders_per_tick: float = 0.9
    informed_window: int = 12  # ticks of informed arrivals after an event
    informed_patience: float = 0.55  # will pay up to this share of its edge

    # Events
    event_rate_per_1000: float = 12.0
    event_impact_bps: float = 22.0
    impact_ticks: int = 10

    @property
    def tick_vol_bps(self) -> float:
        per_year = 365 * 24 * 3600 / (self.interval_ms / 1000.0)
        return self.annual_vol / math.sqrt(per_year) * 1e4


@dataclass(frozen=True)
class Quote:
    """What we are showing. A side with zero size is not quoted."""

    bid: float
    ask: float
    bid_size: float
    ask_size: float

    @staticmethod
    def none() -> "Quote":
        return Quote(0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MMFill:
    seq: int
    side: str  # OUR side: "buy" means someone hit our bid
    price: float
    size: float
    mid_at_fill: float
    informed: bool

    @property
    def signed(self) -> float:
        return self.size if self.side == "buy" else -self.size

    @property
    def edge_bps(self) -> float:
        """Half-spread captured at the moment of the fill, in bps."""
        gain = (self.mid_at_fill - self.price) * (1 if self.side == "buy" else -1)
        return gain / self.mid_at_fill * 1e4


@dataclass
class Observation:
    seq: int
    mid: float
    vol_bps: float
    event: NewsEvent | None
    recent_flow_imbalance: float  # +1 all buys, -1 all sells, over the last window
    ticks_since_event: int | None


@dataclass
class MarketSim:
    config: MarketConfig = field(default_factory=MarketConfig)

    def __post_init__(self) -> None:
        cfg = self.config
        # Two independent streams. The price path and the event schedule must be
        # identical across strategies, or the arms are not comparable -- and
        # flow draws depend on our own quotes, so a shared stream would make
        # every arm see a different market.
        self.price_rng = random.Random(cfg.seed)
        self.flow_rng = random.Random(cfg.seed + 7919)
        self.events = EventGenerator(
            seed=cfg.seed + 101,
            rate_per_1000=cfg.event_rate_per_1000,
            impact_bps=cfg.event_impact_bps,
        )
        self.mid = cfg.start_price
        self.seq = -1
        self.mids: list[float] = []
        self._pending_drift: list[float] = [0.0] * (cfg.n_ticks + cfg.impact_ticks + 2)
        self._recent_sides: list[float] = []
        self._active_event: NewsEvent | None = None
        self._event_seq: int | None = None
        self._current_event: NewsEvent | None = None

    # ------------------------------------------------------------- observe

    def observe(self) -> Observation | None:
        """Advance to the next tick and report what is visible before quoting."""
        cfg = self.config
        self.seq += 1
        if self.seq >= cfg.n_ticks:
            return None

        if self.seq > 0:
            shock = self.price_rng.gauss(0.0, cfg.tick_vol_bps / 1e4)
            drift = self._pending_drift[self.seq] / 1e4
            self.mid *= math.exp(shock + drift)
        self.mids.append(self.mid)

        event = self.events.maybe(self.seq)
        self._current_event = event
        if event is not None:
            self._active_event = event
            self._event_seq = self.seq
            if event.impact_bps:
                # The move arrives as a ramp over impact_ticks, starting next tick.
                per = event.impact_bps / cfg.impact_ticks
                for k in range(1, cfg.impact_ticks + 1):
                    if self.seq + k < len(self._pending_drift):
                        self._pending_drift[self.seq + k] += per

        since = None
        if self._event_seq is not None:
            gap = self.seq - self._event_seq
            since = gap if gap <= cfg.informed_window else None
            if since is None:
                self._active_event = None
                self._event_seq = None

        imbalance = (
            sum(self._recent_sides) / len(self._recent_sides)
            if self._recent_sides
            else 0.0
        )
        return Observation(
            seq=self.seq,
            mid=self.mid,
            vol_bps=cfg.tick_vol_bps,
            event=event,
            recent_flow_imbalance=imbalance,
            ticks_since_event=since,
        )

    # ------------------------------------------------------------- advance

    def advance(self, quote: Quote) -> list[MMFill]:
        """Run one tick of order flow against our quote."""
        cfg = self.config
        fills: list[MMFill] = []

        # --- noise flow: fill probability decays with our distance from mid
        n_noise = self._poisson(cfg.noise_orders_per_tick)
        for _ in range(n_noise):
            buy_side = self.flow_rng.random() < 0.5  # the counterparty's side
            if buy_side and quote.ask_size > 0:
                dist = (quote.ask - self.mid) / self.mid * 1e4
                if self.flow_rng.random() < math.exp(-cfg.noise_kappa * max(dist, 0.0)):
                    fills.append(self._fill("sell", quote.ask, quote.ask_size, False))
            elif not buy_side and quote.bid_size > 0:
                dist = (self.mid - quote.bid) / self.mid * 1e4
                if self.flow_rng.random() < math.exp(-cfg.noise_kappa * max(dist, 0.0)):
                    fills.append(self._fill("buy", quote.bid, quote.bid_size, False))

        # --- informed flow: only while an event's move is still ahead of us
        event, gap = self._active_event, None
        if event is not None and self._event_seq is not None:
            gap = self.seq - self._event_seq
        if (
            event is not None
            and event.impact_bps
            and gap is not None
            and 0 <= gap <= cfg.informed_window
            and cfg.toxicity > 0
        ):
            remaining = abs(event.impact_bps) * max(
                0.0, 1.0 - gap / max(cfg.impact_ticks, 1)
            )
            n_inf = self._poisson(cfg.informed_orders_per_tick * cfg.toxicity)
            for _ in range(n_inf):
                if event.impact_bps > 0 and quote.ask_size > 0:
                    dist = (quote.ask - self.mid) / self.mid * 1e4
                    if dist <= remaining * cfg.informed_patience:
                        fills.append(
                            self._fill("sell", quote.ask, quote.ask_size, True)
                        )
                elif event.impact_bps < 0 and quote.bid_size > 0:
                    dist = (self.mid - quote.bid) / self.mid * 1e4
                    if dist <= remaining * cfg.informed_patience:
                        fills.append(self._fill("buy", quote.bid, quote.bid_size, True))

        for fill in fills:
            self._recent_sides.append(1.0 if fill.side == "sell" else -1.0)
        del self._recent_sides[:-40]
        return fills

    def _fill(self, side: str, price: float, size: float, informed: bool) -> MMFill:
        size = size * math.exp(self.flow_rng.gauss(0.0, 0.25)) * self.config.noise_size_mean
        return MMFill(
            seq=self.seq,
            side=side,
            price=price,
            size=max(size, 1e-6),
            mid_at_fill=self.mid,
            informed=informed,
        )

    def _poisson(self, rate: float) -> int:
        if rate <= 0:
            return 0
        # Knuth, fine at these rates.
        limit, k, p = math.exp(-rate), 0, 1.0
        while True:
            p *= self.flow_rng.random()
            if p <= limit:
                return k
            k += 1
