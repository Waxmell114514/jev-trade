"""Tick sources.

Three of them:

* :class:`SyntheticFeed` -- a seeded microstructure simulator. Reproducible,
  and it lets you dial the amount of genuine predictability in the tape, which
  is what makes the backtest interpretable (see README).
* :class:`CsvReplayFeed` -- replays a CSV written by ``jevtrade fetch``.
* :func:`fetch_kraken_ohlc` -- pulls real recent BTC/ETH bars from Kraken's
  public API.

Real 1-minute bars have no top-of-book in them, so the replay feed
*reconstructs* a plausible bid/ask and book sizes from the bar. That is an
approximation, not market data, and it is flagged as such everywhere it is used.
"""

from __future__ import annotations

import csv
import json
import math
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .types import Tick

KRAKEN_OHLC = "https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"

# Kraken's asset codes are idiosyncratic; map the names people actually type.
KRAKEN_PAIRS = {
    "BTC": "XBTUSD",
    "BTCUSD": "XBTUSD",
    "XBTUSD": "XBTUSD",
    "ETH": "ETHUSD",
    "ETHUSD": "ETHUSD",
    "SOL": "SOLUSD",
    "SOLUSD": "SOLUSD",
}

CSV_COLUMNS = [
    "seq",
    "ts",
    "symbol",
    "bid",
    "ask",
    "last",
    "volume",
    "bid_size",
    "ask_size",
]


class Feed(Iterable[Tick]):
    """Anything that yields ticks in time order."""

    symbol: str
    interval_ms: int

    def __iter__(self) -> Iterator[Tick]:  # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------------ synthetic


@dataclass
class SyntheticConfig:
    symbol: str = "BTC-USD"
    start_price: float = 76_000.0
    n_ticks: int = 3_000
    interval_ms: int = 250
    seed: int = 7

    # Per-tick volatility is derived from an annualised figure so that changing
    # the snapshot interval stays self-consistent. ~55% a year is ordinary for
    # BTC. Set base_vol to override it directly.
    annual_vol: float = 0.55
    base_vol: float | None = None

    # GARCH(1,1)-ish volatility clustering.
    garch_alpha: float = 0.08
    garch_beta: float = 0.90

    # Strength of the one genuinely predictable component in the tape.
    # 0.0 => an efficient market: no strategy can beat zero before costs.
    alpha: float = 0.35
    alpha_persistence: float = 0.93

    # Microstructure. A quarter of a basis point is about $1.90 on a $76k BTC,
    # which is the right order of magnitude for a major venue's touch.
    base_spread_bps: float = 0.25
    jump_prob: float = 0.004
    jump_size_vols: float = 6.0

    @property
    def tick_vol(self) -> float:
        """Standard deviation of one tick's log return."""
        if self.base_vol is not None:
            return self.base_vol
        ticks_per_year = 365 * 24 * 3600 / (self.interval_ms / 1000.0)
        return self.annual_vol / math.sqrt(ticks_per_year)


class SyntheticFeed(Feed):
    """A seeded order-book simulator with a known amount of real edge.

    The hidden state ``s`` is an AR(1) process. It biases the *next* return and
    it leaks into the *observable* book imbalance and volume. So book imbalance
    genuinely predicts the next move, by a controllable amount -- a strategy
    that reads the book should beat one that doesn't, and with ``alpha=0``
    nothing should beat zero. That property is what makes this harness useful
    for judging a decision model rather than just producing a pretty curve.
    """

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.symbol = self.config.symbol
        self.interval_ms = self.config.interval_ms

    def __iter__(self) -> Iterator[Tick]:
        cfg = self.config
        rng = random.Random(cfg.seed)

        base_vol = cfg.tick_vol
        price = cfg.start_price
        var = base_vol**2
        prev_ret = 0.0
        hidden = 0.0
        ts = 1_700_000_000.0

        for seq in range(cfg.n_ticks):
            # --- volatility clustering
            var = (
                base_vol**2 * (1 - cfg.garch_alpha - cfg.garch_beta)
                + cfg.garch_alpha * prev_ret**2
                + cfg.garch_beta * var
            )
            vol = math.sqrt(max(var, 1e-12))

            # --- hidden predictable component, AR(1)
            hidden = cfg.alpha_persistence * hidden + rng.gauss(
                0.0, math.sqrt(1 - cfg.alpha_persistence**2)
            )

            # --- return: predictable drift + noise + occasional jump
            ret = cfg.alpha * hidden * vol + rng.gauss(0.0, vol)
            if rng.random() < cfg.jump_prob:
                ret += rng.choice((-1.0, 1.0)) * cfg.jump_size_vols * vol

            price *= math.exp(ret)
            prev_ret = ret

            # --- observables that partially reveal `hidden`
            vol_ratio = vol / base_vol
            imbalance = math.tanh(0.8 * hidden + rng.gauss(0.0, 0.7))
            spread = (
                cfg.base_spread_bps
                * (0.6 + 0.6 * vol_ratio)
                * (1.0 + 0.5 * rng.random())
            )
            half = price * spread / 2e4

            base_size = math.exp(rng.gauss(0.7, 0.5))
            bid_size = base_size * (1.0 + imbalance)
            ask_size = base_size * (1.0 - imbalance)

            traded = math.exp(
                rng.gauss(0.0, 0.7)
            ) * (0.35 + 1.6 * vol_ratio + 2.0 * abs(ret) / base_vol * 0.05)

            yield Tick(
                seq=seq,
                ts=ts,
                symbol=cfg.symbol,
                bid=price - half,
                ask=price + half,
                last=price,
                volume=round(traded, 8),
                bid_size=round(max(bid_size, 1e-4), 6),
                ask_size=round(max(ask_size, 1e-4), 6),
            )
            ts += cfg.interval_ms / 1000.0


# --------------------------------------------------------------- CSV / replay


class CsvReplayFeed(Feed):
    """Replays ticks from a CSV written by :func:`write_csv`."""

    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        self.path = Path(path)
        self.limit = limit
        rows = list(self._read())
        if not rows:
            raise ValueError(f"{self.path} contains no ticks")
        self._rows = rows
        self.symbol = rows[0].symbol
        self.interval_ms = (
            int(round((rows[1].ts - rows[0].ts) * 1000)) if len(rows) > 1 else 1000
        )

    def _read(self) -> Iterator[Tick]:
        with self.path.open(newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                if self.limit is not None and i >= self.limit:
                    break
                yield Tick(
                    seq=int(row["seq"]),
                    ts=float(row["ts"]),
                    symbol=row["symbol"],
                    bid=float(row["bid"]),
                    ask=float(row["ask"]),
                    last=float(row["last"]),
                    volume=float(row["volume"]),
                    bid_size=float(row["bid_size"]),
                    ask_size=float(row["ask_size"]),
                )

    def __iter__(self) -> Iterator[Tick]:
        return iter(self._rows)


def write_csv(path: str | Path, ticks: Iterable[Tick]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for tick in ticks:
            writer.writerow(
                [
                    tick.seq,
                    f"{tick.ts:.3f}",
                    tick.symbol,
                    f"{tick.bid:.6f}",
                    f"{tick.ask:.6f}",
                    f"{tick.last:.6f}",
                    f"{tick.volume:.8f}",
                    f"{tick.bid_size:.6f}",
                    f"{tick.ask_size:.6f}",
                ]
            )
            count += 1
    return count


# ------------------------------------------------------------- live (Kraken)


def fetch_kraken_ohlc(
    symbol: str = "BTC", interval_min: int = 1, timeout: float = 20.0
) -> list[Tick]:
    """Fetch recent OHLCV bars from Kraken's public API (no key required).

    Returns up to 720 bars. Kraken gives ``[time, open, high, low, close, vwap,
    volume, trades]``; there is no book in that, so bid/ask and sizes are
    *reconstructed*: the spread is estimated from the bar's high-low range and
    the imbalance from where the close sits inside the bar. Good enough to
    exercise the pipeline on real prices and volumes, not a substitute for L1.
    """
    pair = KRAKEN_PAIRS.get(symbol.upper().replace("-", ""), symbol.upper())
    url = KRAKEN_OHLC.format(pair=pair, interval=interval_min)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode())

    if payload.get("error"):
        raise RuntimeError(f"Kraken error for {pair}: {payload['error']}")
    result = payload["result"]
    key = next(k for k in result if k != "last")
    bars = result[key]

    ticks: list[Tick] = []
    for seq, bar in enumerate(bars):
        ts, _open, high, low, close, _vwap, volume, trades = bar
        high, low, close = float(high), float(low), float(close)
        volume = float(volume)

        rng_bps = (high - low) / close * 1e4 if close else 0.0
        # Wider bars imply a wider touch; floor it at a realistic 1bp.
        spread_bps = max(1.0, min(rng_bps * 0.10, 25.0))
        half = close * spread_bps / 2e4

        # Close near the bar high => buyers were lifting offers => bid-heavy.
        span = max(high - low, 1e-9)
        imbalance = max(-0.9, min(0.9, (close - (high + low) / 2) / span * 2))
        size = max(volume, 1e-6) / max(int(trades) or 1, 1) * 4

        ticks.append(
            Tick(
                seq=seq,
                ts=float(ts),
                symbol=f"{symbol.upper()}-USD",
                bid=close - half,
                ask=close + half,
                last=close,
                volume=volume,
                bid_size=max(size * (1 + imbalance), 1e-6),
                ask_size=max(size * (1 - imbalance), 1e-6),
            )
        )
    return ticks
