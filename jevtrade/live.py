"""Live top-of-book from Kraken's public API.

Unlike the OHLCV path in ``feed.py``, this gives a **real** order book: Kraken's
ticker carries the best bid and ask with their resting sizes. That matters,
because the whole design reads the book, and a book reconstructed from bar data
is a fabrication the model cannot know to distrust.

No API key is needed. Kraken's public endpoints allow roughly one call a second,
which is also the natural snapshot rate for this demo.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass

from .feed import KRAKEN_PAIRS
from .types import Tick

TICKER_URL = "https://api.kraken.com/0/public/Ticker?pair={pair}"
TRADES_URL = "https://api.kraken.com/0/public/Trades?pair={pair}"


def _get(url: str, timeout: float, attempts: int = 3) -> dict:
    """GET with retry. A demo that runs for an hour will meet a dropped
    connection; one bad poll must not end the session."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            if payload.get("error"):
                raise RuntimeError(f"Kraken error: {payload['error']}")
            return payload["result"]
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at the end
            last = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Kraken request failed after {attempts} attempts: {last}")


@dataclass
class LiveConfig:
    symbol: str = "BTC"
    interval_ms: int = 1000
    timeout_s: float = 25.0


class KrakenLiveFeed:
    """Polls the ticker, one :class:`Tick` per call."""

    def __init__(self, config: LiveConfig | None = None) -> None:
        self.config = config or LiveConfig()
        self.pair = KRAKEN_PAIRS.get(
            self.config.symbol.upper().replace("-", ""), self.config.symbol.upper()
        )
        self.symbol = f"{self.config.symbol.upper()}-USD"
        self.interval_ms = self.config.interval_ms
        self._seq = 0
        self._last_day_volume: float | None = None

    # ------------------------------------------------------------- priming

    def prime(self, max_ticks: int = 150) -> list[Tick]:
        """Build recent 1-second bars from the public trade tape.

        The feature window needs ~60 snapshots before it will emit anything.
        Waiting a minute for a demo to start is silly when the last 1000 trades
        are one request away, so bucket them into bars and warm up instantly.
        """
        result = _get(TRADES_URL.format(pair=self.pair), self.config.timeout_s)
        key = next(k for k in result if k != "last")
        trades = result[key]
        if not trades:
            return []

        bucket_ms = self.config.interval_ms
        buckets: dict[int, list] = defaultdict(list)
        for price, volume, ts, side, *_rest in trades:
            slot = int(float(ts) * 1000 // bucket_ms)
            buckets[slot].append((float(price), float(volume), side))

        ticks: list[Tick] = []
        for slot in sorted(buckets)[-max_ticks:]:
            rows = buckets[slot]
            price = rows[-1][0]
            volume = sum(v for _p, v, _s in rows)
            # Split traded volume by aggressor side. A proxy for resting size,
            # used only to warm up the rolling medians -- no trade is ever
            # placed against a primed tick.
            bought = sum(v for _p, v, s in rows if s == "b") or 1e-6
            sold = sum(v for _p, v, s in rows if s == "s") or 1e-6
            half = price * 0.00005 / 2  # ~0.5 bp placeholder touch
            ticks.append(
                Tick(
                    seq=self._seq,
                    ts=slot * bucket_ms / 1000.0,
                    symbol=self.symbol,
                    bid=price - half,
                    ask=price + half,
                    last=price,
                    volume=volume,
                    bid_size=bought,
                    ask_size=sold,
                )
            )
            self._seq += 1
        return ticks

    # ---------------------------------------------------------------- live

    def poll(self) -> Tick:
        """One real top-of-book snapshot."""
        result = _get(TICKER_URL.format(pair=self.pair), self.config.timeout_s)
        data = next(iter(result.values()))

        ask, _ask_whole, ask_size = data["a"]
        bid, _bid_whole, bid_size = data["b"]
        last_price = float(data["c"][0])
        day_volume = float(data["v"][0])

        # Volume traded since the previous poll. The daily counter resets at
        # UTC midnight, so a negative delta means the rollover happened.
        if self._last_day_volume is None or day_volume < self._last_day_volume:
            traded = 0.0
        else:
            traded = day_volume - self._last_day_volume
        self._last_day_volume = day_volume

        tick = Tick(
            seq=self._seq,
            ts=time.time(),
            symbol=self.symbol,
            bid=float(bid),
            ask=float(ask),
            last=last_price,
            volume=traded,
            bid_size=max(float(bid_size), 1e-6),
            ask_size=max(float(ask_size), 1e-6),
        )
        self._seq += 1
        return tick
