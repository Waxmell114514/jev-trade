"""Label each headline by what the tape actually did afterwards.

Nobody's judgement enters here. For every headline we measure, in units of the
window's own volatility:

* ``post`` -- the largest move in the minutes *after* it printed
* ``pre``  -- the move in the minutes *before* it printed

Both matter. A headline followed by a big move looks material, but if the move
had already started before it printed, the story is *reporting* the move, not
causing it -- and reacting to it is too late by construction. Separating those
two is most of the point, because a market maker can only act on the first kind.

The obvious limitation: an RSS ``pubDate`` is when an article was published,
which trails the underlying event by an unknown amount. That biases towards
calling things reactive, so the "clean mover" count below is a lower bound.
"""

from __future__ import annotations

import json
import math
import statistics
import urllib.request
from dataclasses import dataclass

from .feeds import Headline

KRAKEN = "https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"


@dataclass(frozen=True)
class Bars:
    interval_min: int
    ts: list[float]
    close: list[float]
    sigma: float  # per-bar log-return standard deviation

    def index_at(self, when: float) -> int | None:
        if not self.ts or when < self.ts[0] or when > self.ts[-1]:
            return None
        lo, hi = 0, len(self.ts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.ts[mid] <= when:
                lo = mid
            else:
                hi = mid - 1
        return lo


def fetch_bars(pair: str = "XBTUSD", interval: int = 5, timeout: float = 30.0) -> Bars:
    url = KRAKEN.format(pair=pair, interval=interval)
    payload = json.loads(urllib.request.urlopen(url, timeout=timeout).read())
    result = payload["result"]
    key = next(k for k in result if k != "last")
    rows = result[key]

    ts = [float(r[0]) for r in rows]
    close = [float(r[4]) for r in rows]
    rets = [math.log(b / a) for a, b in zip(close, close[1:]) if a > 0 and b > 0]
    return Bars(
        interval_min=interval,
        ts=ts,
        close=close,
        sigma=statistics.pstdev(rets) if len(rets) > 1 else 0.0,
    )


@dataclass(frozen=True)
class Labelled:
    headline: Headline
    pre: float  # move before, in sigmas
    post: float  # largest move after, in sigmas
    post_signed: float  # signed move at the full horizon, in sigmas
    price: float

    def moved(self, threshold: float) -> bool:
        return self.post >= threshold

    def reactive(self, threshold: float) -> bool:
        """The move was already underway when the headline printed."""
        return self.pre >= threshold

    def clean_mover(self, threshold: float) -> bool:
        """Moved after, and was not already moving before. The actionable kind."""
        return self.moved(threshold) and not self.reactive(threshold)


def label(
    headlines: list[Headline], bars: Bars, horizon_bars: int = 3
) -> list[Labelled]:
    out: list[Labelled] = []
    if bars.sigma <= 0:
        return out

    for headline in headlines:
        i = bars.index_at(headline.ts)
        if i is None or i < horizon_bars or i + horizon_bars >= len(bars.close):
            continue
        base = bars.close[i]
        if base <= 0:
            continue

        def sigmas(a: float, b: float, steps: int) -> float:
            if a <= 0 or b <= 0:
                return 0.0
            return math.log(b / a) / (bars.sigma * math.sqrt(steps))

        # Largest excursion at any horizon up to the full window, so a move that
        # reverses inside the window still counts as a move.
        post = max(
            abs(sigmas(base, bars.close[i + h], h))
            for h in range(1, horizon_bars + 1)
        )
        pre = abs(sigmas(bars.close[i - horizon_bars], base, horizon_bars))
        signed = sigmas(base, bars.close[i + horizon_bars], horizon_bars)

        out.append(
            Labelled(headline=headline, pre=pre, post=post,
                     post_signed=signed, price=base)
        )
    return out
