"""Spot FX bars, and the entry rule that grades every arm the same way.

Spot FX is not a 24/7 tape and the measurement has to know it. There are no bars
from roughly Friday 21:00 UTC to Sunday 21:00 UTC, and Yahoo's 5-minute series
carries occasional null prints inside the week. A horizon computed by bar
*index* across a weekend would claim to measure sixty minutes and actually
measure fifty-one hours, which is how a study accidentally discovers that
central banks move markets. So every horizon here is checked against the bar's
own timestamp, and a signal whose window is not covered by contiguous bars is
dropped rather than estimated.

The listing study's ``forward_returns`` is not reused: it assumes a continuous
one-minute grid with gaps forward-filled, which is right for a crypto pair and
wrong for this tape.

Entry is the open of the first bar *strictly after* the timestamp. On 5-minute
bars that is up to five minutes late, deliberately: a reader that answers in 400
ms is not credited with the move it was fast enough to catch.
"""

from __future__ import annotations

import bisect
import math
import sys
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

from ..listing.announcements import get_json
from ..listing.store import Store

CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?interval={interval}&range={range}"
)
HORIZONS = (5, 15, 30, 60)
PRE_MIN = 15
DEFAULT_BAR_MIN = 5

Raw = list[tuple[int, float, float]]  # (bar open time, open, close)


@dataclass(frozen=True)
class Bars:
    """Real bars only -- no grid, no forward fill, gaps left as gaps."""

    symbol: str
    bar_min: int
    ts: list[int]
    open: list[float]
    close: list[float]

    def __len__(self) -> int:
        return len(self.ts)

    def index_at(self, when: float) -> int | None:
        """The bar that contains ``when``, or None if that moment has no bar."""
        i = bisect.bisect_right(self.ts, when) - 1
        if i < 0 or when - self.ts[i] >= self.bar_min * 60:
            return None
        return i

    def index_after(self, when: float) -> int | None:
        """The first bar that opens strictly after ``when``."""
        i = bisect.bisect_right(self.ts, when)
        return i if i < len(self.ts) else None


# ------------------------------------------------------------------ fetching


def fetch_chart(symbol: str, interval: str = "5m", range_: str = "60d") -> Raw:
    payload = get_json(CHART_URL.format(symbol=symbol, interval=interval, range=range_))
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not result:
        return []
    stamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens, closes = quote.get("open") or [], quote.get("close") or []
    out: Raw = []
    for i, ts in enumerate(stamps):
        o = opens[i] if i < len(opens) else None
        c = closes[i] if i < len(closes) else None
        if o is None or c is None or not (o > 0 and c > 0):
            continue  # Yahoo prints nulls in thin minutes; a null is not a price
        out.append((int(ts), float(o), float(c)))
    out.sort()
    return out


def bars(
    store: Store,
    symbol: str,
    *,
    bar_min: int = DEFAULT_BAR_MIN,
    days: int = 60,
    fetcher: Callable[[str, str, str], Raw] | None = None,
) -> Bars | None:
    """The whole series for a symbol, cached per symbol/interval/day.

    One request covers every signal on that pair, which is why this is not a
    per-event window fetch: 60 days of 5-minute bars is one 1.6 MB payload and
    17k bars, and the study asks it about a few dozen moments.
    """
    fetcher = fetcher or fetch_chart
    interval = f"{bar_min}m"
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"chart:{symbol}:{interval}:{days}d:{day}"
    found, raw = store.get(key)
    if not found:
        try:
            raw = fetcher(symbol, interval, f"{days}d")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            print(f"  {symbol} {interval}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None
        store.put(key, raw)
    rows = [tuple(r) for r in raw or []]
    if len(rows) < 3:
        return None
    return Bars(
        symbol=symbol,
        bar_min=bar_min,
        ts=[int(r[0]) for r in rows],
        open=[float(r[1]) for r in rows],
        close=[float(r[2]) for r in rows],
    )


# --------------------------------------------------------------- measurement


def _log_bps(a: float, b: float) -> float | None:
    if not (a > 0 and b > 0) or math.isnan(a) or math.isnan(b):
        return None
    return math.log(b / a) * 1e4


def forward_returns(
    series: Bars,
    when: float,
    sign: int,
    horizons_min: Iterable[int] = HORIZONS,
    bar_min: int | None = None,
) -> tuple[float, float, dict[int, float]] | None:
    """``(pre, release bar, {minutes: signed forward})`` in bps, or None.

    A horizon is rounded *up* to whole bars, and the bar it lands on must be
    exactly that many bars of real time after entry -- otherwise the window
    crosses a weekend or a hole in the tape and the signal is dropped.
    """
    step = (bar_min or series.bar_min) * 60
    release = series.index_at(when)
    entry = series.index_after(when)
    if release is None or entry is None or entry != release + 1:
        return None
    entry_px = series.open[entry]
    if not (entry_px > 0):
        return None

    back = series.index_at(series.ts[release] - PRE_MIN * 60)
    if back is None or series.ts[release] - series.ts[back] != PRE_MIN * 60:
        return None
    pre = _log_bps(series.open[back], series.open[release])
    bar = _log_bps(series.open[release], series.close[release])
    if pre is None or bar is None:
        return None

    forward: dict[int, float] = {}
    for minutes in horizons_min:
        steps = max(1, math.ceil(minutes * 60 / step))
        j = entry + steps - 1
        if j >= len(series.ts):
            return None
        if series.ts[j] - series.ts[entry] != (steps - 1) * step:
            return None  # a gap inside the window: this is not a `minutes` move
        value = _log_bps(entry_px, series.close[j])
        if value is None:
            return None
        forward[minutes] = sign * value
    return pre, bar, forward


def has_tape(series: Bars | None, when: float, horizons_min: Iterable[int] = HORIZONS) -> bool:
    """Whether a moment could be measured at all -- used to place the nulls."""
    if series is None:
        return False
    return forward_returns(series, when, 1, horizons_min) is not None


__all__ = [
    "Bars", "CHART_URL", "DEFAULT_BAR_MIN", "HORIZONS", "PRE_MIN", "bars",
    "fetch_chart", "forward_returns", "has_tape",
]
