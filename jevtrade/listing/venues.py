"""One-minute candles around a moment, from whichever venue has the market.

Binance's own history only covers symbols that still trade there, so a token
that was delisted, or that Binance listed after it already traded elsewhere,
has to be measured on OKX or Coinbase. The venue order is fixed and the same
for every arm, so no arm is measured on a friendlier tape than another.

Candles come back on a continuous minute grid with gaps forward-filled: thin
markets skip minutes, and a return computed by bar *index* on a gappy series
would silently span more time than it claims.
"""

from __future__ import annotations

import math
import sys
import urllib.error
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .announcements import get_json
from .store import Store

BINANCE = "https://data-api.binance.vision/api/v3/klines?symbol={m}&interval=1m&startTime={s}&endTime={e}&limit=1000"
OKX_HISTORY = "https://www.okx.com/api/v5/market/history-candles?instId={m}&bar=1m&after={after}&limit=100"
OKX_RECENT = "https://www.okx.com/api/v5/market/candles?instId={m}&bar=1m&after={after}&limit=300"
COINBASE = "https://api.exchange.coinbase.com/products/{m}/candles?granularity=60&start={s}&end={e}"

Raw = list[tuple[float, float, float]]  # (epoch seconds, open, close)


def _q(market: str) -> str:
    """Tickers are not always ASCII any more (a meme coin called 牛来 listed in 2026)."""
    return urllib.parse.quote(market, safe="-")


class NoMarket(Exception):
    """The venue has never had this market: cache that, never retry."""


@dataclass(frozen=True)
class Candles:
    venue: str
    market: str
    start: float  # first grid minute, epoch seconds
    open: list[float]
    close: list[float]
    bars: int  # real bars behind the grid, as a thinness signal

    def index_at(self, when: float) -> int | None:
        i = int(math.floor((when - self.start) / 60.0))
        if i < 0 or i >= len(self.close):
            return None
        return i


def _grid(raw: Raw, start: float, minutes: int) -> tuple[list[float], list[float]]:
    by_minute = {int(math.floor(ts / 60.0)) * 60: (o, c) for ts, o, c in raw}
    opens: list[float] = []
    closes: list[float] = []
    last: float | None = None
    for m in range(minutes):
        ts = start + 60 * m
        bar = by_minute.get(ts)
        if bar is not None:
            o, c = bar
            opens.append(o if last is None or o > 0 else last)
            closes.append(c)
            last = c
        else:
            opens.append(last if last is not None else float("nan"))
            closes.append(last if last is not None else float("nan"))
    return opens, closes


# ------------------------------------------------------------------ venues

def binance(token: str, start: float, end: float) -> Raw:
    market = f"{token}USDT"
    try:
        rows = get_json(BINANCE.format(m=_q(market), s=int(start * 1000), e=int(end * 1000)))
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 404):
            raise NoMarket(market) from exc
        raise
    return [(r[0] / 1000.0, float(r[1]), float(r[4])) for r in rows]


def okx(token: str, start: float, end: float) -> Raw:
    market = f"{token}-USDT"
    out: Raw = []
    after = int(end * 1000) + 60_000
    for url in (OKX_HISTORY, OKX_RECENT):
        cursor = after
        for _ in range(6):
            try:
                payload = get_json(url.format(m=_q(market), after=cursor))
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500 and exc.code != 429:
                    raise NoMarket(market) from exc
                raise
            if str(payload.get("code")) in ("51001", "51000"):
                raise NoMarket(market)
            rows = payload.get("data") or []
            if not rows:
                break
            for r in rows:
                ts = int(r[0]) / 1000.0
                if start <= ts <= end:
                    out.append((ts, float(r[1]), float(r[4])))
            cursor = int(rows[-1][0])
            if cursor / 1000.0 <= start:
                break
        if out:
            break
    return out


def coinbase(token: str, start: float, end: float) -> Raw:
    market = f"{token}-USD"
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    try:
        rows = get_json(COINBASE.format(m=_q(market), s=iso(start), e=iso(end)))
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 404):
            raise NoMarket(market) from exc
        raise
    if isinstance(rows, dict):  # {"message": "NotFound"}
        raise NoMarket(market)
    return [(float(r[0]), float(r[3]), float(r[4])) for r in rows]


VENUES: list[tuple[str, Callable[[str, float, float], Raw]]] = [
    ("binance", binance),
    ("okx", okx),
    ("coinbase", coinbase),
]


def window(
    store: Store,
    token: str,
    when: float,
    *,
    before_min: int = 30,
    after_min: int = 90,
    venues: list[tuple[str, Callable[[str, float, float], Raw]]] | None = None,
) -> Candles | None:
    """Candles from ``before_min`` before ``when`` to ``after_min`` after it."""
    anchor = int(math.floor(when / 60.0)) * 60
    start = float(anchor - 60 * before_min)
    minutes = before_min + after_min + 1
    end = start + 60 * (minutes - 1)

    for venue, fetch in venues or VENUES:
        market_key = f"market:{venue}:{token}"
        found, exists = store.get(market_key)
        if found and exists is False:
            continue
        key = f"candles:{venue}:{token}:{int(start)}:{int(end)}"
        found, raw = store.get(key)
        if not found:
            try:
                raw = fetch(token, start, end)
            except NoMarket:
                store.put(market_key, False)
                continue
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                # Transient: skip this venue for now and do not cache the miss.
                print(f"  {venue} {token}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            store.put(key, raw)
        raw = [tuple(r) for r in raw]
        # A market that exists but has no bars near the announcement is as
        # good as no market: skip it and let the next venue try.
        if len(raw) < 3 or not any(start <= r[0] <= anchor for r in raw):
            continue
        opens, closes = _grid(raw, start, minutes)
        return Candles(
            venue=venue, market=f"{venue}:{token}", start=start,
            open=opens, close=closes, bars=len(raw),
        )
    return None
