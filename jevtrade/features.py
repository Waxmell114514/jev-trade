"""Feature computation. Every number in the system is produced here.

This module exists because of one line in TypeSafe's own docs:

    "Jev is not a calculator. We strongly recommend implementing any
     mathematical logic in code."
    -- https://docs.typesafe.ai/model-jaggedness/jev-1.13

So the model is never asked what 76,412.30 minus 76,398.10 is, nor whether
that gap is large. Code computes returns, volatility, z-scores and imbalance;
``discretize.py`` turns them into words; Jev only ever judges the words.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass

from .types import Tick

SHORT = 5
MEDIUM = 15
LONG = 60


@dataclass(frozen=True, slots=True)
class Features:
    """Everything the decision layer knows about the current tape."""

    seq: int
    ts: float
    symbol: str
    mid: float
    spread_bps: float

    ret_1_bps: float  # log return over the last tick, in basis points
    ret_5_bps: float
    ret_15_bps: float

    vol_bps: float  # EWMA per-tick volatility
    vol_ratio: float  # short-horizon vol / long-horizon vol; >1 means expanding
    move_z: float  # size of the 5-tick move in units of its own typical size

    up_fraction: float  # share of the last 10 ticks that closed up
    run_length: int  # signed count of consecutive same-direction ticks

    volume_z: float  # current volume vs the window, in standard deviations
    volume_ratio: float  # current volume / window median

    imbalance: float  # (bid_size - ask_size) / total, in -1..1
    depth_ratio: float  # top-of-book depth vs its window median

    trend_efficiency: float  # |net move| / sum|moves| over MEDIUM; 1=clean trend
    dist_vwap_bps: float  # mid vs window VWAP
    drawdown_bps: float  # mid vs the window's running high

    ticks_seen: int


def _safe_std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


class FeatureWindow:
    """Rolling state over the last ``size`` ticks."""

    def __init__(self, size: int = 120, warmup: int = LONG) -> None:
        if warmup > size:
            raise ValueError("warmup cannot exceed window size")
        self.size = size
        self.warmup = warmup
        self._ticks: deque[Tick] = deque(maxlen=size)
        self._log_returns: deque[float] = deque(maxlen=size)
        self._vol_ewma: float | None = None
        self._ticks_seen = 0

    @property
    def ready(self) -> bool:
        return self._ticks_seen >= self.warmup

    def update(self, tick: Tick) -> Features | None:
        """Ingest a tick. Returns features once the window has warmed up."""
        prev = self._ticks[-1] if self._ticks else None
        self._ticks.append(tick)
        self._ticks_seen += 1

        if prev is not None and prev.mid > 0 and tick.mid > 0:
            self._log_returns.append(math.log(tick.mid / prev.mid))

        if self._log_returns:
            last_sq = self._log_returns[-1] ** 2
            self._vol_ewma = (
                last_sq
                if self._vol_ewma is None
                else 0.94 * self._vol_ewma + 0.06 * last_sq
            )

        if not self.ready:
            return None
        return self._compute(tick)

    # ------------------------------------------------------------------ math

    def _sum_returns(self, n: int) -> float:
        if not self._log_returns:
            return 0.0
        window = list(self._log_returns)[-n:]
        return sum(window)

    def _compute(self, tick: Tick) -> Features:
        returns = list(self._log_returns)
        ticks = list(self._ticks)

        vol = math.sqrt(self._vol_ewma or 0.0)
        vol_bps = vol * 1e4

        short_vol = _safe_std(returns[-MEDIUM:])
        long_vol = _safe_std(returns[-LONG:])
        vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0

        ret_5 = self._sum_returns(SHORT)
        # A 5-tick move is sqrt(5) times a 1-tick move under a random walk.
        expected = vol * math.sqrt(SHORT)
        move_z = ret_5 / expected if expected > 0 else 0.0

        recent = returns[-10:]
        ups = sum(1 for r in recent if r > 0)
        up_fraction = ups / len(recent) if recent else 0.5

        run = 0
        for r in reversed(returns):
            if r == 0:
                break
            if run == 0 or (r > 0) == (run > 0):
                run += 1 if r > 0 else -1
            else:
                break

        volumes = [t.volume for t in ticks]
        vol_mean = statistics.fmean(volumes)
        vol_std = _safe_std(volumes)
        volume_z = (tick.volume - vol_mean) / vol_std if vol_std > 0 else 0.0
        vol_median = statistics.median(volumes)
        volume_ratio = tick.volume / vol_median if vol_median > 0 else 1.0

        total_size = tick.bid_size + tick.ask_size
        imbalance = (
            (tick.bid_size - tick.ask_size) / total_size if total_size > 0 else 0.0
        )
        depths = [t.bid_size + t.ask_size for t in ticks]
        depth_median = statistics.median(depths)
        depth_ratio = total_size / depth_median if depth_median > 0 else 1.0

        med_window = returns[-MEDIUM:]
        gross = sum(abs(r) for r in med_window)
        trend_efficiency = abs(sum(med_window)) / gross if gross > 0 else 0.0

        turnover = sum(t.last * t.volume for t in ticks)
        traded = sum(t.volume for t in ticks)
        vwap = turnover / traded if traded > 0 else tick.mid
        dist_vwap_bps = (tick.mid - vwap) / vwap * 1e4 if vwap > 0 else 0.0

        high = max(t.mid for t in ticks)
        drawdown_bps = (tick.mid - high) / high * 1e4 if high > 0 else 0.0

        return Features(
            seq=tick.seq,
            ts=tick.ts,
            symbol=tick.symbol,
            mid=tick.mid,
            spread_bps=tick.spread_bps,
            ret_1_bps=(returns[-1] if returns else 0.0) * 1e4,
            ret_5_bps=ret_5 * 1e4,
            ret_15_bps=self._sum_returns(MEDIUM) * 1e4,
            vol_bps=vol_bps,
            vol_ratio=vol_ratio,
            move_z=move_z,
            up_fraction=up_fraction,
            run_length=run,
            volume_z=volume_z,
            volume_ratio=volume_ratio,
            imbalance=imbalance,
            depth_ratio=depth_ratio,
            trend_efficiency=trend_efficiency,
            dist_vwap_bps=dist_vwap_bps,
            drawdown_bps=drawdown_bps,
            ticks_seen=self._ticks_seen,
        )
