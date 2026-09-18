"""The live demo server.

Runs the same decision loop as ``engine.py`` against Kraken's real top-of-book,
and streams every tick, decision and fill to a browser over server-sent events.

Two things differ from the backtest engine, both because this is real time:

* **Latency is not simulated, it is measured.** Each decision reports how long
  Jev actually took. The deadline is the snapshot interval itself: a decision
  that takes longer than one snapshot is dropped rather than acted on, which is
  the honest rule for a trading loop and makes the latency budget visible.
* **Fills happen at the next snapshot.** We only know a price when we poll, so
  a decision formed at t is executed against the book at t+1.

The API key stays in this process. The browser never sees it.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import statistics
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .discretize import build_state
from .execution import Broker, ExecutionConfig
from .features import FeatureWindow
from .jev.client import JevError, resolve_client, usd_cost
from .live import KrakenLiveFeed, LiveConfig
from .policy import PolicyConfig, decide

WEB_ROOT = Path(__file__).parent / "web"
CHART_POINTS = 240


class Hub:
    """Fan-out of events to every connected browser."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue] = []

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(event)
            except queue.Full:  # a browser that stopped reading
                pass


class Demo(threading.Thread):
    """The trading loop."""

    daemon = True

    def __init__(self, hub: Hub, args: argparse.Namespace) -> None:
        super().__init__(name="demo")
        self.hub = hub
        self.args = args
        self.feed = KrakenLiveFeed(
            LiveConfig(symbol=args.symbol, interval_ms=args.interval_ms)
        )
        self.client = resolve_client(args.provider, latency_ms=args.mock_latency_ms)
        self.policy = PolicyConfig(max_units=args.max_units)
        self.broker = Broker(ExecutionConfig(taker_fee_bps=args.fee_bps))
        self.window = FeatureWindow(size=180, warmup=args.warmup)

        self.history: deque[dict] = deque(maxlen=CHART_POINTS)
        self.trades: deque[dict] = deque(maxlen=40)
        self.latencies: deque[float] = deque(maxlen=200)
        self.last_decision: dict | None = None
        self.stats: dict[str, Any] = {}
        self.error: str | None = None

        self._pending: float | None = None
        self._decisions = 0
        self._dropped = 0
        self._tokens = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ snapshot

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "type": "hello",
                "config": {
                    "symbol": self.feed.symbol,
                    "interval_ms": self.args.interval_ms,
                    "provider": self.client.provider,
                    "model": self.client.model,
                    "max_units": self.args.max_units,
                    "fee_bps": self.args.fee_bps,
                    "warmup": self.args.warmup,
                },
                "history": list(self.history),
                "trades": list(self.trades),
                "latencies": list(self.latencies),
                "decision": self.last_decision,
                "stats": self.stats,
                "error": self.error,
            }

    # ---------------------------------------------------------------- loop

    def run(self) -> None:
        try:
            primed = self.feed.prime()
            for tick in primed:
                self.window.update(tick)
            self.hub.publish({"type": "primed", "bars": len(primed)})
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.error = f"priming failed: {exc}"
            self.hub.publish({"type": "error", "message": self.error})

        interval = self.args.interval_ms / 1000.0
        while True:
            started = time.perf_counter()
            try:
                self._step()
                self.error = None
            except Exception as exc:  # noqa: BLE001 - a bad poll is not fatal
                self.error = str(exc)
                self.hub.publish({"type": "error", "message": str(exc)})
            time.sleep(max(0.05, interval - (time.perf_counter() - started)))

    def _step(self) -> None:
        tick = self.feed.poll()
        features = self.window.update(tick)

        # A decision from the previous snapshot executes against this book.
        if self._pending is not None:
            fill = self.broker.move_to(tick, self._pending)
            self._pending = None
            if fill is not None:
                row = {
                    "ts": fill.ts,
                    "seq": fill.seq,
                    "side": fill.side,
                    "units": round(fill.units, 6),
                    "price": round(fill.price, 2),
                }
                with self._lock:
                    self.trades.appendleft(row)
                self.hub.publish({"type": "fill", **row})

        point = {
            "seq": tick.seq,
            "ts": tick.ts,
            "mid": round(tick.mid, 2),
            "bid": round(tick.bid, 2),
            "ask": round(tick.ask, 2),
            "spread_bps": round(tick.spread_bps, 4),
            "volume": round(tick.volume, 6),
            "position": round(self.broker.position.units, 6),
            "equity": round(self.broker.position.equity(tick.mid), 4),
        }
        with self._lock:
            self.history.append(point)
        self.hub.publish({"type": "tick", **point})

        if features is None:
            self._publish_stats(tick.mid, warming=True)
            return

        state = build_state(
            features,
            position_units=self.broker.position.units,
            max_units=self.policy.max_units,
            open_pnl_vols=self._open_pnl_vols(tick, features),
            interval_ms=self.args.interval_ms,
        )
        self.hub.publish({"type": "thinking", "seq": tick.seq, "state": state})

        try:
            response = self.client.evaluate(state)
        except JevError as exc:
            self.hub.publish({"type": "error", "message": f"jev: {exc}"})
            self._publish_stats(tick.mid)
            return

        self._tokens += response.input_tokens
        with self._lock:
            self.latencies.append(round(response.latency_ms, 1))

        # The deadline is one snapshot. A decision that misses it describes a
        # book that no longer exists, so it is dropped rather than traded.
        late = response.latency_ms > self.args.interval_ms
        decision = decide(features, response, self.broker.position.units, self.policy)
        if not late:
            self._pending = decision.target_units
            self._decisions += 1
        else:
            self._dropped += 1

        payload = {
            "type": "decision",
            "seq": tick.seq,
            "latency_ms": round(response.latency_ms, 1),
            "budget_pct": round(
                response.latency_ms / self.args.interval_ms * 100, 1
            ),
            "late": late,
            "model": response.model,
            "edge": round(decision.edge, 4),
            "conviction": round(decision.conviction, 4),
            "confidence": round(decision.confidence, 4),
            "gate": decision.gate,
            "target": round(decision.target_units, 6),
            "position": round(self.broker.position.units, 6),
            "state": state,
            "answers": _answers_json(response),
            "tokens": response.input_tokens,
        }
        with self._lock:
            self.last_decision = payload
        self.hub.publish(payload)
        self._publish_stats(tick.mid)

    def _open_pnl_vols(self, tick, features) -> float:
        position = self.broker.position
        if position.units == 0.0 or position.avg_price <= 0:
            return 0.0
        sign = 1.0 if position.units > 0 else -1.0
        pnl_bps = (tick.mid - position.avg_price) / position.avg_price * 1e4 * sign
        return pnl_bps / max(features.vol_bps, 1e-9)

    def _publish_stats(self, mid: float, warming: bool = False) -> None:
        latencies = sorted(self.latencies)
        position = self.broker.position
        stats = {
            "type": "stats",
            "warming": warming,
            "warmup_progress": min(
                1.0, self.window._ticks_seen / max(self.args.warmup, 1)
            ),
            "position": round(position.units, 6),
            "avg_price": round(position.avg_price, 2),
            "gross": round(position.gross_pnl(mid), 2),
            "fees": round(position.fees_paid, 2),
            "net": round(position.equity(mid), 2),
            "fills": len(position.fills),
            "decisions": self._decisions,
            "dropped": self._dropped,
            "p50": round(statistics.median(latencies), 0) if latencies else 0,
            "p95": round(latencies[int(len(latencies) * 0.95)], 0)
            if len(latencies) > 4
            else 0,
            "tokens": self._tokens,
            "cost_usd": round(usd_cost(self._tokens), 6),
        }
        with self._lock:
            self.stats = stats
        self.hub.publish(stats)


def _answers_json(response) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, answer in response.answers.items():
        if answer.type == "noul":
            out[key] = {"type": "noul", "noul": round(answer.noul, 4)}
        elif answer.type == "choice":
            out[key] = {
                "type": "choice",
                "choice": answer.choice,
                "confidence": round(answer.confidence, 4),
                "probabilities": {
                    k: round(v, 4) for k, v in answer.probabilities.items()
                },
            }
        else:
            out[key] = {
                "type": "score",
                "score": round(answer.score, 3),
                "levels": len(answer.legend),
                "confidence": round(answer.confidence, 4),
                "probabilities": {
                    k: round(v, 4) for k, v in answer.probabilities.items()
                },
            }
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    demo: Demo
    hub: Hub

    def log_message(self, *_args) -> None:  # quiet console
        pass

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?")[0]
        if path == "/stream":
            self._stream()
        elif path in ("/", "/index.html"):
            self._static("index.html")
        elif path == "/snapshot":
            self._json(self.demo.snapshot())
        else:
            candidate = (WEB_ROOT / path.lstrip("/")).resolve()
            if candidate.is_file() and WEB_ROOT.resolve() in candidate.parents:
                self._static(candidate.name)
            else:
                self.send_error(404)

    def _static(self, name: str) -> None:
        file = WEB_ROOT / name
        if not file.is_file():
            self.send_error(404)
            return
        body = file.read_bytes()
        kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q = self.hub.subscribe()
        try:
            self._send_event(self.demo.snapshot())
            while True:
                try:
                    event = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                self._send_event(event)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.hub.unsubscribe(q)

    def _send_event(self, event: dict) -> None:
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevtrade-server", description="Live Jev trading demo."
    )
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8787)))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--provider", default="auto", choices=("auto", "jev", "mock"))
    parser.add_argument("--mock-latency-ms", type=float, default=180.0)
    parser.add_argument("--max-units", type=float, default=0.05)
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=0.0,
        help="exchange fee to model; the bid/ask spread is always paid",
    )
    parser.add_argument("--warmup", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hub = Hub()
    demo = Demo(hub, args)

    Handler.demo = demo
    Handler.hub = hub
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True

    demo.start()
    print(f"  jev-trade demo  http://{args.host}:{args.port}")
    print(f"  provider: {demo.client.provider} / {demo.client.model}")
    if demo.client.provider == "mock":
        print("  NOTE: no TYPESAFE_API_KEY, running the offline simulator")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
