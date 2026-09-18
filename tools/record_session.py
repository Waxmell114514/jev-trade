"""Record a live demo session to JSON.

    python -m jevtrade.server &
    python tools/record_session.py 200 session.json

Captures the server's event stream so a run can be replayed later without an
API key -- which is how the shareable replay page in the README was built.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8787/stream"


def record(seconds: float, out: str, url: str = DEFAULT_URL) -> None:
    events: list[dict] = []
    started = time.time()
    with urllib.request.urlopen(url, timeout=seconds + 60) as stream:
        while time.time() - started < seconds:
            line = stream.readline().decode()
            if not line:
                break
            if line.startswith("data: "):
                events.append(
                    {"t": round(time.time() - started, 3), "e": json.loads(line[6:])}
                )

    with open(out, "w") as handle:
        json.dump(events, handle)

    kinds: dict[str, int] = {}
    for row in events:
        kinds[row["e"]["type"]] = kinds.get(row["e"]["type"], 0) + 1
    print(f"recorded {len(events)} events over {time.time() - started:.0f}s -> {out}")
    print(f"  {kinds}")


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    out = sys.argv[2] if len(sys.argv) > 2 else "session.json"
    record(seconds, out, sys.argv[3] if len(sys.argv) > 3 else DEFAULT_URL)
