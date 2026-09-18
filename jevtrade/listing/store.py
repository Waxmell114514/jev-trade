"""A small on-disk JSON cache keyed by string.

Everything the listing study fetches -- announcement pages, their bodies,
candles -- is immutable once published, so caching it has no staleness problem
and makes a re-run cost nothing but Jev calls. It also lets the tests run the
whole pipeline from a temporary directory with no network at all.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable


class Store:
    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        assert self.root is not None
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:60]
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        return self.root / f"{stem}.{digest}.json"

    def get(self, key: str) -> tuple[bool, Any]:
        """``(found, value)`` -- a stored ``None`` is a legitimate value."""
        if self.root is None:
            return False, None
        path = self._path(key)
        if not path.exists():
            return False, None
        try:
            return True, json.loads(path.read_text())["value"]
        except (ValueError, KeyError, OSError):
            return False, None

    def put(self, key: str, value: Any) -> None:
        if self.root is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "value": value}))
        tmp.replace(path)

    def cached(self, key: str, fetch: Callable[[], Any]) -> Any:
        found, value = self.get(key)
        if found:
            self.hits += 1
            return value
        self.misses += 1
        value = fetch()
        self.put(key, value)
        return value
