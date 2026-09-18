"""Cache Jev answers by exact state.

Two reasons, one practical and one methodological.

Practical: the same headline recurs, and an identical state deserves an
identical answer. Any production system doing this would cache it.

Methodological: sweeping a threshold in ``JevStrategy`` does not change the
state the model sees, so the answers must not change either. Re-querying would
add sampling noise to a comparison that is supposed to isolate the threshold.
Cache once, sweep offline.
"""

from __future__ import annotations

import json
from typing import Any

from ..types import JevResponse


class CachingClient:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self._cache: dict[str, JevResponse] = {}
        self.hits = 0
        self.misses = 0

    @property
    def questions(self):
        return getattr(self.inner, "questions", None)

    @questions.setter
    def questions(self, value) -> None:
        if hasattr(self.inner, "questions"):
            self.inner.questions = value

    def evaluate(self, state: dict[str, Any]) -> JevResponse:
        key = json.dumps(state, sort_keys=True)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        response = self.inner.evaluate(state)
        self._cache[key] = response
        return response
