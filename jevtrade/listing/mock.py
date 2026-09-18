"""Offline stand-in for Jev on the listing questions.

It answers whatever question map it is handed, so the reader's plumbing runs
end to end without a key. But it cannot read: it decides "what the announcement
is about" by whether the ticker appears in the *title*, and the direction by a
handful of title words. That makes the offline ``jev`` arm the title-bot with
extra steps, and it should score like it. The gap between that and the real
model's score is the whole result.
"""

from __future__ import annotations

import re
from typing import Any

from ..types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer
from . import reader as R

_BEARISH = re.compile(r"\b(delist\w*|removal|remove\w*|monitoring tag|suspend\w*|cease\w*)\b", re.I)
_LISTING = re.compile(r"\bwill list\b|\blists?\b", re.I)
_MORE = re.compile(r"\b(launch\w*|adds?|will add|support\w*|margin|earn)\b", re.I)
_STOCK = re.compile(r"bstocks|tokenized (stock|securit)", re.I)


class MockListingClient:
    provider = "mock"

    def __init__(self, model: str = "mock-title", **_ignored: Any) -> None:
        self.model = model
        self.questions: dict[str, Any] | None = None
        self.calls = 0

    def evaluate(self, state: dict[str, Any]) -> JevResponse:
        self.calls += 1
        title = str(state.get("title", ""))
        tokens = list(state.get("candidate_tokens") or [])
        if _BEARISH.search(title):
            kind, side, level = R.DELISTING, R.NEGATIVE, 2.4
        elif _STOCK.search(title):
            kind, side, level = R.TOKENIZED_STOCK, R.NONE, 0.5
        elif _LISTING.search(title):
            kind, side, level = R.NEW_LISTING, R.POSITIVE, 2.8
        elif _MORE.search(title):
            kind, side, level = R.MORE_MARKETS, R.POSITIVE, 1.3
        else:
            kind, side, level = R.HOUSEKEEPING, R.NONE, 0.3

        answers: dict[str, Any] = {}
        for key, question in (self.questions or {}).items():
            qtype = question["type"]
            if key == R.EVENT_TYPE:
                answers[key] = _peaked_choice(kind, list(question["criteria"]), 0.7)
            elif key == R.EFFECT or key.startswith("effect_"):
                answers[key] = _peaked_choice(side, list(question["criteria"]), 0.7)
            elif key == R.MAGNITUDE:
                answers[key] = _score(level, list(question["criteria"]))
            elif key.startswith("subject_"):
                token = tokens[int(key.split("_")[1])]
                answers[key] = NoulAnswer(noul=0.9 if _in_title(token, title) else 0.15)
            elif key.startswith("unaffected_"):
                token = tokens[int(key.split("_")[1])]
                answers[key] = NoulAnswer(noul=0.1 if _in_title(token, title) else 0.85)
            elif qtype == "noul":
                answers[key] = NoulAnswer(noul=0.15)
            elif qtype == "choice":
                answers[key] = _peaked_choice(next(iter(question["criteria"])), list(question["criteria"]), 0.5)
            else:
                answers[key] = _score(0.0, list(question["criteria"]))
        return JevResponse(
            model=self.model, answers=answers, input_tokens=len(str(state)) // 4,
            output_tokens=0, latency_ms=1.0, provider=self.provider,
        )


def _in_title(token: str, title: str) -> bool:
    return re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", title) is not None


def _peaked_choice(choice: str, options: list[str], p: float) -> ChoiceAnswer:
    rest = (1.0 - p) / max(len(options) - 1, 1)
    probs = {o: (p if o == choice else rest) for o in options}
    return ChoiceAnswer(choice=choice, probabilities=probs, confidence=0.7)


def _score(level: float, levels: list[str]) -> ScoreAnswer:
    weights = [max(0.02, 1.0 - abs(i - level)) for i in range(len(levels))]
    total = sum(weights)
    return ScoreAnswer(
        score=level, legend={str(i): l for i, l in enumerate(levels)},
        probabilities={str(i): w / total for i, w in enumerate(weights)}, confidence=0.6,
    )
