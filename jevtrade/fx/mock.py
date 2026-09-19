"""Offline stand-in for Jev on the FX questions.

Same bargain as the listing mock: it answers whatever question map it is handed,
with the right answer type for every key, so the whole pipeline runs end to end
without a key -- and it cannot read. It counts hawkish words against dovish
words, spots five intervention phrases, and guesses the kind from the title.
That makes the offline ``jev`` arm a slightly better-dressed keyword bot, and it
should score like one. The gap between that and the real model is the result.

It is deterministic: the same document always gets the same answers, so a cached
run and a fresh one agree.
"""

from __future__ import annotations

import re
from typing import Any

from ..types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer
from . import reader as R

HAWKISH_WORDS = re.compile(
    r"\b(raise[sd]?|raising|hike[sd]?|tighten\w*|restrictive|elevated|"
    r"further increases?|firmer|upside risks?|vigilan\w*|price stability|"
    r"inflationary|overheating)\b|remains elevated",
    re.I,
)
DOVISH_WORDS = re.compile(
    r"\b(cut[s]?|cutting|lower\w*|ease|easing|accommodat\w*|pause[ds]?|"
    r"downside risks?|slowdown|weaken\w*|softer|patient|supportive)\b",
    re.I,
)
INTERVENTION_PHRASES = (
    (re.compile(r"\b(interven\w+|entered the market|conducted .{0,20}operations? in the "
                r"foreign exchange)\b", re.I), 4.0),
    (re.compile(r"decisive action|appropriate action|rate check|all options", re.I), 3.0),
    (re.compile(r"excessive|one-?sided|disorderly|rapid (and )?speculative", re.I), 2.0),
    (re.compile(r"exchange rate|currency|foreign exchange|\bfx\b", re.I), 1.0),
)

_KIND_PATTERNS = (
    (re.compile(r"press conference|interview|q&a", re.I), R.PRESS_CONFERENCE),
    (re.compile(r"minutes|summary of opinions|account of", re.I), R.MINUTES),
    (re.compile(r"speech|remarks|testimony|lecture|address", re.I), R.SPEECH_OR_TESTIMONY),
    (re.compile(r"fomc statement|monetary policy statement|guideline for money market|"
                r"bank rate|policy rate|monetary policy (summary|decision)|"
                r"issues fomc", re.I), R.RATE_DECISION),
    (re.compile(r"exchange rate|intervention|foreign exchange", re.I), R.FX_COMMENT),
    (re.compile(r"survey|statistics|projections|forecast|results", re.I), R.DATA_OR_SURVEY),
    (re.compile(r"enforcement|supervis|regulat|appointment|payment system", re.I), R.OPERATIONAL),
)


def _lean(text: str) -> tuple[str, float]:
    """Hawkish minus dovish word counts -> (stance, how lopsided, 0..1)."""
    hawks = len(HAWKISH_WORDS.findall(text or ""))
    doves = len(DOVISH_WORDS.findall(text or ""))
    total = hawks + doves
    if total == 0 or hawks == doves:
        return R.NEUTRAL, 0.0
    stance = R.HAWKISH if hawks > doves else R.DOVISH
    return stance, abs(hawks - doves) / total


def _intervention_level(text: str) -> float:
    for pattern, level in INTERVENTION_PHRASES:
        if pattern.search(text or ""):
            return level
    return 0.0


def _kind(title: str, feed_kind: str) -> str:
    for pattern, kind in _KIND_PATTERNS:
        if pattern.search(title or ""):
            return kind
    if feed_kind == "testimony":
        return R.SPEECH_OR_TESTIMONY
    return R.OTHER


class MockFxClient:
    provider = "mock"

    def __init__(self, model: str = "mock-keyword", **_ignored: Any) -> None:
        self.model = model
        self.questions: dict[str, Any] | None = None
        self.calls = 0

    def evaluate(self, state: dict[str, Any]) -> JevResponse:
        self.calls += 1
        title = str(state.get("title", ""))
        body = str(state.get("body", ""))
        text = f"{title}\n{body}"
        changes = list(state.get("changed_sentences") or [])
        stance, lopsided = _lean(text)
        kind = _kind(title, str(state.get("feed_kind", "")))
        talky = kind in (R.MINUTES, R.PRESS_CONFERENCE)
        level = 2.6 if kind == R.RATE_DECISION else (1.6 if talky else 0.6)
        currency = str(state.get("currency", ""))
        want = {R.HAWKISH: R.BUY, R.DOVISH: R.SELL}.get(stance, R.NEITHER)

        answers: dict[str, Any] = {}
        for key, question in (self.questions or {}).items():
            options = list(question["criteria"])
            if key == R.KIND:
                answers[key] = _choice(kind, options, 0.65)
            elif key == R.STANCE:
                answers[key] = _choice(stance, options, 0.45 + 0.45 * lopsided)
            elif key == R.STANCE_REVERSED:
                answers[key] = _choice(want, options, 0.45 + 0.45 * lopsided)
            elif key == R.HORIZON:
                answers[key] = _choice(
                    R.HOURS_H if kind == R.RATE_DECISION else R.MINUTES_H, options, 0.6
                )
            elif key == R.MAGNITUDE:
                answers[key] = _score(level, options)
            elif key == R.INTERVENTION_TIER:
                answers[key] = _score(_intervention_level(text), options)
            elif key == R.POLICY_RELEVANT:
                answers[key] = NoulAnswer(
                    noul=0.85 if kind in (R.RATE_DECISION, R.MINUTES, R.PRESS_CONFERENCE)
                    else (0.55 if kind == R.SPEECH_OR_TESTIMONY else 0.15)
                )
            elif key == R.NEW_INFORMATION:
                answers[key] = NoulAnswer(noul=0.7 if changes else 0.4)
            elif key == R.GUIDANCE_CHANGED:
                answers[key] = NoulAnswer(noul=0.6 if changes else 0.2)
            elif key == R.SURPRISE:
                answers[key] = NoulAnswer(noul=0.5 if state.get("calendar") else 0.3)
            elif key == R.HOLDER_UNAFFECTED:
                answers[key] = NoulAnswer(noul=max(0.05, 0.8 - 0.7 * lopsided))
            elif key.startswith("diff_stance_"):
                i = int(key.rsplit("_", 1)[1])
                sentence = changes[i]["now"] if i < len(changes) else ""
                answers[key] = _choice(_lean(sentence)[0], options, 0.6)
            elif key.startswith("diff_material_"):
                i = int(key.rsplit("_", 1)[1])
                sentence = changes[i]["now"] if i < len(changes) else ""
                answers[key] = NoulAnswer(noul=0.75 if _lean(sentence)[1] > 0 else 0.25)
            elif question["type"] == "noul":
                answers[key] = NoulAnswer(noul=0.2)
            elif question["type"] == "choice":
                answers[key] = _choice(options[0], options, 0.5)
            else:
                answers[key] = _score(0.0, options)

        return JevResponse(
            model=self.model,
            answers=answers,
            input_tokens=len(str(state)) // 4 + len(currency),
            output_tokens=0,
            latency_ms=1.0,
            provider=self.provider,
        )


def _choice(choice: str, options: list[str], p: float) -> ChoiceAnswer:
    if choice not in options:
        choice = options[-1]
    p = min(max(p, 1.0 / max(len(options), 1)), 0.95)
    rest = (1.0 - p) / max(len(options) - 1, 1)
    return ChoiceAnswer(
        choice=choice,
        probabilities={o: (p if o == choice else rest) for o in options},
        confidence=0.7,
    )


def _score(level: float, levels: list[str]) -> ScoreAnswer:
    weights = [max(0.02, 1.0 - abs(i - level)) for i in range(len(levels))]
    total = sum(weights)
    return ScoreAnswer(
        score=level,
        legend={str(i): text for i, text in enumerate(levels)},
        probabilities={str(i): w / total for i, w in enumerate(weights)},
        confidence=0.6,
    )


__all__ = ["MockFxClient", "HAWKISH_WORDS", "DOVISH_WORDS"]
