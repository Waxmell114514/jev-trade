"""HTTP client for ``POST https://api.typesafe.ai/v1/systemone``.

Implemented against the reference at https://docs.typesafe.ai/api using only
the standard library, so the simulator has no install step. TypeSafe also ships
first-party SDKs (``pip install typesafe-sdk``) which handle retries and typed
questions for you; this client exists to keep the wire format visible, since
the wire format *is* the interesting part of a System One model.

One deviation from the SDK defaults is deliberate. The SDKs retry generously,
which is right for a support queue and wrong for a trading loop: a decision
that arrives after its deadline is worse than no decision. So retries default
to a single extra attempt and the caller enforces its own deadline.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from ..types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
API_KEY_ENV = "TYPESAFE_API_KEY"
BASE_URL_ENV = "TYPESAFE_BASE_URL"
MODEL_ENV = "JEV_MODEL"

# $0.042 per million input tokens; output tokens are free.
# https://docs.typesafe.ai/models
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000


class JevError(RuntimeError):
    """Any failure talking to the System One endpoint."""


class JevAuthError(JevError):
    """401: missing or invalid API key."""


class JevValidationError(JevError):
    """422: the request body failed validation."""


class JevRateLimitError(JevError):
    """429: over the account's token or request rate limit."""


class JevOverloadedError(JevError):
    """529: TypeSafe is temporarily overloaded."""


class JevTimeoutError(JevError):
    """The request did not complete within the client timeout."""


class JevProtocolError(JevError):
    """The response did not match the documented answer schema."""


class JevClient(Protocol):
    """What the engine needs from a decision provider."""

    provider: str
    model: str

    def evaluate(self, state: dict[str, Any]) -> JevResponse: ...


# ------------------------------------------------------------------- parsing


def parse_response(
    payload: dict[str, Any], latency_ms: float, provider: str = "jev"
) -> JevResponse:
    """Turn a raw /v1/systemone body into typed answers.

    Jev cannot emit a value outside the declared schema, so this is a mapping
    step rather than the defensive parse-and-repair an LLM would need. It still
    validates, because a transport or a proxy can mangle anything.
    """
    try:
        raw_answers = payload["answers"]
        usage = payload.get("usage") or {}
        answers = {key: _parse_answer(key, value) for key, value in raw_answers.items()}
        return JevResponse(
            model=str(payload.get("model", "unknown")),
            answers=answers,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=latency_ms,
            provider=provider,
        )
    except JevProtocolError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise JevProtocolError(f"malformed System One response: {exc}") from exc


def _parse_answer(key: str, raw: dict[str, Any]):
    kind = raw.get("type")
    if kind == "noul":
        return NoulAnswer(noul=float(raw["noul"]))
    if kind == "choice":
        return ChoiceAnswer(
            choice=str(raw["choice"]),
            probabilities={k: float(v) for k, v in raw["probabilities"].items()},
            confidence=float(raw["confidence"]),
        )
    if kind == "score":
        return ScoreAnswer(
            score=float(raw["score"]),
            legend={str(k): str(v) for k, v in raw["legend"].items()},
            probabilities={str(k): float(v) for k, v in raw["probabilities"].items()},
            confidence=float(raw["confidence"]),
        )
    raise JevProtocolError(f"answer {key!r} has unknown type {kind!r}")


def usd_cost(input_tokens: int) -> float:
    return input_tokens * USD_PER_INPUT_TOKEN


# -------------------------------------------------------------------- client


@dataclass
class RetryPolicy:
    max_attempts: int = 2
    base_backoff_s: float = 0.2
    max_backoff_s: float = 2.0


class HttpJevClient:
    """Calls the real Jev API."""

    provider = "jev"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        timeout_s: float = 5.0,
        retry: RetryPolicy | None = None,
        questions: dict[str, Any] | None = None,
    ) -> None:
        # Keys routinely arrive from a .env file written on Windows, and the
        # trailing \r makes an unusable header with a baffling error deep in
        # http.client. Strip it here instead.
        self.api_key = (api_key or os.environ.get(API_KEY_ENV, "")).strip()
        if not self.api_key:
            raise JevAuthError(
                f"no API key: set {API_KEY_ENV} or pass api_key explicitly"
            )
        self.model = model
        self.base_url = (
            base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout_s = timeout_s
        self.retry = retry or RetryPolicy()
        # Public: a caller with a different job (the market-making loop asks a
        # different set) swaps this rather than reaching into the client.
        self.questions = questions
        self._rng = random.Random(0)

    # The question map is static, so build it once per client.
    def _question_map(self) -> dict[str, Any]:
        if self.questions is None:
            from ..questions import trading_questions

            self.questions = trading_questions()
        return self.questions

    def evaluate(self, state: dict[str, Any]) -> JevResponse:
        body = {
            "state": state,
            "model": self.model,
            "questions": self._question_map(),
        }
        started = time.perf_counter()
        payload = self._post("/v1/systemone", body)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return parse_response(payload, latency_ms, provider=self.provider)

    def list_models(self) -> list[dict[str, Any]]:
        return self._get("/v1/models").get("models", [])

    # ---------------------------------------------------------- transport

    def _request(self, request: urllib.request.Request) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.retry.max_attempts):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                detail = _read_error(exc)
                if exc.code == 401:
                    raise JevAuthError(f"401 unauthorized: {detail}") from exc
                if exc.code == 422:
                    raise JevValidationError(f"422 invalid request: {detail}") from exc
                if exc.code in (429, 529) or exc.code >= 500:
                    last = (
                        JevRateLimitError(f"429 rate limited: {detail}")
                        if exc.code == 429
                        else JevOverloadedError(f"{exc.code}: {detail}")
                    )
                    if attempt + 1 < self.retry.max_attempts:
                        self._sleep(attempt, exc.headers.get("retry-after"))
                        continue
                    raise last from exc
                raise JevError(f"{exc.code}: {detail}") from exc
            except TimeoutError as exc:
                raise JevTimeoutError(
                    f"no response within {self.timeout_s}s"
                ) from exc
            except urllib.error.URLError as exc:
                last = JevError(f"connection failed: {exc.reason}")
                if attempt + 1 < self.retry.max_attempts:
                    self._sleep(attempt, None)
                    continue
                raise last from exc
        raise last or JevError("request failed")  # pragma: no cover

    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), self.retry.max_backoff_s))
                return
            except ValueError:
                pass
        backoff = min(
            self.retry.base_backoff_s * (2**attempt), self.retry.max_backoff_s
        )
        time.sleep(backoff * (0.5 + self._rng.random()))

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "jev-trade/0.1",
            },
        )
        return self._request(request)

    def _get(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            method="GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "jev-trade/0.1",
            },
        )
        return self._request(request)


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode()[:400]
    except Exception:  # pragma: no cover - best effort
        return exc.reason or ""


# Settings that only mean something to the offline stub. Callers pass one set
# of options for either provider, so these are dropped before the real client
# sees them rather than crashing it.
MOCK_ONLY_OPTIONS = frozenset({"latency_ms", "noise"})


def resolve_client(
    provider: str = "auto", *, model: str | None = None, **kwargs: Any
) -> JevClient:
    """Pick a provider.

    ``auto`` uses the real API when ``TYPESAFE_API_KEY`` is set and the offline
    mock otherwise, so the repo is runnable before your early-access key lands.
    """
    from .mock import MockJevClient

    model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
    if provider not in ("auto", "jev", "mock"):
        raise ValueError(f"unknown provider {provider!r}")

    use_real = provider == "jev" or (
        provider == "auto" and bool(os.environ.get(API_KEY_ENV))
    )
    if use_real:
        options = {k: v for k, v in kwargs.items() if k not in MOCK_ONLY_OPTIONS}
        return HttpJevClient(model=model, **options)
    return MockJevClient(model=model, **kwargs)
