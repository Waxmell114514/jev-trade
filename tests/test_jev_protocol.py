"""The request we build and the response we parse must match the published API.

Reference: https://docs.typesafe.ai/api
"""

import pytest

from jevtrade.jev import MockJevClient, parse_response, resolve_client
from jevtrade.jev.client import (
    DEFAULT_MODEL,
    HttpJevClient,
    JevAuthError,
    JevProtocolError,
    usd_cost,
)
from jevtrade.questions import QUESTION_IDS, SETUP_LEVELS, build_request


def test_request_matches_the_documented_schema():
    body = build_request({"instrument": "BTC-USD"})
    assert set(body) == {"state", "model", "questions"}
    assert body["model"] == DEFAULT_MODEL

    for key, question in body["questions"].items():
        assert key in QUESTION_IDS
        assert question["type"] in {"noul", "choice", "score"}
        assert isinstance(question["instructions"], str)
        assert question["instructions"].strip()

        if question["type"] == "choice":
            # criteria is a map of option -> description, at least two options.
            assert isinstance(question["criteria"], dict)
            assert len(question["criteria"]) >= 2
        elif question["type"] == "score":
            # criteria is an ordered list with at least two levels.
            assert isinstance(question["criteria"], list)
            assert len(question["criteria"]) >= 2
        else:
            assert set(question.get("criteria", {})) <= {"true", "false"}


def test_score_rubric_has_the_levels_the_policy_assumes():
    assert len(SETUP_LEVELS) >= 2


def test_parse_response_handles_all_three_answer_types():
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "is_urgent": {"type": "noul", "noul": 0.92},
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
                "confidence": 0.82,
            },
            "frustration": {
                "type": "score",
                "score": 1.6,
                "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
                "confidence": 0.78,
            },
        },
        "usage": {"input_tokens": 312, "output_tokens": 48},
    }
    response = parse_response(payload, latency_ms=91.0)

    assert response.model == "jev-1.13.0"
    assert response.noul("is_urgent").noul == 0.92
    assert response.choice("department").choice == "technical"
    assert response.choice("department").p("technical") == 0.85
    assert response.choice("department").p("absent_option") == 0.0
    assert response.score("frustration").score == 1.6
    assert response.score("frustration").normalized == pytest.approx(0.8)
    assert response.input_tokens == 312
    assert response.latency_ms == 91.0


def test_asking_for_the_wrong_answer_type_is_an_error():
    response = parse_response(
        {"model": "m", "answers": {"a": {"type": "noul", "noul": 0.5}}, "usage": {}},
        latency_ms=1.0,
    )
    with pytest.raises(TypeError):
        response.choice("a")


def test_unknown_answer_type_is_rejected():
    with pytest.raises(JevProtocolError):
        parse_response(
            {"model": "m", "answers": {"a": {"type": "vibes", "x": 1}}, "usage": {}},
            latency_ms=1.0,
        )


def test_malformed_response_is_rejected():
    with pytest.raises(JevProtocolError):
        parse_response({"model": "m"}, latency_ms=1.0)


def test_pricing_matches_the_published_rate():
    # $0.042 per million input tokens.
    assert usd_cost(1_000_000) == pytest.approx(0.042)


def test_http_client_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevAuthError):
        HttpJevClient()


def test_resolve_client_falls_back_to_the_mock_without_a_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert isinstance(resolve_client("auto"), MockJevClient)


def test_resolve_client_prefers_the_real_api_when_a_key_exists(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    assert isinstance(resolve_client("auto"), HttpJevClient)


def test_resolve_client_rejects_an_unknown_provider():
    with pytest.raises(ValueError):
        resolve_client("gpt")


def test_mock_answers_every_question_and_stays_in_range():
    client = MockJevClient()
    state = {
        "price_action": {"last_5_snapshots": "rising hard"},
        "order_book": {"resting_size": "heavily weighted to buyers"},
    }
    response = client.evaluate(state)

    assert set(response.answers) == set(QUESTION_IDS)
    for answer in response.answers.values():
        if answer.type == "noul":
            assert 0.0 <= answer.noul <= 1.0
        else:
            total = sum(answer.probabilities.values())
            assert total == pytest.approx(1.0)
            assert 0.0 <= answer.confidence <= 1.0


def test_mock_is_deterministic_for_the_same_state():
    client = MockJevClient()
    state = {"price_action": {"last_5_snapshots": "falling hard"}}
    first = client.evaluate(state).choice("direction").probabilities
    second = client.evaluate(state).choice("direction").probabilities
    assert first == second


def test_mock_leans_with_the_book_it_is_shown():
    """Sanity check that the stub responds to its input at all."""
    client = MockJevClient(noise=0.0)
    bullish = client.evaluate(
        {
            "price_action": {
                "last_snapshot": "rising",
                "last_5_snapshots": "rising hard",
                "last_15_snapshots": "rising",
                "character": "a clean one-way run",
                "versus_session": "around the session's average traded price",
            },
            "order_book": {"resting_size": "heavily weighted to buyers"},
        }
    ).choice("direction")
    bearish = client.evaluate(
        {
            "price_action": {
                "last_snapshot": "falling",
                "last_5_snapshots": "falling hard",
                "last_15_snapshots": "falling",
                "character": "a clean one-way run",
                "versus_session": "around the session's average traded price",
            },
            "order_book": {"resting_size": "heavily weighted to sellers"},
        }
    ).choice("direction")

    assert bullish.p("up") > bullish.p("down")
    assert bearish.p("down") > bearish.p("up")


def test_mock_is_labelled_so_reports_cannot_pass_it_off_as_jev():
    assert MockJevClient().provider == "mock"


def test_mock_only_options_do_not_reach_the_real_client(monkeypatch):
    """The CLI passes one option set for either provider; the real client
    must not choke on a setting that only means something to the stub."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    assert isinstance(resolve_client("auto", latency_ms=5.0), HttpJevClient)
    assert isinstance(resolve_client("jev", latency_ms=5.0, noise=0.1), HttpJevClient)


def test_mock_still_receives_its_own_options(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client = resolve_client("auto", latency_ms=7.0)
    assert isinstance(client, MockJevClient)
    assert client.latency_ms == 7.0


def test_api_key_whitespace_is_stripped(monkeypatch):
    """A key pasted from a CRLF .env must not produce an invalid HTTP header."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test\r\n")
    assert HttpJevClient().api_key == "sk-test"
    assert HttpJevClient(api_key="  sk-spaced  ").api_key == "sk-spaced"


def test_blank_api_key_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
    with pytest.raises(JevAuthError):
        HttpJevClient()
