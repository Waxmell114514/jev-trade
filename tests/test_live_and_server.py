"""The live feed and the demo server's plumbing, without touching the network."""

import json

import pytest

from jevtrade import live as live_mod
from jevtrade.live import KrakenLiveFeed, LiveConfig
from jevtrade.server import Hub, _answers_json
from jevtrade.types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer

TICKER = {
    "XXBTZUSD": {
        "a": ["77380.20000", "1", "1.000"],
        "b": ["77380.10000", "2", "2.000"],
        "c": ["77380.20000", "0.06313918"],
        "v": ["454.61004919", "1961.97086223"],
        "t": [18453, 97025],
    }
}

TRADES = {
    "XXBTZUSD": [
        ["77370.1", "0.5", 1000.2, "b", "m", "", 1],
        ["77371.2", "0.3", 1000.8, "s", "m", "", 2],
        ["77372.3", "0.7", 1001.4, "b", "m", "", 3],
        ["77373.4", "0.2", 1002.1, "s", "m", "", 4],
    ],
    "last": "x",
}


@pytest.fixture
def feed(monkeypatch):
    f = KrakenLiveFeed(LiveConfig(symbol="BTC", interval_ms=1000))

    def fake_get(url, timeout, attempts=3):
        return TRADES if "Trades" in url else TICKER

    monkeypatch.setattr(live_mod, "_get", fake_get)
    return f


def test_ticker_becomes_a_tick_with_a_real_book(feed):
    tick = feed.poll()
    assert tick.bid == 77380.10
    assert tick.ask == 77380.20
    assert tick.bid_size == 2.0  # the resting size, not a reconstruction
    assert tick.ask_size == 1.0
    assert tick.symbol == "BTC-USD"
    assert tick.spread_bps == pytest.approx(0.0129, abs=1e-3)


def test_volume_is_the_delta_between_polls(feed):
    first = feed.poll()
    assert first.volume == 0.0  # nothing to difference against yet

    bumped = json.loads(json.dumps(TICKER))
    bumped["XXBTZUSD"]["v"] = ["456.61004919", "1963.0"]
    feed_get = lambda url, timeout, attempts=3: bumped  # noqa: E731
    import jevtrade.live as m

    m._get = feed_get
    second = feed.poll()
    assert second.volume == pytest.approx(2.0)


def test_volume_survives_the_utc_midnight_reset(feed):
    feed.poll()
    reset = json.loads(json.dumps(TICKER))
    reset["XXBTZUSD"]["v"] = ["0.5", "1.0"]  # counter rolled over
    import jevtrade.live as m

    m._get = lambda url, timeout, attempts=3: reset
    tick = feed.poll()
    assert tick.volume == 0.0  # not a huge negative number


def test_priming_buckets_trades_into_bars(feed):
    ticks = feed.prime()
    # Trades at t=1000.2, 1000.8, 1001.4, 1002.1 fall into three 1s buckets.
    assert len(ticks) == 3
    assert ticks[0].volume == pytest.approx(0.8)  # 0.5 + 0.3
    assert ticks[1].volume == pytest.approx(0.7)
    assert ticks[2].volume == pytest.approx(0.2)
    assert [t.seq for t in ticks] == [0, 1, 2]
    assert all(t.bid < t.last < t.ask for t in ticks)
    # Each bar's last price is the last trade in its bucket.
    assert ticks[0].last == pytest.approx(77371.2)


def test_sequence_numbers_continue_from_priming_into_live(feed):
    primed = feed.prime()
    tick = feed.poll()
    assert tick.seq == primed[-1].seq + 1


def test_hub_broadcasts_to_every_subscriber():
    hub = Hub()
    a, b = hub.subscribe(), hub.subscribe()
    hub.publish({"type": "tick", "mid": 1.0})
    assert a.get_nowait()["mid"] == 1.0
    assert b.get_nowait()["mid"] == 1.0


def test_hub_drops_a_client_that_unsubscribed():
    hub = Hub()
    q = hub.subscribe()
    hub.unsubscribe(q)
    hub.publish({"type": "tick"})
    assert q.empty()


def test_hub_survives_a_browser_that_stopped_reading():
    """A full queue must not raise into the trading loop."""
    hub = Hub()
    q = hub.subscribe()
    for _ in range(q.maxsize + 20):
        hub.publish({"type": "tick"})
    assert q.full()


def test_answers_serialise_to_json_safe_shapes():
    response = JevResponse(
        model="jev-1.13.0",
        answers={
            "direction": ChoiceAnswer(
                choice="up", probabilities={"up": 0.6, "down": 0.4}, confidence=0.5
            ),
            "setup_quality": ScoreAnswer(
                score=1.5,
                legend={"0": "a", "1": "b", "2": "c"},
                probabilities={"0": 0.2, "1": 0.3, "2": 0.5},
                confidence=0.4,
            ),
            "disorderly": NoulAnswer(noul=0.25),
        },
        input_tokens=1,
        output_tokens=0,
        latency_ms=1.0,
        provider="test",
    )
    payload = _answers_json(response)
    json.dumps(payload)  # must be serialisable

    assert payload["direction"]["choice"] == "up"
    assert payload["setup_quality"]["levels"] == 3
    assert payload["disorderly"]["noul"] == 0.25
