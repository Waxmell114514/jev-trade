"""The tick judge: Dukascopy's files, the entry rule, and what the spread costs.

Four things here are load-bearing and would each be invisible if wrong. The
**month is zero-based** in the URL, so an off-by-one fetches the wrong month and
finds prices that look entirely plausible. The **price scale** is 1e5 except for
JPY quotes, where it is 1e3, and getting that backwards moves USDJPY by two
orders of magnitude. The **entry side** is the ask for a long and the bid for a
short, which is the whole difference between this study and one that trades at a
mid nobody quotes. And an hour with **no ticks** -- every weekend hour of
seventeen years -- must be a cached, cheap, permanent "no", not a retry loop.

Everything is offline: the .bi5 fixtures are packed and LZMA'd in the test, so
there are no binary blobs in the repository and the format is stated in code.
"""

import lzma
import struct
from datetime import datetime, timezone

import pytest

from jevtrade.fx import documents as D
from jevtrade.fx import study as S
from jevtrade.fx import ticks as K
from jevtrade.listing.store import Store

HOUR = 3600
# 2026-09-16 18:00 UTC, the September FOMC statement, on the hour for arithmetic.
FOMC = datetime(2026, 9, 16, 18, 0, tzinfo=timezone.utc).timestamp()


def bi5(records, *, scale=1e5):
    """``[(seconds into the hour, bid, ask)]`` -> the bytes Dukascopy would serve."""
    body = b"".join(
        struct.pack(">IIIff", int(round(s * 1000)), int(round(ask * scale)),
                    int(round(bid * scale)), 1.0, 1.0)
        for s, bid, ask in records
    )
    return lzma.compress(body)


def flat_hour(hour_start, *, n=60, bid=1.1000, spread=0.00002, step=0.0, scale=1e5):
    """One tick a minute, drifting by ``step`` per tick."""
    rows = []
    for i in range(n):
        px = bid + step * i
        rows.append((i * 60.0, px, px + spread))
    return bi5(rows, scale=scale)


class FakeFeed:
    """A fetcher over a ``{url: bytes}`` map that counts what it was asked for."""

    def __init__(self, files):
        self.files = files
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        return self.files.get(url, b"")


def tape_over(hours, *, symbol="EURUSD=X", store=None, **options):
    """A tape whose feed serves ``{hour start: bytes}`` for that symbol."""
    feed = FakeFeed({
        K.hour_url(K.feed_symbol(symbol), datetime.fromtimestamp(h, timezone.utc)): raw
        for h, raw in hours.items()
    })
    return K.TickTape(store or Store(None), symbol, fetcher=feed, workers=2, **options), feed


# ------------------------------------------------------------------- the format

def test_the_url_puts_january_at_zero_and_pads_the_day_and_hour():
    when = datetime(2009, 1, 15, 4, 30, tzinfo=timezone.utc)
    assert K.hour_url("EURUSD", when).endswith("/EURUSD/2009/00/15/04h_ticks.bi5")
    december = datetime(2016, 12, 14, 19, 0, tzinfo=timezone.utc)
    assert K.hour_url("eurusd", december).endswith("/EURUSD/2016/11/14/19h_ticks.bi5")
    assert K.hour_url("USDJPY", december).startswith("https://datafeed.dukascopy.com/datafeed/")


def test_prices_are_scaled_by_a_hundred_thousand_and_jpy_by_a_thousand():
    assert K.scale_for("EURUSD") == 1e5 and K.scale_for("GBPUSD") == 1e5
    assert K.scale_for("USDJPY") == 1e3 and K.scale_for("EURJPY") == 1e3

    # The two records this module was probed against, byte for byte.
    euro = lzma.compress(struct.pack(">IIIff", 58, 115510, 115508, 0.9, 5.85))
    yen = lzma.compress(struct.pack(">IIIff", 220, 156225, 156219, 1.1, 5.4))
    [tick] = K.decode_bi5(euro, K.scale_for("EURUSD"), at=1000.0)
    assert tick == pytest.approx((1000.058, 1.15508, 1.15510))
    [tick] = K.decode_bi5(yen, K.scale_for("USDJPY"), at=0.0)
    assert tick.ts == pytest.approx(0.22)
    assert (tick.bid, tick.ask) == pytest.approx((156.219, 156.225))
    assert tick.mid == pytest.approx(156.222)


def test_an_empty_or_truncated_body_is_an_hour_with_no_ticks():
    assert K.decode_bi5(b"", 1e5) == []
    whole = lzma.compress(struct.pack(">IIIff", 5, 10, 9, 1.0, 1.0) + b"\x00\x01\x02")
    assert len(K.decode_bi5(whole, 1e5)) == 1  # the trailing part-record is ignored


def test_the_study_pairs_map_onto_the_feeds_names():
    assert K.feed_symbol("EURUSD=X") == "EURUSD"
    assert K.feed_symbol("JPY=X") == "USDJPY"  # Yahoo's odd one out
    assert K.feed_symbol("GBPUSD=X") == "GBPUSD"


def test_hours_covering_spans_the_boundary_inclusively():
    assert K.hours_covering(HOUR + 10, HOUR + 20) == [HOUR]
    assert K.hours_covering(HOUR - 1, 2 * HOUR + 1) == [0, HOUR, 2 * HOUR]


# -------------------------------------------------------------------- caching

def test_an_empty_hour_is_cached_and_never_asked_for_twice(tmp_path):
    store = Store(tmp_path)
    tape, feed = tape_over({0: flat_hour(0)}, store=store)
    assert tape.hour(HOUR) == []            # the feed has nothing for this hour
    assert len(feed.calls) == 1
    other, again = tape_over({0: flat_hour(0)}, store=store)
    assert other.hour(HOUR) == []
    assert again.calls == []                # a second run does not re-ask

    # ... and the bytes are cached compressed, not decoded.
    assert other.hour(0)[0].bid == pytest.approx(1.1000)
    keys = [p.name for p in tmp_path.iterdir()]
    assert any(name.startswith("dukas_EURUSD") for name in keys)


def test_a_transient_failure_is_retried_and_is_not_cached_as_empty(tmp_path, monkeypatch):
    import urllib.error

    tries = []

    def flaky(url, timeout=30.0):
        tries.append(url)
        raise urllib.error.HTTPError(url, 503, "busy", {}, None)

    monkeypatch.setattr(K.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(K.time, "sleep", lambda _s: None)
    with pytest.raises(K.TickFeedError):
        K.fetch_bi5("https://x/y.bi5", tries=3)
    assert len(tries) == 3

    store = Store(tmp_path)
    assert K.fetch_hour(store, "EURUSD", 0, fetcher=lambda _u: (_ for _ in ()).throw(
        K.TickFeedError("boom"))) == []
    found, _ = store.get("dukas:EURUSD:1970-01-01T00")
    assert not found  # a bad minute of network is not a permanent hole in the tape


def test_a_missing_file_is_a_settled_empty_hour(monkeypatch):
    import urllib.error

    def missing(url, timeout=30.0):
        raise urllib.error.HTTPError(url, 404, "no", {}, None)

    monkeypatch.setattr(K.urllib.request, "urlopen", missing)
    assert K.fetch_bi5("https://x/y.bi5") == b""


# ---------------------------------------------------------------------- tape

def test_a_window_spans_an_hour_boundary_and_stays_in_order():
    tape, feed = tape_over({0: flat_hour(0, n=60), HOUR: flat_hour(HOUR, n=60, bid=1.2)})
    ticks = tape.window(3400, HOUR + 130)
    assert [t.ts for t in ticks] == [3420.0, 3480.0, 3540.0,
                                     HOUR + 0.0, HOUR + 60.0, HOUR + 120.0]
    assert ticks[3].bid == pytest.approx(1.2)
    assert len(feed.calls) == 2  # both hours, once each
    tape.window(3400, HOUR + 130)
    assert len(feed.calls) == 2  # and they stay in memory


def test_quote_at_takes_the_last_tick_before_the_moment_and_goes_stale():
    tape, _ = tape_over({0: flat_hour(0, n=10)})  # ticks at 0, 60, ... 540
    quote = tape.quote_at(125.0)
    assert quote.ts == 120.0 and quote.bid == pytest.approx(1.1) and quote.mid > quote.bid
    assert tape.quote_at(540.0).ts == 540.0     # "at or before" includes at
    assert tape.quote_at(540 + 299) is not None
    assert tape.quote_at(540 + 301) is None     # the tape went quiet: no quote
    assert tape.quote_at(540 + 400, max_age_s=600) is not None
    assert tape.mid_at(125.0) == pytest.approx(1.10001)


def test_first_tick_after_is_the_entry_and_gives_up_when_the_tape_is_shut():
    tape, _ = tape_over({0: flat_hour(0, n=10)})
    assert tape.first_tick_after(121.0).ts == 180.0
    assert tape.first_tick_after(180.0).ts == 180.0  # "at or after" includes at
    assert tape.first_tick_after(540.0).ts == 540.0
    assert tape.first_tick_after(541.0) is None      # nothing prints for two minutes


def test_the_decoded_cache_is_bounded_but_the_answers_are_not():
    hours = {h * HOUR: flat_hour(h * HOUR, n=4, bid=1.0 + h / 100) for h in range(6)}
    tape, feed = tape_over(hours, max_hours=2)
    for h in range(6):
        assert tape.quote_at(h * HOUR + 100).bid == pytest.approx(1.0 + h / 100)
    assert len(tape._hours) <= 2
    assert tape.quote_at(100).bid == pytest.approx(1.0)  # evicted, re-decoded, same answer


# --------------------------------------------------------------- measurement

def event_tape(*, step=0.0, spread=0.00002, hours=3, bid=1.1000):
    """Three hours of one-minute ticks around ``FOMC``, starting an hour before."""
    start = K.hour_start(FOMC) - HOUR
    files = {}
    px = bid
    for i in range(hours):
        files[start + i * HOUR] = flat_hour(start + i * HOUR, n=60, bid=px,
                                            spread=spread, step=step)
        px = px + step * 60
    return tape_over(files)


def test_a_long_pays_the_ask_and_a_short_gets_the_bid():
    tape, _ = event_tape(spread=0.00022)  # 2 bps wide, flat price
    long = K.forward_returns_ticks(tape, FOMC + 5, +1, latency_s=1.0, horizons_min=(15,))
    short = K.forward_returns_ticks(tape, FOMC + 5, -1, latency_s=1.0, horizons_min=(15,))
    assert long.entry_px > long.entry_mid > short.entry_px
    assert long.spread_bps == pytest.approx(2.0, abs=0.01)
    assert short.spread_bps == pytest.approx(long.spread_bps)
    # Nothing moved, so both sides are out exactly the half spread, and the
    # same trade measured mid-to-mid shows zero.
    assert long.fwd_bps[15] == pytest.approx(-1.0, abs=0.01)
    assert short.fwd_bps[15] == pytest.approx(-1.0, abs=0.01)
    assert long.fwd_mid_bps[15] == pytest.approx(0.0, abs=1e-9)
    assert short.fwd_mid_bps[15] == pytest.approx(0.0, abs=1e-9)


def test_the_entry_is_the_first_tick_after_the_latency_and_rush_is_what_it_missed():
    tape, _ = event_tape(step=0.0001)  # +1 pip a minute, about +0.9 bps
    fast = K.forward_returns_ticks(tape, FOMC + 1, +1, latency_s=0.0, horizons_min=(15,))
    slow = K.forward_returns_ticks(tape, FOMC + 1, +1, latency_s=120.0, horizons_min=(15,))
    assert fast.entry_ts == FOMC + 60.0      # the next one-minute print
    assert slow.entry_ts == FOMC + 180.0     # two minutes of latency, two prints later
    # A long in a rising market: the slow entry has already given up the move.
    assert fast.rush_bps > 0 and slow.rush_bps > fast.rush_bps
    assert slow.fwd_bps[15] < fast.fwd_bps[15]
    # A short in the same market is hurt by the rush, so the sign flips.
    short = K.forward_returns_ticks(tape, FOMC + 1, -1, latency_s=120.0, horizons_min=(15,))
    assert short.rush_bps == pytest.approx(-slow.rush_bps)


def test_the_pre_window_is_unsigned_and_needs_a_quote_fifteen_minutes_back():
    tape, _ = event_tape(step=0.0001)
    both = [K.forward_returns_ticks(tape, FOMC + 1, s, latency_s=1.0, horizons_min=(5,))
            for s in (+1, -1)]
    assert both[0].pre_bps == pytest.approx(both[1].pre_bps)
    assert both[0].pre_bps > 0
    # An event ten minutes into the tape has no quote fifteen minutes before it.
    start = K.hour_start(FOMC) - HOUR
    assert K.forward_returns_ticks(tape, start + 600, +1, latency_s=1.0,
                                   horizons_min=(5,)) is None


def test_a_horizon_that_runs_past_the_close_drops_the_whole_signal():
    """The Friday close: a "+60m" return measured across it is a 51-hour return."""
    tape, _ = event_tape()
    late = K.hour_start(FOMC) + 2 * HOUR - 600  # ten minutes before the tape stops
    assert K.forward_returns_ticks(tape, late, +1, latency_s=1.0, horizons_min=(5,))
    assert K.forward_returns_ticks(tape, late, +1, latency_s=1.0, horizons_min=(5, 60)) is None
    # ... and a moment inside the closed session has no entry at all.
    shut = K.hour_start(FOMC) + 6 * HOUR
    assert K.forward_returns_ticks(tape, shut, +1, latency_s=1.0, horizons_min=(5,)) is None


def test_has_ticks_agrees_with_the_measurement_and_rejects_the_weekend_cheaply():
    tape, feed = event_tape()
    assert K.has_ticks(tape, FOMC, latency_s=1.0, horizons_min=(15,))
    before = len(feed.calls)
    shut = K.hour_start(FOMC) + 10 * HOUR
    assert not K.has_ticks(tape, shut, latency_s=1.0, horizons_min=(15, 60))
    assert len(feed.calls) - before <= 3  # one empty hour answers it, not a dozen


# ---------------------------------------------------------------- the study

def tick_tapes(tape):
    """A ``TickTapes`` that hands every symbol the same prepared tape."""
    tapes = S.TickTapes(Store(None), factory=lambda *a, **k: tape)
    return tapes


def test_measure_on_ticks_carries_the_spread_the_rush_and_the_latency():
    tape, _ = event_tape(spread=0.00022)
    tapes = tick_tapes(tape)
    signals = [S.Signal("a", FOMC + 5, "FOMC", "EURUSD=X", -1)]
    [outcome] = S.measure(tapes, signals, horizons=(15,), latency_s=1.0, workers=2)
    assert outcome.spread_bps == pytest.approx(2.0, abs=0.01)
    assert outcome.latency_s == 1.0 and outcome.entry_ts > FOMC
    assert outcome.fwd_bps[15] == pytest.approx(-1.0, abs=0.01)
    assert outcome.fwd_mid_bps[15] == pytest.approx(0.0, abs=1e-9)
    assert outcome.release_bar_bps == 0.0  # there is no bar on a tick tape


def test_the_tick_null_keeps_the_pair_and_side_and_lands_where_the_tape_is():
    start = K.hour_start(FOMC) - 3 * 86400
    files = {start + i * HOUR: flat_hour(start + i * HOUR, n=60) for i in range(6 * 24)}
    tape, _ = tape_over(files)
    tapes = tick_tapes(tape)
    signal = S.Signal("c", start + 3 * 86400, "t", "EURUSD=X", -1)
    outcome = S.Outcome(signal, "EURUSD", 0.0, 0.0, {15: 0.0})
    nulls = S.null_signals(tapes, [outcome], per=3, span_days=2.0, exclude_h=3.0,
                           horizons=(15,), latency_s=1.0)
    assert len(nulls) == 3
    assert all(n.pair == "EURUSD=X" and n.sign == -1 for n in nulls)
    assert all(3 * 3600 <= abs(n.ts - signal.ts) <= 2 * 86400 for n in nulls)
    assert all(K.has_ticks(tape, n.ts, latency_s=1.0, horizons_min=(15,)) for n in nulls)


def test_summarize_reports_the_spread_the_rush_and_the_no_spread_number():
    signal = S.Signal("c", FOMC, "t", "EURUSD=X", +1)
    outcomes = [
        S.Outcome(signal, "EURUSD", 0.0, 0.0, {15: v - 1.0}, spread_bps=2.0,
                  rush_bps=3.0, latency_s=1.0, fwd_mid_bps={15: v})
        for v in (30.0, 20.0, -5.0, 25.0)
    ]
    nulls = [S.Outcome(signal, "EURUSD", 0.0, 0.0, {15: v}) for v in (1.0, -2.0, 0.0, 1.0)]
    summary = S.summarize("x", [signal] * 4, outcomes, nulls, horizons=(15,), latency_s=1.0)
    assert summary.spread.mean == 2.0 and summary.rush.mean == 3.0
    assert summary.fwd[15].mean == pytest.approx(16.5)
    assert summary.fwd_mid[15].mean == pytest.approx(17.5)  # a bp better, per side
    assert summary.latency_s == 1.0


def test_the_latency_sweep_charges_for_being_slow():
    tape, _ = event_tape(step=0.0001)  # a market moving up a pip a minute
    tapes = tick_tapes(tape)
    signals = [S.Signal("a", FOMC + 5, "FOMC", "EURUSD=X", +1)]
    rows = S.latency_sweep(tapes, signals, latencies=(0.0, 120.0), horizons=(15,),
                           per=1, workers=1)
    assert [r.name for r in rows] == ["0s", "120s"]
    assert [r.latency_s for r in rows] == [0.0, 120.0]
    assert rows[0].measured == rows[1].measured == 1
    assert rows[0].fwd[15].mean > rows[1].fwd[15].mean  # two seconds' late costs
    assert rows[1].rush.mean > rows[0].rush.mean
    assert all(r.spread.mean > 0 for r in rows)


# ------------------------------------------------------- the reading cache key

def test_a_changed_diff_invalidates_a_cached_reading(tmp_path):
    """A statement whose previous edition appears is a different question."""
    from jevtrade.fx import reader as R
    from jevtrade.fx.mock import MockFxClient

    def document(ts, title, body):
        return D.Document(id=f"fed:{int(ts)}:x", issuer="fed", kind=D.MONETARY_POLICY,
                          ts=ts, title=title, body=body, url="https://x/a.htm",
                          currency="USD")

    now = document(FOMC, "Federal Reserve issues FOMC statement",
                   "Inflation remains elevated. The Committee decided to raise rates.")
    earlier = document(FOMC - 49 * 86400, "Federal Reserve issues FOMC statement",
                       "Inflation is easing. The Committee decided to hold rates.")

    store = Store(tmp_path)
    client = MockFxClient()
    alone = S.read_all([now], lambda: R.Reader(client), store=store, cache_tag="mock:v1")
    assert alone[0].diffs == []
    calls = client.calls

    # The same document, same id, same tree -- but now it has a predecessor, so
    # round one sees twelve more questions and must be asked again.
    together = S.read_all([earlier, now], lambda: R.Reader(client), store=store,
                          cache_tag="mock:v1")
    assert client.calls > calls
    assert together[1].diffs  # the diff really is part of the question now

    calls = client.calls
    again = S.read_all([earlier, now], lambda: R.Reader(client), store=store,
                       cache_tag="mock:v1")
    assert client.calls == calls  # and the second time it is cached
    assert [len(r.diffs) for r in again] == [len(r.diffs) for r in together]


def test_the_state_hash_moves_with_the_body_the_diff_and_the_calendar():
    document = D.Document(id="fed:1:x", issuer="fed", kind=D.MONETARY_POLICY, ts=FOMC,
                          title="FOMC statement", body="a body", url="u", currency="USD")
    base = S.state_hash(document, [], None)
    assert base == S.state_hash(document, [], None)
    assert base != S.state_hash(document, [("was", "now")], None)
    assert base != S.state_hash(document, [], {"title": "Fed Rate", "forecast": "4.00%"})
    other = D.Document(**{**document.__dict__, "body": "another body"})
    assert base != S.state_hash(other, [], None)
    # ... and not with anything the model never sees.
    quiet = D.Document(**{**document.__dict__, "url": "https://elsewhere/x.htm"})
    assert base == S.state_hash(quiet, [], None)


def test_previous_of_finds_the_last_edition_across_a_long_window():
    """Seventeen years of statements: the September one diffs against July's."""
    from jevtrade.fx import diff as DF

    docs = [
        D.Document(id=f"fed:{i}:x", issuer="fed", kind=D.MONETARY_POLICY,
                   ts=FOMC - (200 - i) * 45 * 86400,
                   title=f"Federal Reserve issues FOMC statement, meeting {i}",
                   body=f"The Committee decided to hold rates at meeting {i}.",
                   url=f"u{i}", currency="USD")
        for i in range(200)
    ]
    index = DF.Editions(docs)
    assert index.previous(docs[199]) is docs[198]
    assert index.previous(docs[0]) is None
    assert DF.previous_of(docs[100], docs) is docs[99]
    _, changes = DF.diff_for(docs[150], index)
    assert changes and "meeting 149" in changes[0][0]


# ----------------------------------------------------------------------- CLI

def test_cli_fx_grades_on_ticks_end_to_end(tmp_path, monkeypatch, capsys):
    """``--tape dukascopy --provider mock`` with a fake feed and fixture pages."""
    import json
    from pathlib import Path

    from jevtrade import cli
    from jevtrade.fx import documents as docs_module

    fixtures = Path(__file__).parent / "fixtures" / "fx"

    def get_text(url, timeout=30.0):
        name = url.rsplit("/", 1)[-1]
        if name in {"ne-press.json", "ne-speeches.json", "ne-testimony.json"}:
            return (fixtures / name).read_text()
        if "monetary20260916a" in name:
            return (fixtures / "fomc-2026-09-16.html").read_text()
        if "monetary20260729a" in name:
            return (fixtures / "fomc-2026-07-29.html").read_text()
        return "<html><body><p>" + "no body for this one at all, but long enough.</p></body></html>"

    # A year of one-minute ticks, so every document and every null is coverable.
    span_start = K.hour_start(FOMC) - 400 * 86400

    def feed(url):
        stamp = url.rsplit("/datafeed/", 1)[-1]
        _symbol, year, month, day, hour = stamp.split("/")
        when = datetime(int(year), int(month) + 1, int(day), int(hour[:2]), tzinfo=timezone.utc)
        return flat_hour(when.timestamp(), n=60, bid=1.1 + (when.timestamp() % 997) / 1e6)

    monkeypatch.setattr(docs_module, "get_text", get_text)
    monkeypatch.setattr(docs_module, "fetch_body",
                        lambda url: docs_module.body_text(url, get_text(url)))
    monkeypatch.setattr(K, "fetch_bi5", feed)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert span_start < FOMC

    out_file = tmp_path / "run.json"
    code = cli.main([
        "fx", "--provider", "mock", "--issuers", "fed",
        "--since", "2026-07-01", "--until", "2026-09-18",
        "--tape", "dukascopy", "--latency", "1.0", "--latency-sweep",
        "--horizons", "1,15,60", "--cache", str(tmp_path / "cache"),
        "--workers", "2", "--workers-io", "4", "--null-per", "1",
        "--out", str(out_file),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "rush" in printed and "sprd" in printed
    assert "latency sweep" in printed and "no spread" in printed
    assert "paying the ask to go long and the bid to go short" in printed

    record = json.loads(out_file.read_text())
    assert record["tape"] == "dukascopy" and record["latency_s"] == 1.0
    assert record["bar_min"] is None and record["horizons"] == [1, 15, 60]
    assert record["latency_sweep"] and len(record["latency_sweep"]) == len(K.LATENCIES)
    arms = {a["name"]: a for a in record["arms"]}
    graded = arms["all text"]
    assert graded["outcomes"] and len(graded["outcomes"]) == graded["measured"]
    row = graded["outcomes"][0]
    assert set(row) >= {"ts", "title", "pair", "side", "strength", "pre_bps",
                        "rush_bps", "spread_bps", "fwd_bps", "fwd_mid_bps"}
    assert row["spread_bps"] > 0 and row["fwd_bps"]["15"] < row["fwd_mid_bps"]["15"]


def test_cli_warns_when_a_horizon_is_shorter_than_the_bar(tmp_path, monkeypatch, capsys):
    from jevtrade import cli
    from jevtrade.fx import documents as docs_module

    import pathlib

    fixtures = pathlib.Path(__file__).parent / "fixtures" / "fx"
    monkeypatch.setattr(docs_module, "get_text",
                        lambda url, timeout=30.0: (fixtures / url.rsplit("/", 1)[-1]).read_text())
    monkeypatch.setattr(docs_module, "fetch_body", lambda url: "")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    cli.main(["fx", "--provider", "mock", "--issuers", "fed", "--days", "0.001",
              "--horizons", "1,15", "--cache", str(tmp_path / "cache")])
    assert "shorter than one 5-minute bar" in capsys.readouterr().err


def test_cli_takes_since_and_until_over_days():
    from jevtrade.cli import build_parser

    args = build_parser().parse_args(
        ["fx", "--since", "2009-01-01", "--until", "2026-09-19", "--tape", "dukascopy"])
    assert args.since == "2009-01-01" and args.until == "2026-09-19"
    assert args.tape == "dukascopy" and args.latency == 1.0
    assert args.latency_sweep is False and args.workers == 3
