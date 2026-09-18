"""The listing study: reading an announcement in one second, graded by the tape.

These test the measuring instrument, not the result. The one that matters most
is the entry rule: a reader answering in a second still enters at the open of
the *next* minute here, so a move inside the release minute is never credited.
"""

import json
import math
import urllib.error

import pytest

from jevtrade.listing import reader as R
from jevtrade.listing import study as S
from jevtrade.listing.announcements import Announcement, body_text, collect
from jevtrade.listing.baseline import body_bot, title_bot
from jevtrade.listing.mock import MockListingClient
from jevtrade.listing.store import Store
from jevtrade.listing.tickers import candidates
from jevtrade.listing.venues import Candles, NoMarket, _grid, window
from jevtrade.types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer


def ann(title, body="", ts=1_700_000_000.0, code="c1", catalog=48):
    return Announcement(id=1, code=code, catalog=catalog, ts=ts, title=title, body=body)


# ------------------------------------------------------------ announcements

def test_body_text_flattens_the_cms_node_tree_with_line_breaks():
    tree = {"node": "root", "child": [
        {"node": "element", "tag": "p", "child": [
            {"node": "text", "text": "Binance will list "},
            {"node": "element", "tag": "strong", "child": [{"node": "text", "text": "X (X)"}]},
        ]},
        {"node": "element", "tag": "p", "child": [{"node": "text", "text": "Pairs: X/USDT&nbsp;"}]},
    ]}
    assert body_text(json.dumps(tree)) == "Binance will list X (X)\nPairs: X/USDT"


def test_body_text_falls_back_to_html_for_older_articles():
    assert body_text("<p>Hello <b>there</b></p><p>Second</p>") == "Hello there\nSecond"


def test_collect_pages_until_the_window_is_covered_and_caches_bodies(tmp_path):
    calls = {"pages": [], "bodies": 0}
    rows = {  # newest first, as the endpoint returns them
        1: [{"id": 3, "code": "c", "title": "T3", "releaseDate": 3_000_000}],
        2: [{"id": 2, "code": "b", "title": "T2", "releaseDate": 2_000_000},
            {"id": 1, "code": "a", "title": "T1", "releaseDate": 1_000_000}],
        3: [{"id": 0, "code": "z", "title": "T0", "releaseDate": 500_000}],
    }

    def page(catalog, n, size):
        calls["pages"].append(n)
        return rows.get(n, [])

    def body(code):
        calls["bodies"] += 1
        return f"body of {code}"

    store = Store(tmp_path)
    out = collect(store, since=1500.0, until=2500.0, catalogs=(48,),
                  page_fetcher=page, body_fetcher=body)
    assert [a.code for a in out] == ["b"]
    assert out[0].body == "body of b"
    assert calls["pages"] == [1, 2]  # page 2 reaches below `since`; page 3 never fetched
    assert calls["bodies"] == 1

    collect(store, since=1500.0, until=2500.0, catalogs=(48,), page_fetcher=page, body_fetcher=body)
    assert calls["bodies"] == 1  # bodies are immutable: cached


# ----------------------------------------------------------------- tickers

@pytest.mark.parametrize("title,body,expected", [
    ("Binance Will List MarsCoin (MARSCOIN) with Seed Tag Applied",
     "Pairs: MARSCOIN/USDT at 10:15 (UTC)", ["MARSCOIN"]),
    ("Binance Futures Will Launch USDⓈ-Margined PONSUSDT and 哈基米USDT Perpetual Contracts", "",
     ["PONS", "哈基米"]),
    ("Notice of Removal of Spot Trading Pairs", "ADA/TUSD, 1000SATS/FDUSD, PEPE/USDC, USDC/USDT",
     ["ADA", "SATS", "PEPE"]),
    ("Quarterly Delivery Contracts", "BTCUSDT Quarterly 0326 and ETHUSDT", ["BTC", "ETH"]),
    ("A notice with nothing in it - 2026-09-18", "See you at 12:00 (UTC).", []),
])
def test_candidate_tickers(title, body, expected):
    assert candidates(title, body) == expected


def test_candidates_put_title_tokens_first_and_cap_the_list():
    body = " ".join(f"T{i:02d}/USDT" for i in range(40))
    out = candidates("Binance Will List Zed (ZED)", body, limit=10)
    assert out[0] == "ZED" and len(out) == 10


# ---------------------------------------------------------------- baseline

def test_title_bot_buys_every_ticker_in_a_listing_title_and_shorts_removals():
    a = ann("Binance Will Support the Broadcom (AVGO) and Seagate (STX) Cash Dividend",
            "Details mention BNB/USDT.")
    assert [(s.token, s.sign) for s in title_bot(a)] == [("AVGO", 1), ("STX", 1)]
    assert [(s.token, s.sign) for s in body_bot(a)] == [("AVGO", 1), ("STX", 1), ("BNB", 1)]
    d = ann("Notice of Removal of Spot Trading Pairs", "OPEN/BTC, SAGA/BTC")
    assert title_bot(d) == []  # nothing in the title to trade
    assert [(s.token, s.sign) for s in body_bot(d)] == [("OPEN", -1), ("SAGA", -1)]


def test_title_bot_reads_a_seed_tag_removal_as_a_removal():
    """A documented failure of matching: the word 'remove' wins, the meaning loses."""
    a = ann("Binance Will Remove the Seed Tag from Pepe (PEPE)")
    assert [(s.token, s.sign) for s in title_bot(a)] == [("PEPE", -1)]


# ------------------------------------------------------------------ venues

def test_grid_forward_fills_missing_minutes():
    raw = [(0.0, 10.0, 11.0), (120.0, 12.0, 13.0)]  # minute 1 missing
    opens, closes = _grid(raw, 0.0, 3)
    assert closes == [11.0, 11.0, 13.0]
    assert opens == [10.0, 11.0, 12.0]


def _bars(start, n, price=100.0):
    return [(start + 60 * i, price, price) for i in range(n)]


def test_window_falls_through_venues_and_remembers_a_missing_market(tmp_path):
    calls = []

    def nope(token, s, e):
        calls.append("nope")
        raise NoMarket(token)

    def yes(token, s, e):
        calls.append("yes")
        return _bars(s, 121)

    store = Store(tmp_path)
    venues = [("first", nope), ("second", yes)]
    c = window(store, "ABC", 3_600_000.0, venues=venues)
    assert c.venue == "second" and len(c.close) == 121
    window(store, "ABC", 3_700_000.0, venues=venues)
    assert calls == ["nope", "yes", "yes"]  # the first venue is never asked again


def test_window_skips_a_market_with_no_bars_before_the_moment(tmp_path):
    def late(token, s, e):  # bars start only after the announcement
        return _bars(e - 600, 10)

    def full(token, s, e):
        return _bars(s, 121)

    c = window(Store(tmp_path), "ABC", 3_600_000.0, venues=[("late", late), ("full", full)])
    assert c.venue == "full"


def test_window_does_not_cache_a_transient_failure(tmp_path):
    calls = []

    def flaky(token, s, e):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("reset")
        return _bars(s, 121)

    store = Store(tmp_path)
    assert window(store, "ABC", 3_600_000.0, venues=[("v", flaky)]) is None
    assert window(store, "ABC", 3_600_000.0, venues=[("v", flaky)]) is not None


# -------------------------------------------------------------- measurement

def candles_with_step(step_bps, *, at="after", release=30, n=121):
    """Flat at 100, then a one-off jump. ``at`` places the jump relative to entry."""
    start = 0.0
    opens, closes = [], []
    price = 100.0
    for i in range(n):
        o = price
        if at == "inside_release" and i == release:
            price = price * math.exp(step_bps / 1e4)
        elif at == "after" and i == release + 1:
            price = price * math.exp(step_bps / 1e4)
        opens.append(o)
        closes.append(price)
    return Candles(venue="t", market="t:X", start=start, open=opens, close=closes, bars=n)


def test_forward_returns_credit_a_move_after_entry_at_every_horizon():
    c = candles_with_step(50.0, at="after")
    pre, bar, fwd = S.forward_returns(c, 30 * 60 + 12.0, +1, (1, 5, 60))
    assert pre == 0.0 and bar == 0.0
    assert all(abs(v - 50.0) < 1e-6 for v in fwd.values())
    _, _, short = S.forward_returns(c, 30 * 60 + 12.0, -1, (5,))
    assert abs(short[5] + 50.0) < 1e-6


def test_a_move_inside_the_release_minute_is_never_credited():
    """The entry is the open of the minute *after* the announcement, on purpose."""
    c = candles_with_step(50.0, at="inside_release")
    pre, bar, fwd = S.forward_returns(c, 30 * 60 + 12.0, +1, (1, 15))
    assert abs(bar - 50.0) < 1e-6  # it happened, and it is reported as the release bar
    assert fwd == {1: 0.0, 15: 0.0}  # but no arm gets paid for it


def test_forward_returns_refuse_a_tape_that_does_not_cover_the_horizon():
    c = candles_with_step(0.0, n=40)
    assert S.forward_returns(c, 30 * 60.0, +1, (60,)) is None
    assert S.forward_returns(c, 99_999.0, +1, (1,)) is None


def test_null_signals_keep_token_and_side_and_stay_away_from_the_event():
    sig = S.Signal("c", 1_000_000.0, "t", "ABC", -1)
    out = S.Outcome(sig, "v", 0.0, 0.0, {15: 0.0})
    nulls = S.null_signals([out], per=20, span_days=2, exclude_h=3, seed=1)
    assert len(nulls) == 20
    assert all(n.token == "ABC" and n.sign == -1 for n in nulls)
    assert all(3 * 3600 <= abs(n.ts - sig.ts) <= 2 * 86400 for n in nulls)


def test_summarize_reports_hit_rate_and_z_against_the_null():
    sig = S.Signal("c", 0.0, "t", "ABC", +1)
    outcomes = [S.Outcome(sig, "v", 0.0, 0.0, {15: v}) for v in (30.0, 20.0, -5.0, 25.0)]
    nulls = [S.Outcome(sig, "v", 0.0, 0.0, {15: v}) for v in (1.0, -2.0, 0.0, 1.0)]
    s = S.summarize("x", [sig] * 6, outcomes, nulls, horizons=(15,))
    assert s.signals == 6 and s.measured == 4
    assert s.hit[15] == 0.75
    assert s.fwd[15].mean == 17.5 and s.z[15] > 2


# ------------------------------------------------------------------ reader

def scripted(subject=0.9, side=R.POSITIVE, level=3.0, conditional=0.0, priced_in=0.0,
             unaffected=0.1, confidence=0.9):
    """A client whose answers are derived from whatever question map it is handed."""

    class Client:
        provider = "test"
        model = "scripted"

        def __init__(self):
            self.questions = None
            self.calls = 0

        def evaluate(self, state):
            self.calls += 1
            answers = {}
            for key, q in self.questions.items():
                if q["type"] == "choice":
                    opts = list(q["criteria"])
                    pick = side if side in opts else opts[0]
                    if key == R.EVENT_TYPE:
                        pick = R.NEW_LISTING
                    probs = {o: (0.8 if o == pick else 0.2 / (len(opts) - 1)) for o in opts}
                    answers[key] = ChoiceAnswer(pick, probs, confidence)
                elif q["type"] == "score":
                    levels = q["criteria"]
                    answers[key] = ScoreAnswer(level, {str(i): l for i, l in enumerate(levels)},
                                               {str(i): 1 / len(levels) for i in range(len(levels))}, 0.5)
                elif key.startswith("subject_"):
                    answers[key] = NoulAnswer(subject)
                elif key.startswith("unaffected_"):
                    answers[key] = NoulAnswer(unaffected)
                elif key == R.CONDITIONAL:
                    answers[key] = NoulAnswer(conditional)
                elif key == R.PRICED_IN:
                    answers[key] = NoulAnswer(priced_in)
                else:
                    answers[key] = NoulAnswer(0.0)
            return JevResponse("scripted", answers, 1000, 0, 400.0, "test")

    return Client()


LISTING = ann("Binance Will List Zed (ZED)", "Pairs: ZED/USDT. Fee paid in BNB/USDT.")


def test_round_two_runs_only_when_a_token_clears_the_subject_floor():
    client = scripted(subject=0.9)
    reading = R.Reader(client).read(LISTING)
    assert client.calls == 2 and reading.rounds == 2
    assert reading.latency_ms == 800.0

    client = scripted(subject=0.2)
    reading = R.Reader(client).read(LISTING)
    assert client.calls == 1 and reading.rounds == 1
    # No second round, no trade: whatever the first round said about direction,
    # a token the model says the announcement is not about is never a signal.
    assert reading.signals(0.0) == []
    assert reading.verdicts[0].side == R.POSITIVE and reading.verdicts[0].strength == 0.0


def test_verdict_strength_is_arithmetic_on_the_model_probabilities():
    reading = R.Reader(scripted(subject=0.9, level=3.0, unaffected=0.1)).read(LISTING)
    zed = next(v for v in reading.verdicts if v.token == "ZED")
    assert zed.sign == 1
    assert abs(zed.strength - 0.9 * 0.9 * 1.0 * 0.8) < 1e-9

    damped = R.Reader(scripted(conditional=1.0, priced_in=1.0)).read(LISTING)
    assert damped.verdicts[0].strength < 0.2 * zed.strength


def test_a_none_verdict_or_low_confidence_never_becomes_a_signal():
    none = R.Reader(scripted(side=R.NONE)).read(LISTING)
    assert none.signals(0.0) == []
    shaky = R.Reader(scripted(confidence=0.3)).read(LISTING)
    assert shaky.signals(0.0, min_confidence=0.5) == [] and shaky.verdicts[0].strength > 0


def test_mock_client_answers_every_question_with_the_right_type():
    client = MockListingClient()
    tokens = candidates(LISTING.title, LISTING.body)
    for questions in (R.round_one_questions(tokens), R.round_two_questions(tokens, [0, 1])):
        client.questions = questions
        response = client.evaluate(R.build_state(LISTING, tokens, round_no=1, body_chars=100))
        for key, q in questions.items():
            assert response.answers[key].type == q["type"], key
    reading = R.Reader(MockListingClient()).read(LISTING)
    assert [v.token for v in reading.signals(0.2)] == ["ZED"]  # a title match, nothing more


def test_read_all_caches_readings_and_gives_each_worker_its_own_reader(tmp_path):
    made = []

    def make():
        client = scripted()
        made.append(client)
        return R.Reader(client)

    anns = [ann(f"Binance Will List T{i} (T{i})", code=f"c{i}") for i in range(6)]
    store = Store(tmp_path)
    first = S.read_all(anns, make, store=store, cache_tag="t", workers=3)
    assert len(first) == 6 and 1 <= len(made) <= 3
    calls = sum(c.calls for c in made)
    second = S.read_all(anns, make, store=store, cache_tag="t", workers=3)
    assert sum(c.calls for c in made) == calls  # nothing asked again
    assert [v.strength for r in second for v in r.verdicts] == [v.strength for r in first for v in r.verdicts]


def test_disagreements_show_where_reading_and_matching_differ():
    reading = R.Reader(scripted(side=R.NEGATIVE)).read(
        ann("Binance Will Remove the Seed Tag from Zed (ZED)", "ZED/USDT", code="k"))
    reading.verdicts[0].side = R.POSITIVE  # the reader read it as good news
    reading.verdicts[0].strength = 0.9
    out = S.Outcome(S.Signal("k", 0.0, "", "ZED", +1), "v", 0.0, 0.0, {15: 40.0})
    diffs = S.disagreements([reading], 0.25, [out])
    assert len(diffs) == 1
    assert diffs[0].bot == [("ZED", -1)] and diffs[0].reader == [("ZED", 1)]
    assert diffs[0].outcomes == {"ZED": 40.0}


def test_bot_and_mention_arms_produce_signals_per_token():
    anns = [LISTING, ann("Notice of Removal of Spot Trading Pairs", "OPEN/BTC, SAGA/BTC", code="d")]
    assert [(s.token, s.sign) for s in S.bot_signals(anns, title_bot)] == [("ZED", 1)]
    assert [(s.token, s.sign) for s in S.bot_signals(anns, body_bot)] == [
        ("ZED", 1), ("BNB", 1), ("OPEN", -1), ("SAGA", -1)]
    assert all(s.sign == 1 for s in S.mention_signals(anns)) and len(S.mention_signals(anns)) == 4


def test_measure_uses_the_window_function_and_drops_unmeasurable_signals(tmp_path):
    def fake_window(store, token, when, **kw):
        return None if token == "GHOST" else candles_with_step(30.0, at="after")

    signals = [S.Signal("c", 30 * 60 + 5.0, "", "ZED", +1), S.Signal("c", 30 * 60 + 5.0, "", "GHOST", +1)]
    outcomes = S.measure(Store(tmp_path), signals, horizons=(5,), window_fn=fake_window, workers=2)
    assert [o.signal.token for o in outcomes] == ["ZED"]
    assert abs(outcomes[0].fwd_bps[5] - 30.0) < 1e-6


def test_cli_has_a_listing_command():
    from jevtrade.cli import build_parser

    args = build_parser().parse_args(["listing", "--provider", "mock", "--days", "3"])
    assert args.func.__name__ == "cmd_listing" and args.threshold == 0.25
