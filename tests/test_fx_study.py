"""The FX study: reading central-bank text, graded by the spot tape.

These test the instrument, not the result. Three of them carry most of the
weight. The first is the clock: a Fed timestamp is US Eastern *local*, so a
statement at 2:00 p.m. is 18:00 UTC in September and 19:00 UTC in January, and
an hour of error would put every measurement in the wrong place. The second is
the entry rule: the first bar strictly after the release, so a 400 ms reader is
still up to five minutes late on 5-minute bars. The third is the gap rule: spot
FX is shut all weekend, and a "+60 minute" return measured by bar index across a
Friday close would be a 51-hour return wearing a disguise.

Everything runs offline, from fixtures under ``tests/fixtures/fx``.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jevtrade.fx import baseline as B
from jevtrade.fx import diff as DF
from jevtrade.fx import documents as D
from jevtrade.fx import reader as R
from jevtrade.fx import study as S
from jevtrade.fx import tape as T
from jevtrade.fx.mock import MockFxClient
from jevtrade.listing.store import Store
from jevtrade.types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer

FIXTURES = Path(__file__).parent / "fixtures" / "fx"
FOMC_SEP_TS = 1789581600.0  # 2026-09-16 14:00 EDT = 18:00 UTC


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def doc(title="Federal Reserve issues FOMC statement", body="", issuer="fed",
        kind=D.MONETARY_POLICY, ts=FOMC_SEP_TS, url="https://x/a.htm", speaker=""):
    return D.Document(id=f"{issuer}:{int(ts)}:{abs(hash(title)) % 9999}", issuer=issuer,
                      kind=kind, ts=ts, title=title, body=body, url=url, speaker=speaker,
                      currency=D.ISSUER_CURRENCY.get(issuer, ""))


# ------------------------------------------------------------------- clocks

@pytest.mark.parametrize("stamp,expected", [
    ("9/16/2026 2:00:00 PM", "2026-09-16 18:00"),   # EDT, UTC-4
    ("1/15/2026 2:00:00 PM", "2026-01-15 19:00"),   # EST, UTC-5
    ("3/8/2026 3:00:00 AM", "2026-03-08 07:00"),    # the hour after the spring switch
    ("11/1/2026 1:00:00 AM", "2026-11-01 05:00"),   # the hour before the autumn one
    ("12/31/2025 11:59:00 PM", "2026-01-01 04:59"),
])
def test_fed_timestamps_are_us_eastern_local(stamp, expected):
    ts = D.parse_fed_date(stamp)
    assert datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M") == expected


def test_a_date_with_no_time_of_day_is_refused_rather_than_guessed():
    assert D.parse_fed_date("1/3/2006") is None
    assert D.parse_fed_date("") is None
    assert D.parse_fed_date("not a date 2026") is None


def test_the_dst_fallback_agrees_with_the_tz_database(monkeypatch):
    """The same answers when zoneinfo has no data to read."""
    import builtins

    real_import = builtins.__import__

    def no_tzdata(name, *args, **kwargs):
        if name == "zoneinfo":
            raise ImportError("no tz database in this container")
        return real_import(name, *args, **kwargs)

    stamps = ["9/16/2026 2:00:00 PM", "1/15/2026 2:00:00 PM", "3/8/2026 3:00:00 AM",
              "11/1/2026 1:00:00 AM", "6/1/2026 8:30:00 AM"]
    with_data = [D.parse_fed_date(s) for s in stamps]
    monkeypatch.setattr(builtins, "__import__", no_tzdata)
    assert [D.parse_fed_date(s) for s in stamps] == with_data
    assert D.local_string(FOMC_SEP_TS, "fed") == "2026-09-16 14:00"


def test_the_eu_rule_switches_on_the_last_sunday_of_march_and_october():
    assert D.eu_offset(datetime(2026, 3, 29, 3, 0), 1).total_seconds() == 2 * 3600
    assert D.eu_offset(datetime(2026, 3, 29, 1, 0), 1).total_seconds() == 1 * 3600
    assert D.eu_offset(datetime(2026, 10, 25, 4, 0), 1).total_seconds() == 1 * 3600
    assert D.eu_offset(datetime(2026, 7, 1, 12, 0), 0).total_seconds() == 1 * 3600  # London


@pytest.mark.parametrize("stamp,expected", [
    ("Fri, 18 Sep 2026 10:00:00 +0200", "2026-09-18 08:00"),   # ECB
    ("Fri, 18 Sep 2026 12:40:00 +0900", "2026-09-18 03:40"),   # BoJ
    ("Fri, 18 Sep 2026 12:00:00 +0100", "2026-09-18 11:00"),   # BoE
    ("Thu, 17 Sep 2026 14:00:00 GMT", "2026-09-17 14:00"),
])
def test_rss_pubdates_carry_their_own_offset(stamp, expected):
    ts = D.parse_pubdate(stamp)
    assert datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M") == expected
    assert D.parse_pubdate("last Tuesday") is None


# -------------------------------------------------------------- body + feeds

def test_fed_body_extraction_keeps_the_statement_and_drops_the_chrome():
    text = D.fed_body_text(fixture("fomc-2026-09-16.html"))
    assert text.startswith("The Federal Open Market Committee approved the following")
    assert "raise the target range" in text and "12 – 0 vote" in text
    assert "site chrome" not in text          # the div after the body is not read
    assert "Implementation Note" not in text  # a separate release on a separate page
    assert "For media inquiries" not in text


def test_the_july_statement_keeps_its_dissent_sentences():
    """The vote matters: 9-3 versus 12-0 is the whole story of September."""
    text = D.fed_body_text(fixture("fomc-2026-07-29.html"))
    assert "9 – 3 vote" in text
    assert "Voting against the monetary policy action" in text


def test_generic_body_extraction_takes_paragraphs_and_skips_boilerplate():
    html = (
        "<html><body><p>We use necessary cookies to make our site work and so on.</p>"
        "<p>At its meeting ending on 17 September the Committee voted to maintain Bank Rate.</p>"
        "<p>short</p></body></html>"
    )
    assert D.generic_body_text(html) == (
        "At its meeting ending on 17 September the Committee voted to maintain Bank Rate."
    )


def test_fed_archive_tags_each_row_by_its_type_and_reports_what_it_dropped():
    rows, skipped = D.fed_archive(
        Store(None), since=0.0, until=2e9,
        fetcher=lambda url: fixture(url.rsplit("/", 1)[-1]),
    )
    kinds = sorted({r["kind"] for r in rows})
    assert "monetary_policy" in kinds and "speech" in kinds and "testimony" in kinds
    assert "enforcement_actions" in kinds  # kept here; the default --kinds filter drops it
    assert skipped == 1  # the 2006 row has a date but no time
    assert all(r["issuer"] == "fed" and r["url"].startswith("https://") for r in rows)
    fomc = [r for r in rows if r["title"] == "Federal Reserve issues FOMC statement"]
    assert len(fomc) == 2 and fomc[0]["ts"] != fomc[1]["ts"]


def test_a_date_only_row_outside_the_window_is_not_counted_as_dropped():
    _, skipped = D.fed_archive(
        Store(None), since=FOMC_SEP_TS - 86400, until=FOMC_SEP_TS + 86400,
        fetcher=lambda url: fixture(url.rsplit("/", 1)[-1]),
    )
    assert skipped == 0


@pytest.mark.parametrize("issuer,name,first_title", [
    ("ecb", "ecb-press.xml", "ECB Consumer Expectations Survey"),
    ("boj", "boj-whatsnew.xml", "Change in the Guideline for Money Market Operations"),
    ("boe", "boe-news.xml", "Minutes of the Transaction and Post-trade"),
])
def test_rss_feeds_parse_into_timestamped_rows(issuer, name, first_title):
    rows, skipped = D.rss(Store(None), issuer, "http://x", fetcher=lambda _u: fixture(name))
    assert len(rows) == 3 and skipped == 0
    assert rows[0]["title"].startswith(first_title)
    assert all(r["ts"] > 1.7e9 and r["issuer"] == issuer for r in rows)


def test_rss_kinds_separate_policy_from_the_rest():
    assert D.rss_kind("Change in the Guideline for Money Market Operations") == D.MONETARY_POLICY
    assert D.rss_kind("Monetary policy decisions") == D.MONETARY_POLICY
    assert D.rss_kind("Christine Lagarde: Interview with Ouest-France") == D.SPEECH
    assert D.rss_kind("ECB Consumer Expectations Survey results") == D.PRESS_RELEASE


def _fed_only(store, **kw):
    def body(url):
        if "monetary20260916a" in url:
            return D.fed_body_text(fixture("fomc-2026-09-16.html"))
        if "monetary20260729a" in url:
            return D.fed_body_text(fixture("fomc-2026-07-29.html"))
        return ""

    return D.collect(
        store, since=0.0, until=2e9, issuers=("fed",),
        fed_fetcher=lambda url: fixture(url.rsplit("/", 1)[-1]),
        body_fetcher=body, **kw,
    )


def test_collect_filters_by_kind_sorts_by_time_and_caches_bodies(tmp_path):
    store = Store(tmp_path)
    got = _fed_only(store)
    assert [d.ts for d in got] == sorted(d.ts for d in got)
    assert {d.kind for d in got} <= set(D.POLICY_KINDS)
    assert got.per_issuer == {"fed": len(got.documents)} and got.skipped_no_time == 1
    assert all(d.currency == "USD" for d in got)

    calls = []
    again = D.collect(
        store, since=0.0, until=2e9, issuers=("fed",),
        fed_fetcher=lambda url: fixture(url.rsplit("/", 1)[-1]),
        body_fetcher=lambda url: calls.append(url) or "",
    )
    assert calls == []  # bodies are immutable: cached
    assert len(again) == len(got)


def test_collect_with_no_kind_filter_keeps_the_regulatory_noise():
    got = _fed_only(Store(None), kinds=None)
    assert any(d.kind == "enforcement_actions" for d in got)


# ------------------------------------------------------------------ calendar

def test_calendar_snapshots_accumulate_by_week_and_are_never_back_filled(tmp_path):
    store = Store(tmp_path)
    assert D.calendar_rows(store) == []  # nothing stored: the surprise arm is dark
    rows = json.loads(fixture("ff_calendar.json"))
    key, stored = D.calendar_snapshot(store, now=FOMC_SEP_TS, fetcher=lambda: rows)
    assert key == D.week_key(FOMC_SEP_TS) and len(stored) == len(rows)
    assert len(D.calendar_rows(store)) == len(rows)

    other = [dict(rows[0], title="Next week", date="2026-09-25T14:00:00-04:00")]
    D.calendar_snapshot(store, now=FOMC_SEP_TS + 7 * 86400, fetcher=lambda: other)
    titles = {r["title"] for r in D.calendar_rows(store)}
    assert "Next week" in titles and len(D.calendar_rows(store)) == len(rows) + 1


def test_a_document_matches_the_calendar_row_at_its_own_minute():
    rows = [r for r in (D._calendar_row(r) for r in json.loads(fixture("ff_calendar.json")))]
    statement = doc(ts=FOMC_SEP_TS)
    match = D.match_calendar(statement, rows)
    assert match is not None and match["country"] == "USD" and match["forecast"] == "4.00%"
    assert D.match_calendar(doc(ts=FOMC_SEP_TS + 86400), rows) is None


# ---------------------------------------------------------------------- diff

def test_sentence_splitting_keeps_initials_and_drops_fragments():
    text = ("Voting against were Beth M. Hammack and Lorie K. Logan, who preferred a hike. "
            "Share. Inflation remains elevated for now.")
    out = DF.split_sentences(text)
    assert out[0].startswith("Voting against were Beth M. Hammack and Lorie K. Logan")
    assert out == [out[0], "Inflation remains elevated for now."]


def test_changed_sentences_on_the_two_real_fomc_statements():
    old = D.fed_body_text(fixture("fomc-2026-07-29.html"))
    new = D.fed_body_text(fixture("fomc-2026-09-16.html"))
    pairs = DF.changed_sentences(old, new)
    assert len(pairs) == 7
    assert "9 – 3 vote" in pairs[0][0] and "12 – 0 vote" in pairs[0][1]
    assert any("Inflation remains elevated." == now for _was, now in pairs)
    assert any(was.startswith("Voting against") and now == "" for was, now in pairs)
    assert len(DF.changed_sentences(old, new, limit=3)) == 3
    assert DF.changed_sentences(new, new) == []


def test_title_families_collapse_editions_and_previous_of_picks_the_last_one():
    assert DF.title_family("Minutes of the Federal Open Market Committee, July 28-29, 2026") == \
        DF.title_family("Minutes of the Federal Open Market Committee, June 16-17, 2026")
    jul = doc(ts=FOMC_SEP_TS - 49 * 86400, body="old")
    jun = doc(ts=FOMC_SEP_TS - 91 * 86400, body="older")
    other = doc(title="Minutes of the FOMC, June 2026", ts=FOMC_SEP_TS - 30 * 86400, body="x")
    sep = doc(ts=FOMC_SEP_TS, body="new")
    assert DF.previous_of(sep, [jun, jul, other, sep]) is jul
    assert DF.previous_of(jun, [jun, jul, sep]) is None
    # a different issuer with the same title is not a predecessor
    assert DF.previous_of(sep, [doc(ts=jul.ts, issuer="ecb", body="x")]) is None


# -------------------------------------------------------------- sign convention

@pytest.mark.parametrize("currency,stance,expected", [
    ("USD", R.HAWKISH, ("EURUSD=X", -1)),   # hawkish Fed: dollar up, EURUSD down
    ("USD", R.DOVISH, ("EURUSD=X", +1)),
    ("EUR", R.HAWKISH, ("EURUSD=X", +1)),
    ("EUR", R.DOVISH, ("EURUSD=X", -1)),
    ("GBP", R.HAWKISH, ("GBPUSD=X", +1)),
    ("JPY", R.HAWKISH, ("JPY=X", -1)),      # JPY=X is USDJPY: a strong yen is down
    ("JPY", R.DOVISH, ("JPY=X", +1)),
])
def test_the_sign_table_says_hawkish_means_the_issuers_currency_strengthens(
    currency, stance, expected
):
    assert R.signed_pair(currency, stance) == expected


def test_a_neutral_stance_or_an_unmapped_currency_has_no_side():
    assert R.signed_pair("USD", R.NEUTRAL) is None
    assert R.signed_pair("CHF", R.HAWKISH) is None
    assert set(R.PAIRS) == set(D.ISSUER_CURRENCY.values())


# -------------------------------------------------------------------- reader

def scripted(stance=R.HAWKISH, p_stance=0.8, level=3.0, relevant=0.9, new_information=0.9,
             surprise=0.8, unaffected=0.1, material=0.9, intervention=0.0, confidence=0.9,
             reversed_agrees=True):
    """A client whose answers come from whatever question map it is handed."""

    class Client:
        provider = "test"
        model = "scripted"

        def __init__(self):
            self.questions = None
            self.calls = 0
            self.seen = []

        def evaluate(self, state):
            self.calls += 1
            self.seen.append((state, dict(self.questions)))
            answers = {}
            for key, q in self.questions.items():
                options = list(q["criteria"])
                if q["type"] == "choice":
                    if key in (R.STANCE, R.KIND) or key.startswith("diff_stance_"):
                        pick = stance if stance in options else options[0]
                        if key == R.KIND:
                            pick = R.RATE_DECISION
                    elif key == R.STANCE_REVERSED:
                        want = {R.HAWKISH: R.BUY, R.DOVISH: R.SELL}.get(stance, R.NEITHER)
                        pick = want if reversed_agrees else R.NEITHER
                    else:
                        pick = options[0]
                    rest = (1.0 - p_stance) / max(len(options) - 1, 1)
                    probs = {o: (p_stance if o == pick else rest) for o in options}
                    answers[key] = ChoiceAnswer(pick, probs, confidence)
                elif q["type"] == "score":
                    value = intervention if key == R.INTERVENTION_TIER else level
                    answers[key] = ScoreAnswer(
                        value, {str(i): t for i, t in enumerate(options)},
                        {str(i): 1 / len(options) for i in range(len(options))}, 0.5)
                elif key == R.POLICY_RELEVANT:
                    answers[key] = NoulAnswer(relevant)
                elif key == R.NEW_INFORMATION:
                    answers[key] = NoulAnswer(new_information)
                elif key == R.SURPRISE:
                    answers[key] = NoulAnswer(surprise)
                elif key == R.HOLDER_UNAFFECTED:
                    answers[key] = NoulAnswer(unaffected)
                elif key.startswith("diff_material_"):
                    answers[key] = NoulAnswer(material)
                else:
                    answers[key] = NoulAnswer(0.2)
            return JevResponse("scripted", answers, 1000, 0, 400.0, "test")

    return Client()


CHANGES = [("Inflation remains elevated relative to the goal.", "Inflation remains elevated."),
           ("Voting against were three members.", "")]


def test_round_one_asks_everything_in_one_request_and_width_grows_with_the_diff():
    bare = R.round_one_questions("USD", [])
    assert set(bare) == {R.KIND, R.POLICY_RELEVANT, R.STANCE, R.NEW_INFORMATION,
                         R.MAGNITUDE, R.GUIDANCE_CHANGED, R.SURPRISE, R.INTERVENTION_TIER}
    wide = R.round_one_questions("USD", [("a", "b")] * 12)
    assert len(wide) == len(bare) + 24 == 32
    assert wide[R.MAGNITUDE]["type"] == "score" and len(wide[R.MAGNITUDE]["criteria"]) == 4
    assert len(wide[R.INTERVENTION_TIER]["criteria"]) == 5
    assert set(wide[R.STANCE]["criteria"]) == {R.HAWKISH, R.DOVISH, R.NEUTRAL}
    assert "USD" in wide[R.STANCE]["instructions"]


def test_the_reader_caps_the_diff_at_max_diffs():
    client = scripted()
    reading = R.Reader(client, max_diffs=3).read(doc(body="b"), changes=[("a", "b")] * 9)
    assert len(reading.diffs) == 3
    assert R.diff_stance_key(3) not in client.seen[0][1]


def test_round_two_runs_on_a_decisive_stance_and_confirms_it_with_a_reversed_framing():
    client = scripted(stance=R.HAWKISH, p_stance=0.8, unaffected=0.1)
    reading = R.Reader(client).read(doc(body="b"), changes=CHANGES)
    assert client.calls == 2 and reading.rounds == 2
    assert reading.latency_ms == 800.0
    assert reading.confirm == pytest.approx((1 - 0.1) * 0.8)
    assert reading.verdict.pair == "EURUSD=X" and reading.verdict.sign == -1
    second_state, second_questions = client.seen[1]
    assert set(second_questions) == {R.HOLDER_UNAFFECTED, R.STANCE_REVERSED, R.HORIZON}
    assert second_state["round"] == 2 and "first_round_verdicts" in second_state


def test_no_second_round_means_no_signal():
    client = scripted(stance=R.NEUTRAL, material=0.1, intervention=0.0)
    reading = R.Reader(client).read(doc(body="b"), changes=CHANGES)
    assert client.calls == 1 and reading.rounds == 1 and reading.confirm == 0.0
    assert reading.verdict is None and reading.signals(0.0) == []


def test_a_material_sentence_change_opens_round_two_even_when_the_tone_is_flat():
    client = scripted(stance=R.NEUTRAL, material=0.9)
    assert R.Reader(client).read(doc(body="b"), changes=CHANGES).rounds == 2
    # ... and so does intervention language, with nothing else going on
    loud = scripted(stance=R.NEUTRAL, material=0.1, intervention=4.0)
    assert R.Reader(loud).read(doc(body="b")).rounds == 2


def test_verdict_strength_is_arithmetic_on_the_model_probabilities():
    reading = R.Reader(scripted(level=3.0, relevant=1.0, new_information=1.0,
                                surprise=1.0, unaffected=0.0)).read(doc(body="b"))
    assert reading.verdict.strength == pytest.approx(0.8 * 0.8 * 1.0 * 1.0 * 1.0 * 1.0)

    damped = R.Reader(scripted(new_information=0.0, surprise=0.0)).read(doc(body="b"))
    assert damped.verdict.strength < 0.5 * reading.verdict.strength
    # a reversed framing that picks the other side collapses the confirmation
    disagreeing = R.Reader(scripted(reversed_agrees=False)).read(doc(body="b"))
    assert disagreeing.confirm < 0.15
    assert disagreeing.verdict.strength < 0.2 * reading.verdict.strength


def test_low_confidence_or_a_weak_verdict_never_becomes_a_signal():
    shaky = R.Reader(scripted(confidence=0.3)).read(doc(body="b"))
    assert shaky.verdict.strength > 0 and shaky.signals(0.0, min_confidence=0.5) == []
    weak = R.Reader(scripted(level=0.0)).read(doc(body="b"))
    assert weak.verdict.strength == 0.0 and weak.signals(0.0) == []


def test_the_state_carries_the_clock_the_diff_and_the_calendar():
    client = scripted()
    row = {"title": "Federal Funds Rate", "forecast": "4.00%", "previous": "3.75%",
           "impact": "High", "ts": FOMC_SEP_TS}
    R.Reader(client, body_chars=10).read(
        doc(body="a very long body indeed"), changes=CHANGES,
        previous=doc(ts=FOMC_SEP_TS - 49 * 86400), calendar=row)
    state = client.seen[0][0]
    assert state["published_utc"] == "2026-09-16 18:00 UTC"
    assert state["published_local"] == "2026-09-16 14:00"
    assert state["body"] == "a very lon"
    assert state["calendar"]["forecast"] == "4.00%"
    assert [c["i"] for c in state["changed_sentences"]] == [1, 2]
    assert "compared_with" in state


# ---------------------------------------------------------------- mock client

def test_mock_client_answers_every_question_with_the_right_type():
    client = MockFxClient()
    changes = CHANGES
    maps = [R.round_one_questions("USD", changes), R.round_two_questions("USD")]
    for questions in maps:
        client.questions = questions
        response = client.evaluate(R.build_state(
            doc(body="The Committee decided to raise the target range."),
            changes, round_no=1, body_chars=500))
        assert set(response.answers) == set(questions)
        for key, question in questions.items():
            assert response.answers[key].type == question["type"], key


def test_the_mock_reads_hawkish_and_dovish_words_and_nothing_else():
    hawk = R.Reader(MockFxClient()).read(
        doc(body="The Committee decided to raise the target range further. "
                 "Inflation remains elevated and policy will stay restrictive."))
    assert hawk.stance == R.HAWKISH and hawk.verdict.sign == -1  # short EURUSD
    dove = R.Reader(MockFxClient()).read(
        doc(body="The Committee decided to cut and to ease policy; downside risks "
                 "have grown and the stance is accommodative."))
    assert dove.stance == R.DOVISH and dove.verdict.sign == +1
    assert dove.rounds == 2 and dove.kind == R.RATE_DECISION


def test_the_mock_climbs_the_intervention_ladder_on_phrases():
    steps = [
        ("nothing about currencies at all", 0.0),
        ("we are watching the exchange rate", 0.25),
        ("recent moves have been excessive and one-sided", 0.5),
        ("we stand ready to take decisive action", 0.75),
        ("the authorities intervened in the market", 1.0),
    ]
    for body, expected in steps:
        reading = R.Reader(MockFxClient()).read(doc(title="Statement", body=body))
        assert reading.intervention == pytest.approx(expected), body


def test_the_mock_is_deterministic():
    d = doc(body="raise rates, inflation remains elevated")
    first = R.Reader(MockFxClient()).read(d)
    second = R.Reader(MockFxClient()).read(d)
    assert first.verdict == second.verdict and first.stance == second.stance


# --------------------------------------------------------------------- tape

def synthetic_bars(n=40, bar_min=5, step_bps=0.0, at=None, start=0, skip=()):
    """Flat at 100 with one optional jump; ``skip`` drops bar indices to make a gap."""
    ts, opens, closes = [], [], []
    price = 100.0
    for i in range(n):
        if i in skip:
            continue
        ts.append(start + bar_min * 60 * i)
        opens.append(price)
        if at is not None and i == at:
            price *= 2.718281828459045 ** (step_bps / 1e4)
        closes.append(price)
    return T.Bars("TEST=X", bar_min, ts, opens, closes)


def test_entry_is_the_open_of_the_first_bar_after_the_release():
    # The timestamp falls inside bar 8, so bar 8 is the release bar and bar 9 is
    # the entry; the move happens across bar 9, which is what the arm gets paid.
    series = synthetic_bars(at=9, step_bps=50.0)
    pre, bar, fwd = T.forward_returns(series, 8 * 300 + 30.0, +1, (5, 15, 60))
    assert pre == 0.0 and bar == 0.0
    assert all(v == pytest.approx(50.0, abs=1e-6) for v in fwd.values())
    _, _, short = T.forward_returns(series, 8 * 300 + 30.0, -1, (15,))
    assert short[15] == pytest.approx(-50.0, abs=1e-6)


def test_a_move_inside_the_release_bar_is_never_credited():
    series = synthetic_bars(at=8, step_bps=50.0)
    pre, bar, fwd = T.forward_returns(series, 8 * 300 + 30.0, +1, (5, 30))
    assert bar == pytest.approx(50.0, abs=1e-6)  # reported ...
    assert fwd == {5: 0.0, 30: 0.0}              # ... and paid to nobody


def test_horizons_round_up_to_whole_bars():
    series = synthetic_bars(at=9, step_bps=50.0)
    _, _, fwd = T.forward_returns(series, 8 * 300 + 1.0, +1, (1, 4, 5, 6))
    # 1 and 4 minutes both land on the first 5-minute bar after entry
    assert fwd[1] == fwd[4] == fwd[5] == pytest.approx(50.0, abs=1e-6)
    assert fwd[6] == pytest.approx(50.0, abs=1e-6)


def test_a_weekend_or_any_gap_inside_the_window_drops_the_signal():
    """The Friday close: a '+60m' return across it would be a 51-hour return."""
    friday = synthetic_bars(n=20, skip=range(12, 20))  # tape stops after bar 11
    monday = T.Bars("TEST=X", 5, friday.ts + [friday.ts[-1] + 48 * 3600],
                    friday.open + [100.0], friday.close + [101.0])
    when = 10 * 300 + 30.0
    assert T.forward_returns(monday, when, +1, (5,)) is not None
    assert T.forward_returns(monday, when, +1, (60,)) is None   # the window spans the weekend
    # and a timestamp inside the closed session has no release bar at all
    assert T.forward_returns(monday, friday.ts[-1] + 10 * 3600, +1, (5,)) is None


def test_a_release_with_no_bars_before_it_is_dropped():
    series = synthetic_bars(n=30)
    assert T.forward_returns(series, 1 * 300 + 10.0, +1, (5,)) is None  # no 15m of history
    assert T.forward_returns(series, series.ts[-1] + 10, +1, (5,)) is None


def test_yahoo_payloads_drop_null_prints_and_measure_the_real_fomc_move(tmp_path, monkeypatch):
    payload = json.loads(fixture("yahoo-eurusd-5m.json"))
    stamps = payload["chart"]["result"][0]["timestamp"]
    monkeypatch.setattr(T, "get_json", lambda url, *a, **k: payload)
    rows = T.fetch_chart("EURUSD=X")
    assert len(rows) == len(stamps) - 1  # one null close in the fixture is dropped

    series = T.bars(Store(tmp_path), "EURUSD=X", fetcher=lambda *_a: rows)
    assert series.bar_min == 5 and len(series) == len(rows)
    # A hawkish Fed is short EURUSD; the September statement paid at 15 minutes
    # and cost a little at five.
    _, _, fwd = T.forward_returns(series, FOMC_SEP_TS + 30, -1, (5, 15, 30, 60))
    assert fwd[5] < 0 and fwd[15] > 5 and fwd[30] > 20 and fwd[60] > 20


def test_bars_are_cached_per_symbol_and_day(tmp_path):
    calls = []

    def fetch(symbol, interval, range_):
        calls.append((symbol, interval, range_))
        return [(i * 300, 100.0, 100.0) for i in range(10)]

    store = Store(tmp_path)
    assert T.bars(store, "JPY=X", fetcher=fetch) is not None
    assert T.bars(store, "JPY=X", fetcher=fetch) is not None
    assert calls == [("JPY=X", "5m", "60d")]
    assert T.bars(store, "JPY=X", fetcher=lambda *_a: []) is not None  # cached, not refetched


# ------------------------------------------------------------------ baselines

def test_keyword_bot_trades_hawkish_minus_dovish_on_the_issuers_pair():
    hawk = B.keyword_bot(doc(body="raise, hike, tighten; inflation remains elevated"))
    assert [(s.pair, s.sign) for s in hawk] == [("EURUSD=X", -1)]
    dove = B.keyword_bot(doc(issuer="boj", body="cut, ease, accommodative, downside risks"))
    assert [(s.pair, s.sign) for s in dove] == [("JPY=X", +1)]
    assert B.keyword_bot(doc(body="the committee met and adjourned")) == []
    assert B.keyword_bot(doc(body="raise and cut")) == []  # a tie is no opinion


def test_keyword_bot_is_fooled_by_the_word_and_not_the_meaning():
    """The documented failure of counting: a negation does not register."""
    fooled = "The Committee no longer expects to raise rates and will not tighten further."
    assert B.word_lean(fooled)[0] == "hawkish"  # the meaning is the opposite


@pytest.mark.parametrize("text,expected", [
    ("The Committee decided to raise the target range for the federal funds rate "
     "by 1/4 percentage point to 3-3/4 to 4 percent.", 4.0),
    ("The Committee decided to maintain the target range for the federal funds rate "
     "at 3-1/2 to 3-3/4 percent.", 3.75),
    ("Bank Rate was maintained to 3.75%", 3.75),
])
def test_announced_rate_reads_ranges_and_fractions(text, expected):
    assert B.announced_rate(text) == expected


def test_surprise_bot_is_dark_without_a_snapshot_and_trades_the_beat_with_one():
    body = ("The Committee decided to raise the target range for the federal funds rate "
            "by 1/4 percentage point to 3-3/4 to 4 percent.")
    statement = doc(body=body, ts=FOMC_SEP_TS)
    assert B.surprise_bot(statement, []) == []  # no calendar: no signal, ever

    rows = [{"title": "Federal Funds Rate", "country": "USD", "impact": "High",
             "forecast": "3.75%", "previous": "3.75%", "ts": FOMC_SEP_TS}]
    hit = B.surprise_bot(statement, rows)
    assert [(s.pair, s.sign) for s in hit] == [("EURUSD=X", -1)]  # 4.00 beat 3.75: hawkish

    inline = [dict(rows[0], forecast="4.00%")]
    assert B.surprise_bot(statement, inline) == []  # in line with the forecast: nothing
    dovish = [dict(rows[0], forecast="4.25%")]
    assert [(s.sign) for s in B.surprise_bot(statement, dovish)] == [+1]
    assert B.surprise_bot(doc(title="Speech on the outlook", body=body), rows) == []


# --------------------------------------------------------------------- study

def test_null_signals_keep_the_pair_and_side_and_land_only_where_the_tape_is():
    series = synthetic_bars(n=600, start=1_000_000)
    tape = S.Tape(Store(None), loader=lambda *a, **k: series)
    signal = S.Signal("c", 1_000_000 + 300 * 300, "t", "TEST=X", -1)
    outcome = S.Outcome(signal, "TEST=X", 0.0, 0.0, {15: 0.0})
    nulls = S.null_signals(tape, [outcome], per=5, span_days=2, exclude_h=3, horizons=(15,))
    assert len(nulls) == 5
    assert all(n.pair == "TEST=X" and n.sign == -1 for n in nulls)
    assert all(3 * 3600 <= abs(n.ts - signal.ts) <= 2 * 86400 for n in nulls)
    assert all(T.forward_returns(series, n.ts, 1, (15,)) is not None for n in nulls)


def test_measure_drops_signals_the_tape_cannot_cover():
    series = synthetic_bars(n=40, at=9, step_bps=30.0)
    tape = S.Tape(Store(None), loader=lambda _s, symbol, **k: None if symbol == "JPY=X" else series)
    signals = [S.Signal("c", 8 * 300 + 5.0, "", "TEST=X", +1),
               S.Signal("c", 8 * 300 + 5.0, "", "JPY=X", +1)]
    outcomes = S.measure(tape, signals, horizons=(5,), workers=2)
    assert [o.signal.pair for o in outcomes] == ["TEST=X"]
    assert outcomes[0].fwd_bps[5] == pytest.approx(30.0, abs=1e-6)


def test_summarize_reports_hit_rate_and_z_against_the_null():
    signal = S.Signal("c", 0.0, "t", "EURUSD=X", +1)
    outcomes = [S.Outcome(signal, "EURUSD=X", 0.0, 0.0, {15: v}) for v in (30.0, 20.0, -5.0, 25.0)]
    nulls = [S.Outcome(signal, "EURUSD=X", 0.0, 0.0, {15: v}) for v in (1.0, -2.0, 0.0, 1.0)]
    summary = S.summarize("x", [signal] * 6, outcomes, nulls, horizons=(15,))
    assert summary.signals == 6 and summary.measured == 4
    assert summary.hit[15] == 0.75 and summary.measurable == pytest.approx(4 / 6)
    assert summary.fwd[15].mean == 17.5 and summary.z[15] > 2


def test_the_arms_produce_one_signal_per_document_at_most():
    docs = [doc(body="raise and tighten, inflation remains elevated"),
            doc(title="Speech", kind=D.SPEECH, body="the committee met", ts=FOMC_SEP_TS + 60)]
    assert len(S.bot_signals(docs, S.keyword_bot)) == 1
    everything = S.all_text_signals(docs)
    assert len(everything) == 2  # the tied document still gets a side, by convention
    assert everything[1].sign == -1


def test_read_all_caches_readings_computes_the_diff_and_gives_each_worker_a_reader(tmp_path):
    made = []

    def make():
        client = scripted()
        made.append(client)
        return R.Reader(client)

    old = D.fed_body_text(fixture("fomc-2026-07-29.html"))
    new = D.fed_body_text(fixture("fomc-2026-09-16.html"))
    docs = [doc(body=old, ts=FOMC_SEP_TS - 49 * 86400), doc(body=new, ts=FOMC_SEP_TS)]
    docs += [doc(title=f"Speech {i}", kind=D.SPEECH, body="words", ts=FOMC_SEP_TS + i)
             for i in range(4)]
    store = Store(tmp_path)
    first = S.read_all(docs, make, store=store, cache_tag="t", workers=3)
    assert len(first) == 6 and 1 <= len(made) <= 3
    assert len(first[1].diffs) == 7  # the September statement was diffed against July
    calls = sum(c.calls for c in made)

    second = S.read_all(docs, make, store=store, cache_tag="t", workers=3)
    assert sum(c.calls for c in made) == calls  # nothing asked again
    assert [r.verdict for r in second] == [r.verdict for r in first]
    assert [len(r.diffs) for r in second] == [len(r.diffs) for r in first]
    assert S.cost_usd(second) > 0


def test_a_new_tree_version_invalidates_the_cache(tmp_path):
    store = Store(tmp_path)
    client = scripted()
    docs = [doc(body="b")]
    S.read_all(docs, lambda: R.Reader(client), store=store, cache_tag="mock:v1")
    S.read_all(docs, lambda: R.Reader(client), store=store, cache_tag="mock:v1")
    assert client.calls == 2  # one document, two rounds, asked once
    S.read_all(docs, lambda: R.Reader(client), store=store, cache_tag="mock:v2")
    assert client.calls == 4


def test_disagreements_show_where_counting_and_reading_differ():
    statement = doc(body="cut and ease; downside risks dominate", ts=FOMC_SEP_TS)
    reading = R.Reader(scripted(stance=R.HAWKISH)).read(statement)
    outcome = S.Outcome(S.Signal(statement.id, FOMC_SEP_TS, "", "EURUSD=X", -1),
                        "EURUSD=X", 0.0, 0.0, {15: 12.0})
    diffs = S.disagreements([reading], 0.05, [outcome])
    assert len(diffs) == 1
    assert diffs[0].bot == [("EURUSD=X", +1)] and diffs[0].reader == [("EURUSD=X", -1)]
    assert diffs[0].outcomes == {"EURUSD=X": -12.0}
    assert diffs[0].issuer == "fed"


def test_a_reading_survives_a_round_trip_through_the_cache_format():
    reading = R.Reader(scripted()).read(doc(body="b"), changes=CHANGES)
    back = S.reading_from_dict(reading.document, json.loads(json.dumps(S.reading_to_dict(reading))))
    assert back.verdict == reading.verdict and back.stance == reading.stance
    assert [d.now for d in back.diffs] == [d.now for d in reading.diffs]
    assert back.questions_asked == reading.questions_asked


# ----------------------------------------------------------------------- CLI

def test_cli_has_an_fx_command_with_the_documented_defaults():
    from jevtrade.cli import build_parser

    args = build_parser().parse_args(["fx", "--provider", "mock"])
    assert args.func.__name__ == "cmd_fx"
    assert args.days == 60.0 and args.bar == 5 and args.horizons == "5,15,30,60"
    assert args.issuers == "fed,ecb,boj,boe"
    assert args.kinds == "monetary_policy,speech,testimony"
    assert args.cache == ".cache/fx" and args.threshold == 0.15


def test_cli_fx_runs_end_to_end_offline(tmp_path, monkeypatch, capsys):
    """The whole command with mock answers and fixture feeds -- no network."""
    from jevtrade import cli
    from jevtrade.fx import documents as docs_module
    from jevtrade.fx import tape as tape_module

    def get_text(url, timeout=30.0):
        name = url.rsplit("/", 1)[-1]
        if name in {"ne-press.json", "ne-speeches.json", "ne-testimony.json"}:
            return fixture(name)
        if "monetary20260916a" in name:
            return fixture("fomc-2026-09-16.html")
        if "monetary20260729a" in name:
            return fixture("fomc-2026-07-29.html")
        return "<html><body><p>" + "no body for this one at all, but long enough.</p></body></html>"

    payload = json.loads(fixture("yahoo-eurusd-5m.json"))
    monkeypatch.setattr(docs_module, "get_text", get_text)
    monkeypatch.setattr(docs_module, "fetch_body",
                        lambda url: docs_module.body_text(url, get_text(url)))
    monkeypatch.setattr(tape_module, "get_json", lambda url, *a, **k: payload)
    monkeypatch.setattr(docs_module, "fetch_calendar",
                        lambda: json.loads(fixture("ff_calendar.json")))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    out_file = tmp_path / "run.json"
    code = cli.main([
        "fx", "--provider", "mock", "--issuers", "fed", "--days", "365",
        "--cache", str(tmp_path / "cache"), "--snapshot-calendar",
        "--workers", "2", "--workers-io", "2", "--out", str(out_file),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "documents" in printed and "reader (mock)" in printed
    assert "keyword-bot" in printed and "all text" in printed and "surprise-bot" in printed
    assert "rows dropped for having no time of day" in printed
    assert re.search(r"reader >=0\.\d\d", printed)

    record = json.loads(out_file.read_text())
    assert record["provider"] == "mock" and record["bar_min"] == 5
    assert record["documents"] >= 4 and len(record["readings"]) == record["documents"]
    assert {a["name"] for a in record["arms"]} >= {"keyword-bot", "all text"}
    assert record["calendar_weeks"]


def test_cli_fx_says_so_and_fails_when_the_window_is_empty(tmp_path, monkeypatch, capsys):
    from jevtrade import cli
    from jevtrade.fx import documents as docs_module

    monkeypatch.setattr(docs_module, "get_text",
                        lambda url, timeout=30.0: fixture(url.rsplit("/", 1)[-1]))
    monkeypatch.setattr(docs_module, "fetch_body", lambda url: "")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    code = cli.main(["fx", "--provider", "mock", "--issuers", "fed", "--days", "0.001",
                     "--cache", str(tmp_path / "cache")])
    assert code == 1
    assert "no documents in range" in capsys.readouterr().err


# --- the surprise bot must read the rate, not the vote or the target -------------

BOE_SEPTEMBER = (
    "At its meeting ending on 16 September 2026, the Monetary Policy Committee (MPC) "
    "voted by a majority of 6–3 to maintain Bank Rate at 3.75%. Three members voted to "
    "increase Bank Rate by 0.25 percentage points, to 4%. UK CPI inflation increased to "
    "3.1% in August. Monetary policy is being set to ensure inflation comes down to 2% "
    "sustainably. 36% of firms expect... 78% of respondents..."
)


def test_announced_rate_ignores_the_target_the_dissent_and_the_survey_percentages():
    title = "Bank rate maintained at 3.75% - September 2026 Monetary Policy Summary and Minutes"
    assert B.announced_rate(f"{title}\n{BOE_SEPTEMBER}") == 3.75
    # the title alone carries it; the body alone carries it too, via the decision sentence
    assert B.announced_rate(title) == 3.75
    assert B.announced_rate(f"\n{BOE_SEPTEMBER}") == 3.75
    # a page of percentages with no decision sentence answers nothing
    assert B.announced_rate("\nInflation is 3.1%. The target is 2%. Growth was 0.5%.") is None


def test_calendar_match_prefers_the_rate_row_over_the_votes_row():
    rows = [
        {"country": "GBP", "ts": 1000.0, "title": "MPC Official Bank Rate Votes",
         "forecast": "3-0-6", "previous": "3-0-6"},
        {"country": "GBP", "ts": 1000.0, "title": "Monetary Policy Summary",
         "forecast": "", "previous": ""},
        {"country": "GBP", "ts": 1000.0, "title": "Official Bank Rate",
         "forecast": "3.75%", "previous": "3.75%"},
        {"country": "USD", "ts": 1000.0, "title": "Federal Funds Rate",
         "forecast": "4.00%", "previous": "3.75%"},
    ]
    boe = D.Document(id="boe-1", issuer="boe", kind="monetary_policy", ts=1060.0,
                     title="Bank rate maintained at 3.75% - September 2026", body=BOE_SEPTEMBER,
                     url="u", currency="GBP")
    assert D.match_calendar(boe, rows)["title"] == "Official Bank Rate"
    # in line with the forecast: the surprise bot stays out, instead of shorting on "3-0-6"
    assert B.surprise_bot(boe, rows) == []
    beat = [dict(r, forecast="3.50%") if r["title"] == "Official Bank Rate" else r for r in rows]
    assert [s.sign for s in B.surprise_bot(boe, beat)] == [+1]  # 3.75 printed vs 3.50 expected
    # a votes-only snapshot never yields a signal
    votes_only = [r for r in rows if r["title"] != "Official Bank Rate"]
    assert B.surprise_bot(boe, votes_only) == []


def test_is_percent_tells_rates_from_votes():
    assert B._is_percent("4.00%") and B._is_percent("<1.25%")
    assert not B._is_percent("3-0-6") and not B._is_percent("") and not B._is_percent(None)
