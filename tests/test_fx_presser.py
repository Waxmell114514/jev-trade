"""The press conference, thirty minutes after the statement.

Three things here can break quietly and each has its own test. The **split**
between the opening remarks and the Q&A depends on a marker style that has
changed three times since 2011, so all three styles are exercised. The **start
time** is a rule, so the rule is asserted rather than trusted. And the whole
arm rests on an optional dependency, so the case where it is missing has to end
in the word *dark* and not in a traceback.

Everything runs offline: the transcripts below are cut from the real ones and
no PDF ever enters the repository -- ``pdf_text`` is stubbed where a PDF would
be, and the one test that cares about ``pypdf`` removes it from ``sys.modules``.
"""

import json
import sys
from datetime import datetime, timezone

import pytest

from jevtrade.fx import documents as D
from jevtrade.fx import presser as P
from jevtrade.fx import reader as R
from jevtrade.fx.mock import MockFxClient
from jevtrade.fx.reader import Reader
from jevtrade.listing.store import Store
from jevtrade.types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer

DAY = 86400.0

# 2011: the questioner is not named, the running header has no word "Page".
BERNANKE = """April 27, 2011 Chairman Bernanke's Press Conference FINAL
1 of 26

Transcript of Chairman Bernanke's Press Conference
April 27, 2011

CHAIRMAN BERNANKE.  Good afternoon.  Welcome.  In my opening remarks, I'd like to
briefly first review today's policy decision.  The Committee decided to complete
its purchases of 600 billion dollars of longer-term Treasury securities by the end
of the current quarter.  Inflation has picked up, and we are watching it closely.
April 27, 2011 Chairman Bernanke's Press Conference FINAL
2 of 26
Thank you again, and I'd be glad to take your questions.
QUESTION.  Mr. Chairman, tomorrow we're going to get a pretty weak first-quarter
GDP number.  What do you see as the causes of the weak growth?
CHAIRMAN BERNANKE.  Well, we have had some temporary factors, and we expect
growth to pick up.
QUESTION.  Is the market pricing the path correctly?
CHAIRMAN BERNANKE.  I would not want to endorse that characterization.
"""

# 2015: the first reporter's own name is the marker, no moderator.
YELLEN = """September 17, 2015 Chair Yellen's Press Conference FINAL

Page 1 of 23

Transcript of Chair Yellen's Press Conference
September 17, 2015

CHAIR YELLEN.  Good afternoon.  As you know from our policy statement released a
short time ago, the Committee reaffirmed the current target range.  Inflation has
continued to run below our longer-run objective, and downside risks have increased.
Thank you.  Let me stop there.  I'll be happy to take your questions.
STEVE LIESMAN.  Steve Liesman, CNBC.  Madam Chair, this notion of uncertainty in
global developments -- is it fair to say it could be many months?
CHAIR YELLEN.  It could be, although I would not want to put a timeline on it.
ANN SAPHIR.  Ann Saphir, Reuters.  Are markets pricing the path correctly?
CHAIR YELLEN.  That is not what the Committee has said.
"""

# 2024: a press officer calls the first reporter, and her whole turn is a name.
POWELL = """September 18, 2024 Chair Powell's Press Conference FINAL
Page 1 of 26

Transcript of Chair Powell's Press Conference
September 18, 2024

CHAIR POWELL.  Good afternoon.  My colleagues and I remain squarely focused on our
dual-mandate goals.  Today the Committee decided to lower the target range by half
a percentage point.  Inflation has eased substantially, and the labor market has
cooled.  Thank you.  I look forward to your questions.
MICHELLE SMITH.  Steve.
STEVE LIESMAN.  Steve Liesman, CNBC.  Thank you, Mr. Chairman, for taking our
questions.  In July, you said you weren't necessarily thinking about a 50.
CHAIR POWELL.  We made a good strong start, and that is what you see today.
NICK TIMIRAOS.  Nick Timiraos, Wall Street Journal.  Is that the new pace?
CHAIR POWELL.  You should not look at this as the new pace.
"""


def eastern(year, month, day, hour, minute):
    """An epoch second from a US Eastern wall clock, via the module's own offset."""
    naive = datetime(year, month, day, hour, minute)
    return (naive.replace(tzinfo=timezone.utc) - D.us_eastern_offset(naive)).timestamp()


# ------------------------------------------------------------------ the text


def test_running_headers_are_dropped_and_speech_is_not():
    body = P.strip_headers(BERNANKE)
    assert "Press Conference FINAL" not in body
    assert "1 of 26" not in body and "2 of 26" not in body
    assert "Page 1 of 23" not in P.strip_headers(YELLEN)
    assert "Good afternoon" in body and "weak first-quarter" in body


def test_the_2011_question_marker_splits_the_transcript():
    split = P.split_transcript(BERNANKE, date="20110427")
    assert split.style == "question" and split.marker == "QUESTION"
    assert split.remarks.startswith("CHAIRMAN BERNANKE.")
    assert "I'd be glad to take your questions" in split.remarks
    assert "QUESTION" not in split.remarks
    assert split.qa.startswith("QUESTION.")
    assert "weak first-quarter" in split.qa


def test_a_named_reporter_splits_the_transcript():
    split = P.split_transcript(YELLEN, date="20150917")
    assert split.style == "reporter" and split.marker == "STEVE LIESMAN"
    assert "happy to take your questions" in split.remarks
    assert "STEVE LIESMAN" not in split.remarks
    assert split.qa.startswith("STEVE LIESMAN.")
    assert split.speakers == 2  # Liesman and Saphir; the Chair is not a speaker here


def test_a_moderator_handing_over_splits_the_transcript():
    split = P.split_transcript(POWELL, date="20240918")
    assert split.style == "moderator" and split.marker == "MICHELLE SMITH"
    assert "I look forward to your questions" in split.remarks
    assert "MICHELLE SMITH" not in split.remarks
    assert split.qa.startswith("MICHELLE SMITH.")
    assert "Steve Liesman" in split.qa


# June 2024: the opening marker has ONE space after the stop, and a parser that
# insists on two skips it, starts the remarks somewhere in the middle of the Q&A
# and reports a thousand-character opening statement.
ONE_SPACE = """June 12, 2024 Chair Powell's Press Conference FINAL
Page 1 of 30

CHAIR POWELL. Good afternoon.  My colleagues and I remain squarely focused on
achieving our dual-mandate goals.  Inflation has eased substantially but is still
too high.  Thank you, and I look forward to your questions.
MICHELLE SMITH.  Steve.
STEVE LIESMAN.  Thank you, Mr. Chairman.  Steve Liesman, CNBC.  Just wondering
if you could walk me through the Committee's average inflation forecast.
CHAIR POWELL.  So the forecast is a forecast.
"""


def test_a_single_space_after_the_marker_still_starts_a_turn():
    split = P.split_transcript(ONE_SPACE, date="20240612")
    assert split.remarks.startswith("CHAIR POWELL. Good afternoon.")
    assert "I look forward to your questions" in split.remarks
    assert split.style == "moderator" and split.marker == "MICHELLE SMITH"
    assert len(split.remarks) > 200


def test_a_sentence_that_ends_a_line_is_not_a_speaker():
    """One-word capitals with one space after the stop stay prose."""
    body = ("CHAIR POWELL.  We discussed this at the\n"
            "FOMC. The Committee then turned to the balance sheet, and to the\n"
            "U.S. Treasury market.\n"
            "STEVE LIESMAN.  A question about that.\n")
    marks = [name for _s, _e, name in P.speaker_marks(body)]
    assert marks == ["CHAIR POWELL", "STEVE LIESMAN"]


def test_the_chairs_own_marker_is_never_the_boundary():
    """Powell answers three times in the Q&A; none of those ends the remarks."""
    split = P.split_transcript(POWELL)
    assert split.remarks.count("CHAIR POWELL.") == 1
    assert split.qa.count("CHAIR POWELL.") == 2


def test_a_transcript_with_no_questions_is_all_remarks():
    only = "CHAIR POWELL.  Good afternoon.  That is all I have.\n"
    split = P.split_transcript(only)
    assert split.style == "none" and split.qa == "" and split.remarks.startswith("CHAIR")


def test_the_state_caps_both_halves():
    split = P.split_transcript(POWELL, date="20240918")
    presser = P.Presser(date="20240918", ts=2.0, statement_ts=1.0, url="u",
                        transcript=split, dots=["Median projection for end-2025 is 3.4%."])
    state = presser.state(remarks_cap=40, qa_cap=25)
    assert len(state["opening_remarks"]) == 40
    assert len(state["question_and_answer"]) == 25
    assert state["qa_marker_style"] == "moderator"
    assert state["projections_released_with_the_statement"] == [
        "Median projection for end-2025 is 3.4%."]
    assert state["minutes_after_the_statement"] == 0


# ------------------------------------------------------------------ the clock


def test_the_press_conference_starts_half_an_hour_after_a_2013_statement():
    for year, month, day in ((2013, 6, 19), (2019, 1, 30), (2024, 9, 18)):
        statement = eastern(year, month, day, 14, 0)
        start = P.presser_start(statement)
        assert start - statement == 30 * 60
        assert D.local_string(start, "fed").endswith("14:30")


def test_in_2011_and_2012_it_starts_at_a_quarter_past_two():
    """The statement went out at 12:30 and the Chair began at 2:15, so it is not an offset."""
    for year, month, day, hour, minute in (
        (2011, 4, 27, 12, 35), (2011, 11, 2, 12, 35), (2012, 9, 13, 12, 35),
        (2012, 12, 12, 12, 30),
    ):
        statement = eastern(year, month, day, hour, minute)
        assert D.local_string(P.presser_start(statement), "fed").endswith("14:15")
        assert P.presser_start(statement) - statement > 30 * 60


def test_the_rule_changes_at_the_end_of_2012():
    late_2012 = eastern(2012, 12, 12, 12, 30)
    early_2013 = eastern(2013, 3, 20, 14, 0)
    assert D.local_string(P.presser_start(late_2012), "fed").endswith("14:15")
    assert D.local_string(P.presser_start(early_2013), "fed").endswith("14:30")


# ------------------------------------------------------------------ the dates


CALENDAR_PAGE = """<html><body>
<a href="/monetarypolicy/fomcpresconf20240918.htm">Press Conference</a>
<a href="/monetarypolicy/fomcpressconf20260128.htm">Press Conference</a>
</body></html>"""
HISTORICAL_PAGE = """<html><body>
<a href="/monetarypolicy/fomcpresconf20110427.htm">Press Conference</a>
</body></html>"""


def test_both_spellings_of_the_link_are_found():
    assert P.presser_links(CALENDAR_PAGE) == {"20240918", "20260128"}


def test_presser_dates_read_the_calendar_and_the_historical_pages(tmp_path):
    def fetch(url, timeout=30.0):
        if url.endswith("fomccalendars.htm"):
            return CALENDAR_PAGE
        if url.endswith("fomchistorical2011.htm"):
            return HISTORICAL_PAGE
        raise OSError(f"404 {url}")

    store = Store(tmp_path)
    dates = P.presser_dates(
        store, since=datetime(2011, 1, 1, tzinfo=timezone.utc).timestamp(),
        until=datetime(2024, 12, 31, tzinfo=timezone.utc).timestamp(),
        fetcher=fetch, day="2026-09-20")
    assert dates == ["20110427", "20240918"]  # 2026 is outside the window


# ------------------------------------------------------------------ dark


def test_without_pypdf_the_arm_is_dark_and_nothing_raises(tmp_path, monkeypatch):
    """The whole point of the optional extra: a result, not a traceback."""
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(P.PdfUnavailable):
        P.pdf_text(b"%PDF-1.4 whatever")

    store = Store(tmp_path)
    pressers, coverage = P.collect(
        store, ["20240918"], {"20240918": eastern(2024, 9, 18, 14, 0)},
        fetcher=lambda url: b"%PDF-1.4 whatever", workers=1)
    assert pressers == {} and coverage.lit is False
    assert "pypdf" in coverage.dark and coverage.parsed == 0


def test_with_a_reader_the_arm_is_lit(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "pdf_text", lambda raw: POWELL)
    pressers, coverage = P.collect(
        tmp_path and Store(tmp_path), ["20240918"],
        {"20240918": eastern(2024, 9, 18, 14, 0)},
        fetcher=lambda url: b"%PDF-1.4 whatever", workers=1)
    assert coverage.lit and coverage.parsed == 1 and coverage.per_year == {"2024": 1}
    assert pressers["20240918"].transcript.style == "moderator"


def test_the_bytes_are_cached_and_fetched_once(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return b"%PDF-1.4 body"

    store = Store(tmp_path)
    assert P.fetch_transcript(store, "20240918", fetcher=fetch) == b"%PDF-1.4 body"
    assert P.fetch_transcript(store, "20240918", fetcher=fetch) == b"%PDF-1.4 body"
    assert len(calls) == 1


def test_a_missing_transcript_is_none_and_not_an_error(tmp_path):
    def fetch(url):
        raise OSError("404")

    assert P.fetch_transcript(Store(tmp_path), "19990101", fetcher=fetch) is None


# ------------------------------------------------------------------ the tree


class ScriptedClient:
    """A provider that answers whatever it is asked with values a test chose."""

    provider = "scripted"
    model = "scripted"

    def __init__(self, **answers):
        self.answers = answers
        self.questions = None
        self.seen: list[dict] = []
        self.states: list[dict] = []

    def evaluate(self, state):
        self.seen.append(dict(self.questions or {}))
        self.states.append(state)
        out = {}
        for key, question in (self.questions or {}).items():
            options = list(question["criteria"])
            want = self.answers.get(key)
            if question["type"] == "noul":
                out[key] = NoulAnswer(noul=float(want if want is not None else 0.2))
            elif question["type"] == "score":
                out[key] = ScoreAnswer(
                    score=float(want if want is not None else 0.0),
                    legend={str(i): t for i, t in enumerate(options)},
                    probabilities={str(i): 1 / len(options) for i in range(len(options))},
                    confidence=0.7)
            else:
                choice, p = (want if isinstance(want, tuple) else (want or options[0], 0.8))
                rest = (1.0 - p) / max(len(options) - 1, 1)
                out[key] = ChoiceAnswer(
                    choice=choice,
                    probabilities={o: (p if o == choice else rest) for o in options},
                    confidence=0.8)
        return JevResponse(model=self.model, answers=out, input_tokens=100,
                           output_tokens=0, latency_ms=1.0, provider=self.provider)


def statement_document(ts=None):
    return D.Document(
        id="fed:1:monetary20240918a.htm", issuer="fed", kind="monetary_policy",
        ts=ts if ts is not None else eastern(2024, 9, 18, 14, 0),
        title="Federal Reserve issues FOMC statement",
        body="The Committee decided to lower the target range by 1/2 percentage point. "
             "Inflation has eased and the labor market has cooled.",
        url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
        speaker="", currency="USD",
    )


PRESSER_BLOCK = {
    "starts_utc": "2024-09-18 18:30 UTC",
    "opening_remarks": "CHAIR POWELL.  We decided to lower the target range.",
    "question_and_answer": "STEVE LIESMAN.  Is that the new pace?  CHAIR POWELL.  No.",
}


def test_the_pc1_tree_asks_the_seven_and_keeps_the_statement_tree():
    questions = R.round_one_questions("USD", [], mode=R.PRESSER)
    assert set(questions) >= {
        R.REMARKS_VS_STATEMENT, R.QA_VS_REMARKS, R.PUSHBACK_ON_PRICING,
        R.PRESSER_STANCE, R.NEW_INFORMATION, R.SURPRISE_SIZE, R.DOMINANT_TOPIC}
    # ... and the absolute tree is still in there, because width is free.
    assert {R.KIND, R.STANCE, R.MAGNITUDE} <= set(questions)
    # ... and the context tree's questions are not: they are a different mode.
    assert R.RELATIVE_STANCE not in questions and R.EXPECTED_ACTION not in questions
    assert set(questions[R.DOMINANT_TOPIC]["criteria"]) == set(R.TOPICS)
    assert set(questions[R.PRESSER_STANCE]["criteria"]) == {R.HAWKISH, R.DOVISH, R.NEUTRAL}
    assert set(questions[R.QA_VS_REMARKS]["criteria"]) == set(R.VERSUS_OPTIONS)


def test_the_pc1_tree_has_its_own_cache_tag():
    assert R.tree_version(R.PRESSER) == "pc1"
    assert R.tree_version(R.PRESSER) not in (R.TREE_VERSION, R.CONTEXT_VERSION)


def test_round_two_reverses_the_framing_and_asks_the_holder():
    second = R.round_two_questions("USD", mode=R.PRESSER)
    assert set(second) == {R.HOLDER_UNAFFECTED, R.STANCE_REVERSED, R.HORIZON}
    assert "press conference" in second[R.STANCE_REVERSED]["instructions"]
    assert "dots" in second[R.STANCE_REVERSED]["instructions"]


def read_presser(**answers):
    client = ScriptedClient(**answers)
    reading = Reader(client, mode=R.PRESSER).read(statement_document(),
                                                  presser=PRESSER_BLOCK)
    return client, reading


def test_a_consistent_neutral_conference_never_reaches_round_two():
    client, reading = read_presser(
        presser_stance=R.NEUTRAL, remarks_vs_statement=R.VS_CONSISTENT,
        qa_vs_remarks=R.VS_CONSISTENT)
    assert reading.rounds == 1 and len(client.seen) == 1
    assert reading.verdict is None  # a neutral stance has no side

def test_a_disagreement_opens_round_two_even_when_the_stance_is_neutral():
    """The second door is the whole mode: 2022-11-02 is a day the halves disagree."""
    _client, reading = read_presser(
        presser_stance=R.NEUTRAL, remarks_vs_statement=R.VS_MORE_HAWKISH,
        qa_vs_remarks=R.VS_CONSISTENT)
    assert reading.rounds == 2
    assert reading.remarks_vs_statement == R.VS_MORE_HAWKISH


def test_the_qa_alone_opens_round_two():
    _client, reading = read_presser(
        presser_stance=R.NEUTRAL, remarks_vs_statement=R.VS_CONSISTENT,
        qa_vs_remarks=R.VS_MORE_DOVISH)
    assert reading.rounds == 2


def test_the_sign_is_the_conferences_stance_and_not_the_statements():
    _client, reading = read_presser(
        stance=(R.DOVISH, 0.9), presser_stance=(R.HAWKISH, 0.9),
        remarks_vs_statement=R.VS_MORE_HAWKISH, qa_vs_remarks=R.VS_MORE_HAWKISH,
        stance_reversed=(R.BUY, 0.9), surprise_size=3.0, new_information=0.9,
        holder_unaffected=0.1)
    assert reading.stance == R.DOVISH  # the statement read dovish ...
    assert reading.presser_stance == R.HAWKISH  # ... and the conference did not
    assert reading.verdict is not None
    assert reading.verdict.pair == "EURUSD=X" and reading.verdict.sign == -1
    assert reading.verdict.strength > 0.3


def test_the_transcript_reaches_the_state_in_both_rounds():
    client, reading = read_presser(
        presser_stance=(R.HAWKISH, 0.9), remarks_vs_statement=R.VS_MORE_HAWKISH,
        qa_vs_remarks=R.VS_CONSISTENT)
    assert reading.rounds == 2
    for state in client.states:
        assert state["press_conference"]["opening_remarks"].startswith("CHAIR POWELL.")
        assert "question_and_answer" in state["press_conference"]
    assert client.states[1]["first_round_verdicts"]["presser_stance"] == R.HAWKISH


def test_the_mock_answers_the_pc1_tree_offline():
    client = MockFxClient()
    reading = Reader(client, mode=R.PRESSER).read(
        statement_document(),
        presser={"opening_remarks": "Inflation remains elevated and we will tighten "
                                    "further; upside risks to price stability.",
                 "question_and_answer": "STEVE LIESMAN.  Is the market pricing it right? "
                                        "CHAIR POWELL.  Inflation is elevated."})
    assert reading.presser_stance in (R.HAWKISH, R.DOVISH, R.NEUTRAL)
    assert reading.remarks_vs_statement in R.VERSUS_OPTIONS
    assert reading.qa_vs_remarks in R.VERSUS_OPTIONS
    assert reading.dominant_topic in R.TOPICS
    assert 0.0 <= reading.pushback <= 1.0


def test_a_presser_reading_survives_the_cache_round_trip():
    from jevtrade.fx import study as S

    _client, reading = read_presser(
        presser_stance=(R.HAWKISH, 0.9), remarks_vs_statement=R.VS_MORE_HAWKISH,
        qa_vs_remarks=R.VS_MORE_DOVISH, dominant_topic=R.PATH_TOPIC,
        pushback_on_pricing=0.8)
    again = S.reading_from_dict(reading.document, S.reading_to_dict(reading))
    assert again.presser_stance == R.HAWKISH
    assert again.remarks_vs_statement == R.VS_MORE_HAWKISH
    assert again.qa_vs_remarks == R.VS_MORE_DOVISH
    assert again.dominant_topic == R.PATH_TOPIC
    assert again.pushback == pytest.approx(0.8)


def test_an_older_cached_reading_still_loads():
    """A ``v1`` reading written before this tree existed has none of these fields."""
    from jevtrade.fx import study as S

    data = S.reading_to_dict(read_presser()[1])
    for key in ("presser_stance", "remarks_vs_statement", "qa_vs_remarks",
                "pushback", "dominant_topic", "presser_chars", "p_presser"):
        data.pop(key)
    again = S.reading_from_dict(statement_document(), data)
    assert again.presser_stance == "" and again.pushback == 0.0


# ------------------------------------------------------------------ the arms


class FlatTape:
    """A tape that answers a fixed mid at every moment the test asks about."""

    def __init__(self, prices):
        self.prices = prices

    def mid_at(self, when, **_kwargs):
        return self.prices.get(round(when))


def test_the_reversal_table_shows_both_halves_of_the_day():
    statement = eastern(2022, 11, 2, 14, 0)
    start = P.presser_start(statement)
    tape = FlatTape({round(statement): 1.0000, round(start): 0.9950,
                     round(start + 3600): 0.9850})
    document = statement_document(ts=statement)
    _client, reading = read_presser(
        presser_stance=(R.HAWKISH, 0.9), remarks_vs_statement=R.VS_MORE_HAWKISH,
        qa_vs_remarks=R.VS_CONSISTENT)
    reading.document = document
    presser = P.Presser(date="20221102", ts=start, statement_ts=statement, url="u",
                        transcript=P.split_transcript(POWELL, date="20221102"))
    rows = P.reversals([reading], {document.id: presser}, tape)
    assert len(rows) == 1
    assert rows[0].date == "20221102"
    assert rows[0].remarks_vs_statement == R.VS_MORE_HAWKISH
    assert rows[0].statement_bps == pytest.approx(-50.1, abs=0.5)
    assert rows[0].presser_bps == pytest.approx(-101.0, abs=1.0)


def test_a_consistent_day_is_not_in_the_reversal_table():
    document = statement_document()
    _client, reading = read_presser(
        presser_stance=R.NEUTRAL, remarks_vs_statement=R.VS_CONSISTENT,
        qa_vs_remarks=R.VS_CONSISTENT)
    reading.document = document
    presser = P.Presser(date="20240918", ts=1.0, statement_ts=0.0, url="u",
                        transcript=P.split_transcript(POWELL))
    assert P.reversals([reading], {document.id: presser}, None) == []


def test_reader_signals_enter_when_the_chair_started():
    document = statement_document()
    _client, reading = read_presser(
        presser_stance=(R.HAWKISH, 0.9), remarks_vs_statement=R.VS_MORE_HAWKISH,
        qa_vs_remarks=R.VS_MORE_HAWKISH, stance_reversed=(R.BUY, 0.9),
        surprise_size=3.0, new_information=0.9, holder_unaffected=0.1)
    reading.document = document
    start = P.presser_start(document.ts)
    signals = P.reader_signals([reading], 0.05, {document.id: start})
    assert len(signals) == 1
    assert signals[0].ts == start and signals[0].pair == "EURUSD"
    assert signals[0].sign == -1  # hawkish dollar is short EURUSD


def test_the_keyword_arm_reads_the_transcript_and_not_the_statement():
    presser = P.Presser(
        date="20221102", ts=10.0, statement_ts=0.0, url="u",
        transcript=P.Transcript(date="20221102", remarks="Inflation remains elevated and "
                                                         "we will tighten further.",
                                qa="Upside risks and a restrictive stance.",
                                marker="", style="reporter", speakers=1))
    signals = P.keyword_signals([presser])
    assert len(signals) == 1
    assert signals[0].ts == 10.0 and signals[0].sign == -1 and signals[0].pair == "EURUSD"


# ------------------------------------------------------------------ end to end


def flat_hour(start, n=60, bid=1.17, spread=0.00002):
    import lzma
    import struct

    body = b"".join(
        struct.Struct(">IIIff").pack(int(i * 60_000), int(round((bid + spread) * 1e5)),
                                     int(round(bid * 1e5)), 1.0, 1.0)
        for i in range(n)
    )
    return lzma.compress(body, format=lzma.FORMAT_ALONE)


ARCHIVE = json.dumps([
    {"d": "9/18/2024 2:00:00 PM", "t": "Federal Reserve issues FOMC statement",
     "pt": "Monetary Policy", "l": "/newsevents/pressreleases/monetary20240918a.htm"},
    {"d": "11/7/2024 2:00:00 PM", "t": "Federal Reserve issues FOMC statement",
     "pt": "Monetary Policy", "l": "/newsevents/pressreleases/monetary20241107a.htm"},
    {"d": "12/18/2024 2:00:00 PM", "t": "Federal Reserve issues FOMC statement",
     "pt": "Monetary Policy", "l": "/newsevents/pressreleases/monetary20241218a.htm"},
])
CALENDAR_2024 = """<html><body>
<a href="/monetarypolicy/fomcpresconf20240918.htm">Press Conference</a>
<a href="/monetarypolicy/fomcpresconf20241107.htm">Press Conference</a>
<a href="/monetarypolicy/fomcpresconf20241218.htm">Press Conference</a>
</body></html>"""
STATEMENT_PAGE = (
    '<html><body><div class="col-xs-12 col-sm-8 col-md-8"><p>'
    "The Committee decided to lower the target range for the federal funds rate to "
    "4-1/2 to 4-3/4 percent. Inflation has made further progress toward the "
    "Committee's 2 percent objective but remains somewhat elevated. The Committee "
    "judges that the risks to achieving its employment and inflation goals are "
    "roughly in balance and will continue to monitor incoming data."
    "</p></div></body></html>"
)


def test_cli_presser_runs_end_to_end_offline(tmp_path, monkeypatch, capsys):
    """``fx --presser --provider mock --tape dukascopy`` with every fetch stubbed."""
    from jevtrade import cli
    from jevtrade.fx import context as context_module
    from jevtrade.fx import documents as docs_module
    from jevtrade.fx import dots as dots_module
    from jevtrade.fx import presser as presser_module
    from jevtrade.fx import ticks as ticks_module

    def fed_page(url, timeout=30.0):
        if url.endswith("ne-press.json"):
            return ARCHIVE
        if url.endswith(("ne-speeches.json", "ne-testimony.json")):
            return "[]"
        return STATEMENT_PAGE

    def calendar_page(url, timeout=30.0):
        if url.endswith("fomccalendars.htm"):
            return CALENDAR_2024
        raise OSError(f"404 {url}")

    def feed(url):
        stamp = url.rsplit("/datafeed/", 1)[-1]
        _symbol, year, month, day, hour = stamp.split("/")
        when = datetime(int(year), int(month) + 1, int(day), int(hour[:2]),
                        tzinfo=timezone.utc)
        return flat_hour(when.timestamp(), bid=1.17 + (when.timestamp() % 997) / 1e6)

    # One transcript per day, in all three marker styles, carried as bytes so the
    # split is exercised end to end without a PDF anywhere in the repository.
    transcripts = {"20240918": POWELL, "20241107": YELLEN, "20241218": BERNANKE}

    def transcript_bytes(url, timeout=60.0):
        date = url.rsplit("FOMCpresconf", 1)[-1][:8]
        return b"%PDF-1.4\n" + transcripts[date].encode("utf-8")

    monkeypatch.setattr(docs_module, "get_text", fed_page)
    monkeypatch.setattr(docs_module, "fetch_body",
                        lambda url: docs_module.body_text(url, fed_page(url)))
    monkeypatch.setattr(presser_module, "get_text", calendar_page)
    monkeypatch.setattr(dots_module, "get_text", calendar_page)
    monkeypatch.setattr(context_module, "get_text", calendar_page)
    monkeypatch.setattr(presser_module, "get_bytes", transcript_bytes)
    monkeypatch.setattr(presser_module, "pdf_text",
                        lambda raw: raw.decode("utf-8").split("\n", 1)[1])
    monkeypatch.setattr(ticks_module, "fetch_bi5", feed)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    out_file = tmp_path / "presser.json"
    code = cli.main([
        "fx", "--presser", "--provider", "mock", "--tape", "dukascopy",
        "--since", "2024-09-01", "--until", "2024-12-31", "--horizons", "1,15",
        "--cache", str(tmp_path / "cache"), "--workers", "2", "--workers-io", "2",
        "--null-per", "1", "--threshold", "0.01", "--latency-sweep",
        "--out", str(out_file),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "press conferences on the calendar: 3" in printed
    assert "transcripts read: 3/3" in printed
    assert "reader-presser" in printed and "statement-reader" in printed
    assert "dots-rule" in printed and "all pressers" in printed
    assert "where the press conference did not say what the statement said" in printed
    assert "the statement plus thirty" in printed

    record = json.loads(out_file.read_text())
    assert record["mode"] == "presser" and record["tree"] == "pc1"
    assert record["dark"] == "" and record["transcripts"] == 3
    assert {a["name"] for a in record["arms"]} == {
        "presser-reader", "statement-reader", "dots-rule", "all pressers"}
    assert len(record["days"]) == 3
    day = record["days"][0]
    assert day["presser_ts"] - day["statement_ts"] == 30 * 60
    assert day["qa_marker_style"] == "moderator" and day["remarks_chars"] > 0
    styles = {d["date"]: d["qa_marker_style"] for d in record["days"]}
    assert styles == {"20240918": "moderator", "20241107": "reporter",
                      "20241218": "question"}
    assert day["presser_reading"]["mode"] == "presser"
    assert day["presser_reading"]["presser_stance"] in (R.HAWKISH, R.DOVISH, R.NEUTRAL)
    assert isinstance(record["reversals"], list)


def test_cli_presser_reports_the_arm_dark_without_pypdf(tmp_path, monkeypatch, capsys):
    """No ``pypdf``, no transcript arms -- and still an exit code of zero."""
    from jevtrade import cli
    from jevtrade.fx import context as context_module
    from jevtrade.fx import documents as docs_module
    from jevtrade.fx import dots as dots_module
    from jevtrade.fx import presser as presser_module

    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setattr(docs_module, "get_text",
                        lambda url, timeout=30.0: ARCHIVE if url.endswith("ne-press.json")
                        else ("[]" if url.endswith(("ne-speeches.json", "ne-testimony.json"))
                              else STATEMENT_PAGE))
    monkeypatch.setattr(docs_module, "fetch_body",
                        lambda url: docs_module.body_text(url, STATEMENT_PAGE))
    monkeypatch.setattr(presser_module, "get_text",
                        lambda url, timeout=30.0: CALENDAR_2024
                        if url.endswith("fomccalendars.htm") else "")
    monkeypatch.setattr(dots_module, "get_text", lambda url, timeout=30.0: "")
    monkeypatch.setattr(context_module, "get_text", lambda url, timeout=30.0: "")
    monkeypatch.setattr(presser_module, "get_bytes", lambda url, timeout=60.0: b"%PDF-1.4")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    code = cli.main([
        "fx", "--presser", "--provider", "mock", "--tape", "yahoo",
        "--since", "2024-09-01", "--until", "2024-12-31", "--horizons", "15",
        "--cache", str(tmp_path / "cache"), "--workers", "1", "--workers-io", "1",
        "--null-per", "1",
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "press-conference reader: DARK" in printed
    assert "pypdf" in printed
    assert "statement-reader" in printed  # the arms that need no transcript still run
