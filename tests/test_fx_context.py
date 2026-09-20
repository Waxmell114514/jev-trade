"""Reading a statement against what the market already had.

The 2009-2026 run said the reader is a coin flip on FOMC statements, and the
diagnosis was that the tree asks "is this a surprise?" with nothing to be
surprised against. These tests are about the thing that fixes that, and the one
rule it has to keep: **nothing in the context may post-date the release**. That
is asserted directly (a minutes excerpt dated after the statement raises), and
indirectly everywhere else -- the H.15 row is the last one *strictly before* the
release's date, the intermeeting window is open at both ends, and the only item
allowed to share the release's minute is the SEP, which really is published then.

Everything runs offline, from fixtures under ``tests/fixtures/fx`` and from the
trimmed sources below.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jevtrade.fx import context as C
from jevtrade.fx import documents as D
from jevtrade.fx import reader as R
from jevtrade.fx import study as S
from jevtrade.fx.mock import MockFxClient
from jevtrade.listing.store import Store
from jevtrade.types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer

FIXTURES = Path(__file__).parent / "fixtures" / "fx"
FOMC_SEP_TS = 1789581600.0  # 2026-09-16 14:00 EDT = 18:00 UTC
FOMC_JUL_TS = 1785348000.0  # 2026-07-29 14:00 EDT = 18:00 UTC
DAY = 86400.0

# A trimmed H.15 package: the five header rows the Data Download Program emits,
# then four daily rows, one of them all ``ND`` (a holiday) and one with a gap.
_CMT = ("Market yield on U.S. Treasury securities at {} constant maturity, "
        "quoted on investment basis")
_HEAD = ['"Series Description"']
_HEAD += [f'"{_CMT.format(t)}"' for t in ("6-month  ", "2-year  ", "10-year  ")]
_HEAD += ['"Federal funds effective rate"']
H15_CSV = "\n".join([
    ",".join(_HEAD),
    '"Unit:","Percent:_Per_Year","Percent:_Per_Year","Percent:_Per_Year","Percent:_Per_Year"',
    '"Multiplier:","1","1","1","1"',
    '"Currency:","NA","NA","NA","NA"',
    '"Unique Identifier: ","H15/H15/RIFLGFCM06_N.B","H15/H15/RIFLGFCY02_N.B",'
    '"H15/H15/RIFLGFCY10_N.B","H15/H15/RIFSPFF_N.B"',
    '"Time Period","RIFLGFCM06_N.B","RIFLGFCY02_N.B","RIFLGFCY10_N.B","RIFSPFF_N.B"',
    "2026-07-28,4.05,3.80,4.20,4.33",
    "2026-07-29,4.06,3.82,4.21,4.33",
    "2026-09-07,ND,ND,ND,ND",
    "2026-09-15,4.17,4.67,5.00,4.33",
    "",
])

# The funds-rate block of the December 2024 projection table, as it reads once
# the tags are stripped: the header's year columns, the median row, the central
# tendency and range behind it, and September's medians underneath.
SEP_2024 = (
    "Percent Variable Median 1 Central Tendency 2 Range 3 "
    "2024 2025 2026 2027 Longer run 2024 2025 2026 2027 Longer run "
    "2024 2025 2026 2027 Longer run "
    "Change in real GDP 2.5 2.1 2.0 1.9 1.8 2.4–2.5 1.8–2.2 1.9–2.1 "
    "1.8–2.0 1.7–2.0 2.3–2.7 1.6–2.5 1.4–2.5 1.5–2.5 1.7–2.5 "
    "Memo: Projected appropriate policy path "
    "Federal funds rate 4.4 3.9 3.4 3.1 3.0 "
    "4.4–4.6 3.6–4.1 3.1–3.6 2.9–3.6 2.8–3.6 "
    "4.4–4.6 3.1–4.4 2.4–3.9 2.4–3.9 2.4–3.9 "
    "September projection 4.4 3.4 2.9 2.9 2.9 "
    "4.4–4.6 3.1–3.6 2.6–3.6 2.6–3.6 2.5–3.5 "
    "Note: Projections of change in real gross domestic product"
)
# The January 2012 layout: ranges and a histogram, no median row at all.
SEP_2012 = (
    "Percent Variable Central tendency 1 Range 2 2012 2013 2014 Longer run "
    "Change in real GDP 2.2 to 2.7 2.8 to 3.2 3.3 to 4.0 2.3 to 2.6 "
    "Appropriate Pace of Policy Firming Target Federal Funds Rate at Year-End "
    "0.25 14 11 6 0.50 1 1 2"
)

MINUTES = (
    "A joint meeting of the Federal Open Market Committee began at 10:00 a.m. "
    "Staff Review of the Economic Situation The information reviewed at the meeting "
    "indicated that activity expanded at a solid pace. " + "Data data data. " * 80
    + "Participants' Views on Current Conditions and the Economic Outlook "
    + "Participants observed that inflation had eased. " * 40
    + "A couple of participants said the last mile would be the hardest. "
    "Committee Policy Action In their discussion of monetary policy for this "
    "meeting, members agreed that the target range should be maintained. "
    + "Members concurred that the risks were balanced. " * 20
    + "Voting for this action: everyone. Return to top Last Update: January 08, 2025"
)


def doc(title="Federal Reserve issues FOMC statement", body="", ts=FOMC_SEP_TS,
        kind=D.MONETARY_POLICY, speaker="", issuer="fed"):
    return D.Document(id=f"{issuer}:{int(ts)}:{abs(hash(title + str(ts))) % 99999}",
                      issuer=issuer, kind=kind, ts=ts, title=title, body=body,
                      url=f"https://x/{int(ts)}.htm", speaker=speaker,
                      currency=D.ISSUER_CURRENCY.get(issuer, ""))


def rate_table():
    return C.parse_h15(H15_CSV)


# ------------------------------------------------------------------- the rates

def test_h15_columns_are_mapped_by_their_series_description_and_nd_is_absent():
    table = rate_table()
    assert table.dates == ("2026-07-28", "2026-07-29", "2026-09-07", "2026-09-15")
    assert table.rows["2026-09-15"] == {"6m": 4.17, "2y": 4.67, "10y": 5.00, "ff": 4.33}
    # The all-ND holiday row exists as a date and carries no values at all; a
    # missing yield must never arrive as a zero.
    assert table.rows["2026-09-07"] == {}
    assert table.on("2026-09-07") is None
    assert set(table.series) == {"6m", "2y", "10y", "ff"}
    assert "6-month" in table.series["6m"]


def test_a_byte_order_mark_does_not_empty_the_table():
    table = C.parse_h15("\ufeff" + H15_CSV)
    assert set(table.series) == {"6m", "2y", "10y", "ff"}
    assert table.rows["2026-09-15"]["ff"] == 4.33


def test_two_packages_merge_and_the_first_one_wins():
    other = C.parse_h15(H15_CSV.replace("4.17", "9.99"))
    merged = C.merge_tables(rate_table(), other)
    assert merged.rows["2026-09-15"]["6m"] == 4.17
    assert len(merged) == 4


def test_the_treasury_csv_is_read_by_column_name_as_the_fallback():
    csv = ('Date,"1 Mo","6 Mo","2 Yr","10 Yr"\n'
           '09/15/2026,4.00,4.17,4.67,5.00\n'
           '09/14/2026,4.01,4.18,4.68,5.01\n')
    table = C.parse_treasury(csv)
    assert table.dates == ("2026-09-14", "2026-09-15")
    assert table.rows["2026-09-15"] == {"1m": 4.00, "6m": 4.17, "2y": 4.67, "10y": 5.00}


def test_rates_before_takes_the_last_row_strictly_before_the_release_date():
    """The H.15 row named after the meeting day prints at 4:15 p.m., after the 2 p.m.
    statement, so sharing the release's date is already a lookahead."""
    table = rate_table()
    rates = C.rates_before(table, FOMC_SEP_TS, previous_ts=FOMC_JUL_TS)
    assert rates is not None
    assert rates.as_of == "2026-09-15"     # not 2026-09-16
    assert rates.prior_as_of == "2026-07-29"
    assert rates.ts < FOMC_SEP_TS
    assert round(rates.change_bp("2y"), 6) == 85.0
    assert round(rates.change_bp("10y"), 6) == 79.0


def test_a_release_before_every_row_has_no_rates_rather_than_the_nearest_ones():
    before_everything = datetime(2026, 7, 27, tzinfo=timezone.utc).timestamp()
    assert C.rates_before(rate_table(), before_everything) is None


def test_the_all_nd_row_is_skipped_when_it_is_the_last_one_before_the_release():
    table = rate_table()
    rates = C.rates_before(table, datetime(2026, 9, 8, tzinfo=timezone.utc).timestamp())
    assert rates is not None and rates.as_of == "2026-07-29"


# --------------------------------------------------------------- the sentences

def test_pricing_sentences_read_the_bill_as_quarter_point_steps_in_a_hiking_cycle():
    rates = C.Rates(as_of="2026-09-15", ts=FOMC_SEP_TS - DAY,
                    values={"ff": 3.33, "6m": 4.10, "2y": 4.67, "10y": 5.00, "1y": 4.40},
                    prior_as_of="2026-07-29",
                    prior_values={"ff": 3.33, "6m": 4.05, "2y": 3.82, "10y": 4.20, "1y": 4.00})
    previous = doc(body="the Committee decided to maintain the target range for the "
                        "federal funds rate at 3-1/4 to 3-1/2 percent.", ts=FOMC_JUL_TS)
    lines = C.pricing_sentences(rates, previous_statement=previous)
    assert 4 <= len(lines) <= 6
    assert "3.50 percent" in lines[0]
    assert "3 quarter-point hikes within six months" in lines[1]
    assert "85 bp higher" in lines[2] and "moved hawkish" in lines[2]
    assert "80 bp higher" in lines[3]


def test_pricing_sentences_say_so_at_the_zero_bound():
    rates = C.Rates(as_of="2009-03-17", ts=0.0,
                    values={"ff": 0.20, "6m": 0.45, "2y": 1.05, "10y": 3.02},
                    prior_as_of="2009-01-28", prior_values={"2y": 0.89, "10y": 2.71})
    lines = C.pricing_sentences(rates)
    assert "zero bound" in lines[1]
    assert "no room to price cuts" in lines[1]
    assert "1 quarter-point hike within six months" in lines[1]
    assert "16 bp higher" in lines[2]


def test_a_parsed_target_the_effective_rate_contradicts_is_refused():
    """``announced_rate`` reads "at 0 to 1/4 percent" as 1.00; the funds rate says no."""
    zirp = doc(body="the Committee decided to keep its target range for the federal "
                    "funds rate at 0 to 1/4 percent.", ts=FOMC_JUL_TS)
    from jevtrade.fx.baseline import announced_rate

    assert announced_rate(f"{zirp.title}\n{zirp.body}") == 1.0
    rates = C.Rates(as_of="2009-03-17", ts=0.0, values={"ff": 0.20, "6m": 0.45})
    lines = C.pricing_sentences(rates, previous_statement=zirp)
    assert "could not be parsed" in lines[0] and "0.20 percent" in lines[0]


@pytest.mark.parametrize("spread,steps", [(0.51, 2), (-0.26, -1), (0.05, 0), (-0.74, -3)])
def test_quarter_steps_rounds_to_whole_moves(spread, steps):
    assert C.quarter_steps(spread) == steps


# -------------------------------------------------------------------- the dots

def test_sep_medians_parse_out_of_the_december_2024_table():
    dots = C.parse_sep(SEP_2024)
    assert dots is not None
    assert dots.years == ("2024", "2025", "2026", "2027", "longer run")
    assert dots.medians == (4.4, 3.9, 3.4, 3.1, 3.0)
    assert dots.previous_label == "September"
    assert dots.previous == (4.4, 3.4, 2.9, 2.9, 2.9)
    lines = C.dots_sentences(dots)
    assert lines[0] == "Median projection for end-2024 is unchanged at 4.4%."
    assert "moved to 3.9% from 3.4% in September: fewer cuts" in lines[1]


def test_a_projection_page_with_no_median_row_is_no_dots_and_not_a_failure():
    assert C.parse_sep(SEP_2012) is None
    assert C.parse_sep("") is None


def test_a_previous_projection_with_one_fewer_year_keeps_the_longer_run_aligned():
    """September adds a year the June table had no column for; the last value is
    always the longer run, so the new year's cell is blank rather than shifted."""
    text = SEP_2024.replace("September projection 4.4 3.4 2.9 2.9 2.9",
                            "September projection 4.4 3.4 2.9 2.9")
    dots = C.parse_sep(text)
    assert dots is not None and dots.previous == (4.4, 3.4, 2.9, None, 2.9)
    assert C.dots_sentences(dots)[3] == "Median projection for end-2027 is 3.1%."


def test_a_central_tendency_printed_as_a_bare_number_does_not_shift_the_previous_row():
    """2020's table writes ".1" and its central tendency as "0.1", which reads as one
    more median; the longer-run dot is what catches the over-read."""
    text = ("Percent Variable Median 1 Central Tendency 2 Range 3 "
            "2020 2021 2022 2023 Longer run 2020 2021 2022 2023 Longer run "
            "Memo: Projected appropriate policy path "
            "Federal funds rate .1 .1 .1 .1 2.5 0.1 0.1 0.1 0.1–0.4 2.3–2.5 "
            "June projection .1 .1 .1 2.5 0.1 0.1 0.1 2.3–2.5 Note: Projections")
    dots = C.parse_sep(text)
    assert dots is not None
    assert dots.medians == (0.1, 0.1, 0.1, 0.1, 2.5)
    assert dots.previous == (0.1, 0.1, 0.1, None, 2.5)


def test_dots_are_fetched_per_meeting_and_stamped_as_concurrent(tmp_path):
    store = Store(tmp_path)
    served: list[str] = []

    def fetch(url):
        served.append(url)
        if url.endswith("fomcprojtabl20241218.htm"):
            raise OSError("404")  # the Fed has used both spellings
        return "<html><body><p>" + SEP_2024 + "</p></body></html>"

    dots = C.dots(store, FOMC_SEP_TS, fetcher=fetch, local_date="20241218")
    assert dots is not None and dots.ts == FOMC_SEP_TS
    assert served[-1].endswith("fomcprojtable20241218.htm")
    assert dots.medians[1] == 3.9


# ----------------------------------------------------------------- the minutes

def test_minutes_are_sliced_to_the_policy_action_and_the_tail_of_the_views():
    text = C.slice_minutes(MINUTES, cap=4000)
    assert "Committee policy action" in text
    assert "Participants' views, end of the section" in text
    assert "the last mile would be the hardest" in text   # the tail is kept
    assert "Staff Review of the Economic Situation" not in text
    assert "Data data data" not in text
    assert "Last Update" not in text                      # the page chrome is gone
    assert len(text) <= 4000


def test_a_lowercase_mention_of_participants_views_is_not_a_heading():
    body = ("The Committee discussed participants' views of longer-run sustainable "
            "rates. " * 3) + "Committee Policy Action The members voted to hold."
    text = C.slice_minutes(body)
    assert text.startswith("Committee policy action")


def test_minutes_with_no_headings_fall_back_to_their_tail_rather_than_to_nothing():
    text = C.slice_minutes("word " * 2000, cap=300)
    assert len(text) == 300 and text.strip().endswith("word")


def test_minutes_excerpt_follows_the_link_and_never_takes_one_released_after(tmp_path):
    store = Store(tmp_path)
    rows = [
        {"title": "Minutes of Federal Open Market Committee, June 16-17, 2026",
         "ts": FOMC_SEP_TS - 60 * DAY, "url": "https://fed/press/june.htm"},
        {"title": "Minutes of the Federal Open Market Committee, July 28-29, 2026",
         "ts": FOMC_SEP_TS - 28 * DAY, "url": "https://fed/press/july.htm"},
        {"title": "Minutes of the Federal Open Market Committee, September 15-16, 2026",
         "ts": FOMC_SEP_TS + 21 * DAY, "url": "https://fed/press/september.htm"},
        {"title": "Minutes of Board discount rate meetings, July 2026",
         "ts": FOMC_SEP_TS - 2 * DAY, "url": "https://fed/press/discount.htm"},
    ]

    def fetch(url):
        if url.endswith("july.htm"):
            return '<html><a href="/monetarypolicy/fomcminutes20260729.htm">HTML</a></html>'
        if "fomcminutes20260729" in url:
            return "<html><body><p>" + MINUTES + "</p></body></html>"
        raise AssertionError(f"asked for {url}")

    excerpt = C.minutes_excerpt(store, rows, FOMC_SEP_TS, fetcher=fetch)
    assert excerpt is not None
    assert excerpt.ts == FOMC_SEP_TS - 28 * DAY
    assert excerpt.url.endswith("/monetarypolicy/fomcminutes20260729.htm")
    assert "Committee policy action" in excerpt.text


def test_no_minutes_before_the_release_is_none(tmp_path):
    rows = [{"title": "Minutes of the Federal Open Market Committee, September 2026",
             "ts": FOMC_SEP_TS + DAY, "url": "https://fed/press/september.htm"}]
    assert C.minutes_excerpt(Store(tmp_path), rows, FOMC_SEP_TS,
                             fetcher=lambda url: "") is None


# --------------------------------------------------------- what officials said

@pytest.mark.parametrize("speaker,chair", [
    ("Chair Jerome H. Powell", True),
    ("Chairman Ben S. Bernanke", True),
    ("Vice Chair for Supervision Michelle W. Bowman", False),
    ("Vice Chair Philip N. Jefferson", False),
    ("Governor Christopher J. Waller", False),
])
def test_the_chair_is_the_chair_and_not_the_vice_chair(speaker, chair):
    assert C.is_chair(speaker) is chair


def test_intermeeting_takes_the_chair_first_bounded_and_nothing_outside_the_window():
    documents = [
        doc("Before the last meeting", ts=FOMC_JUL_TS - DAY, kind=D.SPEECH,
            speaker="Chair Jerome H. Powell", body="x" * 900),
        doc("At the last meeting's minute", ts=FOMC_JUL_TS, kind=D.SPEECH,
            speaker="Chair Jerome H. Powell", body="x" * 900),
        *[doc(f"Chair remark {i}", ts=FOMC_JUL_TS + (i + 1) * DAY, kind=D.SPEECH,
              speaker="Chair Jerome H. Powell", body=f"body {i} " * 200) for i in range(4)],
        *[doc(f"Governor remark {i}", ts=FOMC_JUL_TS + (i + 10) * DAY, kind=D.SPEECH,
              speaker="Governor Waller", body="y" * 900) for i in range(12)],
        doc("Testimony", ts=FOMC_JUL_TS + 5 * DAY, kind=D.TESTIMONY,
            speaker="Vice Chair Jefferson", body="z" * 900),
        doc("A statement, not a speech", ts=FOMC_JUL_TS + 6 * DAY, kind=D.MONETARY_POLICY),
        doc("After this release", ts=FOMC_SEP_TS + DAY, kind=D.SPEECH,
            speaker="Chair Jerome H. Powell", body="x" * 900),
    ]
    remarks = C.intermeeting_communication(documents, FOMC_JUL_TS, FOMC_SEP_TS)
    titles = [r.title for r in remarks]
    assert "Before the last meeting" not in titles
    assert "At the last meeting's minute" not in titles
    assert "After this release" not in titles
    assert "A statement, not a speech" not in titles
    chairs = [r for r in remarks if r.chair]
    assert [r.title for r in chairs] == ["Chair remark 1", "Chair remark 2", "Chair remark 3"]
    assert all(len(r.body) <= C.REMARK_CHARS and r.body for r in chairs)
    assert all(r.body == "" for r in remarks if not r.chair)
    assert len(remarks) - len(chairs) == C.MAX_OTHERS
    assert all(FOMC_JUL_TS < r.ts < FOMC_SEP_TS for r in remarks)


# ----------------------------------------------------------- the no-lookahead rule

def base_context(**over):
    kwargs = dict(
        released_at=FOMC_SEP_TS,
        pricing=["The target range tops out at 4.00 percent."],
        rates=C.Rates(as_of="2026-09-15", ts=FOMC_SEP_TS - DAY, values={"ff": 4.33}),
        previous=doc("Federal Reserve issues FOMC statement", body="Previous. " * 40,
                     ts=FOMC_JUL_TS),
        minutes=C.Excerpt("Minutes", "https://fed/m.htm", FOMC_SEP_TS - 28 * DAY,
                          "Committee policy action: " + "held. " * 200),
        communication=[C.Remark(FOMC_SEP_TS - 10 * DAY, "Chair Jerome H. Powell",
                                "A speech", "words " * 50, True)],
        drift=C.Drift(FOMC_SEP_TS - 1.0, 1.17, -32.0, 4.0,
                      ["EURUSD is at 1.17000, 32 bp lower than 24 hours before the release."]),
    )
    kwargs.update(over)
    return C.Context(**kwargs)


def test_every_context_item_carries_its_own_timestamp():
    context = base_context()
    kinds = {s.kind for s in context.sources}
    assert kinds == {"rates", "previous_statement", "minutes", "chair_remark", "drift"}
    assert all(s.ts < FOMC_SEP_TS for s in context.sources)
    assert context.has == {"rates": True, "dots": False, "previous": True, "minutes": True,
                           "chair": True, "communication": True, "drift": True}


def test_a_context_item_dated_after_the_release_refuses_to_become_a_context():
    late = C.Excerpt("Minutes", "https://fed/m.htm", FOMC_SEP_TS + 1.0, "words")
    with pytest.raises(C.LookaheadError) as raised:
        base_context(minutes=late)
    assert "minutes" in str(raised.value)

    with pytest.raises(C.LookaheadError):
        base_context(communication=[C.Remark(FOMC_SEP_TS, "Chair Powell", "Now", "", True)])
    with pytest.raises(C.LookaheadError):
        base_context(previous=doc(body="x", ts=FOMC_SEP_TS + 60))


def test_the_concurrent_sep_may_share_the_releases_minute_and_nothing_may_pass_it():
    dots = C.Dots("https://fed/sep.htm", FOMC_SEP_TS, ("2026", "longer run"),
                  (4.1, 3.2), "June", (3.8, 3.1), ["Median projection for end-2026 is 4.1%."])
    context = base_context(projections=dots)
    concurrent = [s for s in context.sources if s.concurrent]
    assert len(concurrent) == 1 and concurrent[0].ts == FOMC_SEP_TS
    with pytest.raises(C.LookaheadError):
        base_context(projections=C.Dots("u", FOMC_SEP_TS + 1.0, ("2026",), (4.1,), "", ()))


def test_as_text_lays_the_blocks_out_in_a_fixed_order_and_respects_the_budget():
    dots = C.Dots("u", FOMC_SEP_TS, ("2026", "longer run"), (4.1, 3.2), "June", (3.8, 3.1),
                  ["Median projection for end-2026 moved to 4.1% from 3.8% in June: fewer cuts."])
    context = base_context(projections=dots)
    text = context.as_text(12000)
    order = [text.index(h) for h in (
        "RATES AND WHAT THEY PRICE",
        "PROJECTIONS RELEASED WITH THIS STATEMENT",
        "THE PREVIOUS STATEMENT",
        "THE LAST MINUTES BEFORE THIS MEETING",
        "WHAT OFFICIALS SAID BETWEEN THE MEETINGS",
        "THE PAIR INTO THE RELEASE",
    )]
    assert order == sorted(order)
    assert "concurrent, not prior" in text
    for budget in (400, 1200, 3000, 12000):
        assert len(context.as_text(budget)) <= budget
    # The short blocks survive a tight budget; the long ones are what gives way.
    assert "RATES AND WHAT THEY PRICE" in context.as_text(1200)


def test_market_drift_reads_the_last_quote_before_the_release_and_never_at_it():
    class Tape:
        def __init__(self):
            self.asked = []

        def quote_at(self, when, max_age_s=None):
            self.asked.append(when)
            from jevtrade.fx.ticks import Quote

            return Quote(1.16990, 1.17010, when)

        def mid_at(self, when, max_age_s=None):
            return 1.17000 if when < FOMC_SEP_TS - 3600 else 1.17050

    tape = Tape()
    drift = C.market_drift(tape, FOMC_SEP_TS)
    assert drift is not None and drift.ts < FOMC_SEP_TS
    assert tape.asked[0] == FOMC_SEP_TS - 0.001
    assert "24 hours before the release" in drift.sentences[0]
    assert "fifteen minutes before the release" in drift.sentences[1]
    assert C.market_drift(None, FOMC_SEP_TS) is None


def test_the_previous_statement_falls_back_when_the_title_family_changed():
    old = doc("FOMC statement", body="Old wording. " * 20, ts=FOMC_JUL_TS - 45 * DAY)
    renamed = doc("Federal Reserve issues FOMC statement", body="New wording. " * 20,
                  ts=FOMC_JUL_TS)
    later = doc("Federal Reserve issues FOMC statement", body="Later wording. " * 20,
                ts=FOMC_SEP_TS)
    corpus = [old, renamed, later]
    assert C.previous_statement(later, corpus) is renamed      # same family
    assert C.previous_statement(renamed, corpus) is old        # the family changed
    assert C.previous_statement(old, corpus) is None


# --------------------------------------------------------------- the questions

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


def test_context_mode_carries_the_absolute_questions_and_adds_the_relative_ones():
    absolute = R.round_one_questions("USD", [])
    context = R.round_one_questions("USD", [], mode=R.CONTEXT)
    assert set(absolute) < set(context)
    added = set(context) - set(absolute)
    assert added == {R.EXPECTED_ACTION, R.ACTUAL_ACTION, R.RELATIVE_STANCE,
                     R.SURPRISE_CHANNEL, R.SURPRISE_SIZE, R.VERSUS_MINUTES}
    assert set(context[R.RELATIVE_STANCE]["criteria"]) == set(R.RELATIVE_OPTIONS)
    assert set(context[R.SURPRISE_CHANNEL]["criteria"]) == set(R.CHANNELS)
    assert set(context[R.EXPECTED_ACTION]["criteria"]) == set(R.ACTIONS)
    assert context[R.SURPRISE_SIZE]["criteria"] == list(R.SURPRISE_LEVELS)
    assert len(R.SURPRISE_LEVELS) == 4


def test_the_context_tree_has_its_own_cache_tag():
    assert R.tree_version(R.ABSOLUTE) == "v1"
    assert R.tree_version(R.CONTEXT) == "ctx1"
    assert R.TREE_VERSION == "v1" and R.CONTEXT_VERSION == "ctx1"


def read_in_context(client, *, context="WHAT THE MARKET ALREADY HAD: everything.", **kw):
    reader = R.Reader(client, mode=R.CONTEXT, **kw)
    return reader.read(doc(body="The Committee decided to maintain the target range."),
                       context=context)


def test_the_relative_call_opens_round_two_and_an_in_line_reading_does_not():
    decisive = ScriptedClient(**{R.RELATIVE_STANCE: (R.MORE_HAWKISH, 0.8),
                                 R.SURPRISE_SIZE: 2.0, R.STANCE: (R.NEUTRAL, 0.9)})
    assert read_in_context(decisive).rounds == 2

    quiet = ScriptedClient(**{R.RELATIVE_STANCE: (R.IN_LINE, 0.9), R.SURPRISE_SIZE: 0.0,
                              R.STANCE: (R.HAWKISH, 0.9)})
    reading = read_in_context(quiet)
    assert reading.rounds == 1 and reading.verdict is None

    # A statement the model calls in-line but enormous still gets the second look.
    loud = ScriptedClient(**{R.RELATIVE_STANCE: (R.IN_LINE, 0.9), R.SURPRISE_SIZE: 3.0})
    assert read_in_context(loud).rounds == 2


def test_the_state_carries_the_context_block_in_both_rounds():
    client = ScriptedClient(**{R.RELATIVE_STANCE: (R.MORE_DOVISH, 0.8), R.SURPRISE_SIZE: 2.0})
    reading = read_in_context(client, context="PRICED: two cuts.")
    assert reading.rounds == 2 and reading.context_chars == len("PRICED: two cuts.")
    assert all(s["context_before_the_release"] == "PRICED: two cuts." for s in client.states)
    assert client.states[1]["first_round_verdicts"]["relative_stance"] == R.MORE_DOVISH


@pytest.mark.parametrize("relative,sign", [
    (R.MORE_HAWKISH, -1),   # hawkish for USD -> EURUSD down
    (R.MORE_DOVISH, +1),
])
def test_the_signal_is_signed_by_the_relative_call_and_not_by_the_tone(relative, sign):
    client = ScriptedClient(**{R.RELATIVE_STANCE: (relative, 0.8), R.SURPRISE_SIZE: 3.0,
                               # The absolute tone points the other way on purpose.
                               R.STANCE: (R.DOVISH if sign < 0 else R.HAWKISH, 0.9),
                               R.STANCE_REVERSED: (R.BUY if relative == R.MORE_HAWKISH
                                                   else R.SELL, 0.8),
                               R.NEW_INFORMATION: 1.0, R.HOLDER_UNAFFECTED: 0.0})
    reading = read_in_context(client)
    assert reading.relative == relative
    assert reading.verdict is not None
    assert reading.verdict.pair == "EURUSD=X" and reading.verdict.sign == sign
    assert reading.verdict.strength > 0


def test_an_in_line_reading_has_no_side_at_all():
    client = ScriptedClient(**{R.RELATIVE_STANCE: (R.IN_LINE, 0.9), R.SURPRISE_SIZE: 3.0})
    assert read_in_context(client).verdict is None


def test_context_strength_is_arithmetic_and_a_priced_in_statement_scores_zero():
    assert R.context_strength(1.0, 1.0, 1.0, 0.0) == 1.0
    assert R.context_strength(0.8, 0.5, 0.5, 0.0) == pytest.approx(0.2)
    assert R.context_strength(1.0, 1.0, 1.0, 1.0) == 0.0
    assert R.context_strength(-2.0, 1.0, 1.0, 0.0) == 0.0


def test_the_absolute_tree_is_untouched_by_the_new_mode():
    client = ScriptedClient(**{R.STANCE: (R.HAWKISH, 0.9), R.NEW_INFORMATION: 1.0,
                               R.MAGNITUDE: 3.0, R.POLICY_RELEVANT: 1.0,
                               R.STANCE_REVERSED: (R.BUY, 0.9), R.HOLDER_UNAFFECTED: 0.0})
    reading = R.Reader(client).read(doc(body="Rates are going up."))
    assert reading.mode == R.ABSOLUTE and reading.relative == "" and reading.context_chars == 0
    assert reading.verdict is not None and reading.verdict.sign == -1
    assert R.RELATIVE_STANCE not in client.seen[0]


def test_the_mock_answers_every_context_question_with_the_right_type():
    client = MockFxClient()
    client.questions = R.round_one_questions("USD", [], mode=R.CONTEXT)
    answers = client.evaluate({"title": "Federal Reserve issues FOMC statement",
                               "body": "decided to raise the target range",
                               "context_before_the_release": "prices roughly 1 "
                                                             "quarter-point hike"}).answers
    assert set(answers) == set(client.questions)
    assert answers[R.RELATIVE_STANCE].choice in R.RELATIVE_OPTIONS
    assert answers[R.SURPRISE_CHANNEL].choice in R.CHANNELS
    assert answers[R.SURPRISE_SIZE].type == "score"
    assert answers[R.EXPECTED_ACTION].choice in R.ACTIONS


@pytest.mark.parametrize("context,body,expected,actual,relative", [
    ("the market prices roughly 2 quarter-point cuts within six months",
     "the Committee decided to maintain the target range for the federal funds rate",
     R.CUT, R.HOLD, R.MORE_HAWKISH),
    ("the market prices roughly 1 quarter-point hike within six months",
     "the Committee decided to raise the target range by 1/4 percentage point",
     R.HIKE, R.HIKE, R.IN_LINE),
    ("the market prices roughly no change within six months",
     "the Committee decided to lower the target range by 1/4 percentage point",
     R.HOLD, R.CUT, R.MORE_DOVISH),
    ("the market prices roughly no change within six months",
     "the Committee does not expect it will be appropriate to reduce the target range "
     "until it has gained greater confidence; it decided to maintain the target range",
     R.HOLD, R.HOLD, R.IN_LINE),
])
def test_the_mocks_rule_reads_the_pricing_line_and_the_decision_verb(
        context, body, expected, actual, relative):
    from jevtrade.fx import mock as M

    assert M.expected_action(context) == expected
    assert M.actual_action(body) == actual
    assert M.relative_stance(expected, actual) == relative


# ------------------------------------------------------------- numeric baselines

def statement(body, ts):
    return doc("Federal Reserve issues FOMC statement", body=body, ts=ts)


def test_bill_surprise_signs_the_decision_against_what_the_bill_priced():
    previous = statement("the Committee decided to maintain the target range for the "
                         "federal funds rate at 5 to 5-1/4 percent.", FOMC_JUL_TS)
    hike = statement("the Committee decided to raise the target range for the federal "
                     "funds rate by 1/4 percentage point to 5-1/4 to 5-1/2 percent.",
                     FOMC_SEP_TS)
    # The bill prices cuts; the statement hiked, so the surprise is hawkish and
    # a hawkish dollar sends EURUSD down.
    rates = C.Rates(as_of="2026-09-15", ts=FOMC_SEP_TS - DAY,
                    values={"ff": 5.33, "6m": 4.90})
    context = C.Context(released_at=FOMC_SEP_TS, rates=rates, previous=previous)
    signals = S.bill_surprise_signals([hike], {hike.id: context})
    assert len(signals) == 1 and signals[0].pair == "EURUSD=X" and signals[0].sign == -1

    # The same bill and a hold: the market priced a cut that did not come, which
    # is still hawkish, but a 52-bp cut against it would not be.
    cut = statement("the Committee decided to lower the target range for the federal "
                    "funds rate by 1/2 percentage point to 4-1/2 to 4-3/4 percent.",
                    FOMC_SEP_TS)
    context = C.Context(released_at=FOMC_SEP_TS, rates=rates, previous=previous)
    signals = S.bill_surprise_signals([cut], {cut.id: context})
    assert len(signals) == 1 and signals[0].sign == +1


def test_bill_surprise_is_silent_without_a_bill_a_funds_rate_or_a_parseable_rate():
    previous = statement("the Committee decided to maintain the target range for the "
                         "federal funds rate at 5 to 5-1/4 percent.", FOMC_JUL_TS)
    here = statement("the Committee decided to do something unspecified.", FOMC_SEP_TS)
    rates = C.Rates(as_of="2026-09-15", ts=FOMC_SEP_TS - DAY, values={"ff": 5.33})
    context = C.Context(released_at=FOMC_SEP_TS, rates=rates, previous=previous)
    assert S.bill_surprise_signals([here], {here.id: context}) == []
    assert S.bill_surprise_signals([here], {}) == []


def test_dots_surprise_signs_next_years_median_against_the_previous_sep():
    here = statement("the Committee decided to maintain the target range.", FOMC_SEP_TS)
    hawkish = C.Dots("u", FOMC_SEP_TS, ("2024", "2025", "2026", "longer run"),
                     (4.4, 3.9, 3.4, 3.0), "September", (4.4, 3.4, 2.9, 2.9))
    signals = S.dots_surprise_signals([here], {here.id: C.Context(
        released_at=FOMC_SEP_TS, projections=hawkish)})
    assert len(signals) == 1 and signals[0].sign == -1   # fewer cuts -> EURUSD down

    dovish = C.Dots("u", FOMC_SEP_TS, ("2024", "2025", "2026", "longer run"),
                    (4.4, 2.9, 3.4, 3.0), "September", (4.4, 3.4, 2.9, 2.9))
    signals = S.dots_surprise_signals([here], {here.id: C.Context(
        released_at=FOMC_SEP_TS, projections=dovish)})
    assert len(signals) == 1 and signals[0].sign == +1

    flat = C.Dots("u", FOMC_SEP_TS, ("2024", "2025", "longer run"), (4.4, 3.4, 3.0),
                  "September", (4.4, 3.4, 3.0))
    assert S.dots_surprise_signals([here], {here.id: C.Context(
        released_at=FOMC_SEP_TS, projections=flat)}) == []
    # A meeting with no projections is not a meeting this arm trades.
    assert S.dots_surprise_signals([here], {here.id: C.Context(released_at=FOMC_SEP_TS)}) == []


def test_a_previous_sep_with_no_cell_for_next_year_is_no_signal():
    here = statement("hold", FOMC_SEP_TS)
    dots = C.Dots("u", FOMC_SEP_TS, ("2026", "2027", "longer run"), (4.1, 4.1, 3.2),
                  "June", (3.8, None, 3.1))
    assert S.dots_surprise_signals([here], {here.id: C.Context(
        released_at=FOMC_SEP_TS, projections=dots)}) == []


def test_statements_are_picked_by_title_and_sorted():
    corpus = [doc("Federal Reserve issues FOMC statement", ts=FOMC_SEP_TS),
              doc("FOMC statement", ts=FOMC_JUL_TS),
              doc("Minutes of the Federal Open Market Committee", ts=FOMC_JUL_TS + DAY),
              doc("Speech by Chair Powell", ts=FOMC_JUL_TS + 2 * DAY, kind=D.SPEECH)]
    picked = S.statements(corpus)
    assert [d.title for d in picked] == ["FOMC statement", "Federal Reserve issues FOMC statement"]


def test_the_parsed_decision_is_hike_hold_or_cut_and_none_when_it_cannot_be_read():
    previous = statement("the Committee decided to maintain the target range for the "
                         "federal funds rate at 5 to 5-1/4 percent.", FOMC_JUL_TS)
    hike = statement("the Committee decided to raise the target range for the federal "
                     "funds rate by 1/4 percentage point to 5-1/4 to 5-1/2 percent.",
                     FOMC_SEP_TS)
    hold = statement("the Committee decided to maintain the target range for the federal "
                     "funds rate at 5 to 5-1/4 percent.", FOMC_SEP_TS)
    assert S.decision_from_rates(hike, previous) == R.HIKE
    assert S.decision_from_rates(hold, previous) == R.HOLD
    assert S.decision_from_rates(previous, hike) == R.CUT
    assert S.decision_from_rates(hike, None) is None
    assert S.decision_from_rates(statement("no rate here", FOMC_SEP_TS), previous) is None


def test_the_confusion_counts_agreement_and_what_it_could_not_parse():
    previous = statement("the Committee decided to maintain the target range for the "
                         "federal funds rate at 5 to 5-1/4 percent.", FOMC_JUL_TS)
    cut = statement("the Committee decided to lower the target range for the federal "
                    "funds rate by 1/4 percentage point to 4-3/4 to 5 percent.", FOMC_SEP_TS)
    unreadable = statement("something happened", FOMC_SEP_TS + DAY)

    def reading(document, said):
        return R.Reading(document=document, kind=R.RATE_DECISION, p_kind=0.9,
                         policy_relevant=0.9, stance=R.NEUTRAL, p_stance=0.5, confidence=0.5,
                         new_information=0.5, magnitude=0.5, guidance_changed=0.5,
                         surprise=0.5, intervention=0.0, confirm=0.0, horizon="",
                         diffs=[], verdict=None, rounds=1, latency_ms=1.0, wall_ms=1.0,
                         input_tokens=1, actual_action=said, expected_action=R.HOLD)

    confusion = S.action_confusion([reading(cut, R.CUT), reading(unreadable, R.HIKE)],
                                   {cut.id: previous, unreadable.id: previous})
    assert confusion.n == 1 and confusion.agreed == 1 and confusion.unparsed == 1
    assert confusion.rows == {(R.CUT, R.CUT): 1}
    wrong = S.action_confusion([reading(cut, R.HOLD)], {cut.id: previous})
    assert wrong.rate == 0.0 and wrong.rows == {(R.CUT, R.HOLD): 1}


def test_the_state_hash_moves_with_the_context_and_is_unchanged_without_one():
    document = statement("hold", FOMC_SEP_TS)
    plain = S.state_hash(document, [], None)
    assert S.state_hash(document, [], None, context="") == plain
    assert S.state_hash(document, [], None, context="priced: two cuts") != plain


def test_signal_disagreements_pair_the_two_arms_and_carry_the_tapes_verdict():
    left = [S.Signal("fed:1", FOMC_SEP_TS, "A statement", "EURUSD=X", -1)]
    right = [S.Signal("fed:1", FOMC_SEP_TS, "A statement", "EURUSD=X", +1),
             S.Signal("fed:2", FOMC_JUL_TS, "Another", "EURUSD=X", +1)]
    outcomes = [
        S.Outcome(signal=left[0], symbol="EURUSD", pre_bps=0.0, release_bar_bps=0.0,
                  fwd_bps={15: -12.0}),
    ]
    split = S.signal_disagreements(left, right, outcomes, horizon=15)
    assert [d.title for d in split] == ["Another", "A statement"]
    statement_row = split[1]
    assert statement_row.bot == [("EURUSD=X", -1)] and statement_row.reader == [("EURUSD=X", +1)]
    # The tape's move is unsigned by either arm: the short made +12 bp, so the
    # pair fell 12 and the short was right.
    assert statement_row.outcomes == {"EURUSD=X": 12.0}


# ------------------------------------------------------------------------- CLI

def test_cli_has_a_context_flag_with_the_documented_defaults():
    from jevtrade.cli import build_parser

    args = build_parser().parse_args(["fx", "--context", "--tape", "dukascopy"])
    assert args.context is True and args.context_chars == 12000
    assert args.func.__name__ == "cmd_fx"


def flat_hour(start, n=60, bid=1.17, spread=0.00002):
    import lzma
    import struct

    body = b"".join(
        struct.Struct(">IIIff").pack(int(i * 60_000), int(round((bid + spread) * 1e5)),
                                     int(round(bid * 1e5)), 1.0, 1.0)
        for i in range(n)
    )
    return lzma.compress(body, format=lzma.FORMAT_ALONE)


def test_cli_context_runs_end_to_end_offline(tmp_path, monkeypatch, capsys):
    """``--context --provider mock --tape dukascopy`` with every fetch stubbed."""
    from jevtrade import cli
    from jevtrade.fx import context as context_module
    from jevtrade.fx import documents as docs_module
    from jevtrade.fx import ticks as ticks_module

    def fed_page(url, timeout=30.0):
        name = url.rsplit("/", 1)[-1]
        if name in {"ne-press.json", "ne-speeches.json", "ne-testimony.json"}:
            return (FIXTURES / name).read_text()
        if "monetary20260916a" in name:
            return (FIXTURES / "fomc-2026-09-16.html").read_text()
        if "monetary20260729a" in name:
            return (FIXTURES / "fomc-2026-07-29.html").read_text()
        return "<html><body><p>" + "nothing much to read here, but long enough.</p></body></html>"

    def context_page(url, timeout=30.0):
        if "datadownload" in url:
            return H15_CSV if "RIFSPFF" not in url else H15_CSV
        if "monetary20260819a" in url:
            return '<html><a href="/monetarypolicy/fomcminutes20260729.htm">HTML</a></html>'
        if "fomcminutes" in url:
            return "<html><body><p>" + MINUTES + "</p></body></html>"
        if "fomcprojtabl20260916" in url:
            return "<html><body><p>" + SEP_2024 + "</p></body></html>"
        raise OSError(f"404 {url}")

    def feed(url):
        stamp = url.rsplit("/datafeed/", 1)[-1]
        _symbol, year, month, day, hour = stamp.split("/")
        when = datetime(int(year), int(month) + 1, int(day), int(hour[:2]), tzinfo=timezone.utc)
        return flat_hour(when.timestamp(), bid=1.17 + (when.timestamp() % 997) / 1e6)

    monkeypatch.setattr(docs_module, "get_text", fed_page)
    monkeypatch.setattr(docs_module, "fetch_body",
                        lambda url: docs_module.body_text(url, fed_page(url)))
    monkeypatch.setattr(context_module, "get_text", context_page)
    monkeypatch.setattr(ticks_module, "fetch_bi5", feed)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    out_file = tmp_path / "context.json"
    code = cli.main([
        "fx", "--context", "--provider", "mock", "--tape", "dukascopy",
        "--since", "2026-07-01", "--until", "2026-09-18", "--latency-sweep",
        "--horizons", "1,15,60", "--cache", str(tmp_path / "cache"),
        "--workers", "2", "--workers-io", "2", "--null-per", "1",
        "--context-chars", "9000", "--out", str(out_file),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "FOMC statements" in printed
    assert "reader-absolute" in printed and "reader-context" in printed
    assert "bill-surprise" in printed and "dots-surprise" in printed
    assert "what the model said the statement did" in printed
    assert "where it put the surprise" in printed
    assert "latency sweep" in printed
    assert "where the absolute and the context readers traded differently" in printed

    record = json.loads(out_file.read_text())
    assert record["mode"] == "context" and record["tree"] == "ctx1"
    assert record["context_chars"] == 9000 and record["statements"] == 2
    assert record["lookahead_refused"] == []
    assert {a["name"] for a in record["arms"]} >= {
        "reader-absolute", "reader-context", "bill-surprise", "dots-surprise",
        "all statements"}
    read = record["statements_read"]
    assert len(read) == 2
    september = max(read, key=lambda r: r["ts"])
    sources = september["context_sources"]
    assert sources and all(s["ts"] < september["ts"] for s in sources if not s["concurrent"])
    assert any(s["kind"] == "dots" and s["concurrent"] and s["ts"] == september["ts"]
               for s in sources)
    assert {s["kind"] for s in sources} >= {"rates", "minutes", "previous_statement"}
    assert september["context_reading"]["mode"] == "context"
    assert september["context_reading"]["relative"] in R.RELATIVE_OPTIONS
    assert september["absolute_reading"]["mode"] == "absolute"
    assert september["context_chars"] <= 9000
