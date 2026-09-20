"""The dot plot as a number, and the rule the context run left behind.

The rule is one arm of five over 31 observations, so the point of these tests is
not that it works: it is that the *measurement* is the same measurement on both
halves of the sample. The histogram parser has to reproduce the median the Fed
prints where both exist, the rule's sign has to be the sign of the change in
next year's median and nothing else, the three pairs have to be signed one way,
and the in-sample/out-of-sample split has to be a date and not a choice.

Everything runs offline, from the layouts below -- which are the 2013, 2016 and
December 2012 tables as the pages really carry them, blank cells included.
"""

import json
from datetime import datetime, timezone

import pytest

from jevtrade.fx import dots as X
from jevtrade.listing.store import Store

DAY = 86400.0


def table(head, rows):
    """An HTML table with real empty cells, which is the whole difficulty."""
    cells = "".join(f'<th class="colhead" scope="col">{h}</th>' for h in head)
    body = ""
    for label, values in rows:
        body += f'<tr><td class="stub" scope="row">{label}</td>'
        body += "".join(f'<td class="data horizctr">{v or "&nbsp;"}</td>' for v in values)
        body += "</tr>"
    return (f'<div class="data-table"><table class="pubtables"><thead><tr>{cells}</tr>'
            f"</thead><tbody>{body}</tbody></table></div>")


# Figure 2 of the 2013-06-19 projection page, verbatim in structure: nineteen
# participants, blanks everywhere the level had none, and the longer-run column
# on the right.
SEP_2013 = (
    "<html><body><h4>Figure 2. Overview of FOMC participants' assessments</h4>"
    "<p><strong><span class='tablesubhead'>Appropriate pace of policy firming</span>"
    "</strong><br><span class='tableunit'>Number of participants with projected "
    "targets</span></p>"
    + table(
        ["Target federal funds rate at year-end<br /> (Percent)", "2013", "2014", "2015",
         "Longer Run"],
        [("0.25", ["18", "15", "1", ""]),
         ("0.50", ["1", "", "2", ""]),
         ("0.75", ["", "", "3", ""]),
         ("1.00", ["", "3", "4", ""]),
         ("1.25", ["", "", "2", ""]),
         ("1.50", ["", "1", "3", ""]),
         ("2.00", ["", "", "1", ""]),
         ("3.00", ["", "", "3", ""]),
         ("3.25", ["", "", "", "1"]),
         ("3.50", ["", "", "", "2"]),
         ("3.75", ["", "", "", "1"]),
         ("4.00", ["", "", "", "9"]),
         ("4.25", ["", "", "", "3"]),
         ("4.50", ["", "", "", "3"])],
    )
    + "</body></html>"
)

# A 2016-style page: the printed median row *and* the histogram, which is what
# makes the overlap a validation and not an assumption. The medians below are
# the ones the histogram implies -- 0.875 for 2016, which prints as 0.9.
PRINTED_2016 = (
    "<p>Percent Variable Median 1 Central Tendency 2 Range 3 "
    "2016 2017 2018 Longer run 2016 2017 2018 Longer run 2016 2017 2018 Longer run "
    "Change in real GDP 2.2 2.1 2.0 2.0 2.1&ndash;2.3 1.9&ndash;2.2 1.8&ndash;2.1 "
    "1.8&ndash;2.1 2.0&ndash;2.4 1.7&ndash;2.4 1.6&ndash;2.2 1.7&ndash;2.3 "
    "Federal funds rate 0.9 1.9 3.0 3.3 "
    "0.9&ndash;1.4 1.6&ndash;2.4 2.5&ndash;3.3 3.0&ndash;3.5 "
    "0.6&ndash;1.4 1.6&ndash;2.8 2.1&ndash;3.9 3.0&ndash;4.0 "
    "December projection 1.4 2.4 3.3 3.5 "
    "1.3&ndash;1.5 2.1&ndash;2.9 2.9&ndash;3.4 3.3&ndash;3.5 "
    "0.9&ndash;1.6 1.9&ndash;3.4 2.6&ndash;3.9 3.0&ndash;4.0 "
    "Note: Projections of change in real gross domestic product</p>"
)
SEP_2016 = (
    "<html><body>" + PRINTED_2016
    + "<h4>Figure 2. FOMC participants' assessments of appropriate monetary policy</h4>"
    "<p><span class='tableunit'><strong>Number of participants with projected midpoint "
    "of target range or target level</strong></span></p>"
    + table(
        ["Midpoint of target range<br /> or target level (Percent)", "2016", "2017",
         "2018", "Longer run"],
        [("0.625", ["1", "", "", ""]),
         ("0.875", ["9", "", "", ""]),
         ("1.125", ["3", "", "", ""]),
         ("1.375", ["4", "", "", ""]),
         ("1.625", ["", "4", "", ""]),
         ("1.875", ["", "5", "", ""]),
         ("2.125", ["", "2", "", ""]),
         ("2.375", ["", "2", "3", ""]),
         ("2.625", ["", "1", "3", ""]),
         ("2.875", ["", "2", "2", ""]),
         ("3.000", ["", "", "2", "7"]),
         ("3.125", ["", "1", "3", ""]),
         ("3.250", ["", "", "", "6"]),
         ("3.375", ["", "", "2", ""]),
         ("3.500", ["", "", "", "4"]),
         ("3.875", ["", "", "2", ""])],
    )
    + "</body></html>"
)

# December 2012's table, which labels the rows with the bucket rather than the
# level and lives on the minutes' accessible-figures page.
SEP_2012_BUCKETS = (
    "<html><body><p><strong><span class='tablesubhead'>Appropriate pace of policy firming"
    "</span></strong></p>"
    + table(
        ["Target federal funds rate at year-end<br /> (Percent)", "2012", "2013", "2014",
         "2015", "Longer Run"],
        [("0 - 0.37", ["19", "17", "14", "1", ""]),
         ("0.38 - 0.62", ["", "1", "1", "5", ""]),
         ("0.63 - 0.87", ["", "", "", "3", ""]),
         ("0.88 - 1.12", ["", "1", "", "3", ""]),
         ("1.13 - 1.37", ["", "", "", "2", ""]),
         ("1.38 - 1.62", ["", "", "2", "", ""]),
         ("1.63 - 1.87", ["", "", "1", "", ""]),
         ("1.88 - 2.12", ["", "", "", "1", ""]),
         ("2.38 -2.62", ["", "", "", "1", ""]),
         ("2.63 - 2.87", ["", "", "1", "", ""]),
         ("2.88 - 3.12", ["", "", "", "", "1"]),
         ("3.38 - 3.62", ["", "", "", "1", "1"]),
         ("3.63 - 3.87", ["", "", "", "1", "3"]),
         ("3.88 - 4.12", ["", "", "", "", "5"]),
         ("4.13 - 4.37", ["", "", "", "", "6"]),
         ("4.38 - 4.62", ["", "", "", "1", "3"])],
    )
    + "</body></html>"
)

CALENDAR = """
<html><body>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="1">2015 FOMC
Meetings</a></h4></div>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month col-md-2"><strong>June</strong></div>
  <div class="fomc-meeting__date col-lg-1">16-17*</div>
  <div class="col-lg-3"><a href="/monetarypolicy/fomcpresconf20150617.htm">Press
  Conference</a><strong>Projection Materials</strong>
  <a href="/monetarypolicy/fomcprojtabl20150617.htm">HTML</a></div>
</div>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month col-md-2"><strong>July</strong></div>
  <div class="fomc-meeting__date col-lg-1">28-29</div>
  <div class="col-lg-3"><br></div>
</div>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month col-md-2"><strong>Oct/Nov</strong></div>
  <div class="fomc-meeting__date col-lg-1">31-1</div>
  <div class="col-lg-3"><br></div>
</div>
</div>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="2">2016 FOMC
Meetings</a></h4></div>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month col-md-2"><strong>March</strong></div>
  <div class="fomc-meeting__date col-lg-1">15-16*</div>
  <div class="col-lg-3"><br></div>
</div>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month col-md-2"><strong>August</strong></div>
  <div class="fomc-meeting__date col-lg-1">22 (notation vote)</div>
  <div class="col-lg-3"><br></div>
</div>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month col-md-2"><strong>December</strong></div>
  <div class="fomc-meeting__date col-lg-1">13-14*</div>
  <div class="col-lg-3"><br></div>
</div>
</div>
</div>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="3">2099 FOMC
Meetings</a></h4></div>
<div class="row fomc-meeting">
  <div class="fomc-meeting__month col-md-2"><strong>June</strong></div>
  <div class="fomc-meeting__date col-lg-1">16-17*</div>
  <div class="col-lg-3"><br></div>
</div>
<footer><a href="/monetarypolicy/fomcpresconf20991231.htm">most recent</a></footer>
</body></html>
"""


# ------------------------------------------------------------------ the table


def test_histogram_keeps_the_column_a_count_belongs_to():
    """Blank cells vanish from flattened text; the parser reads rows and cells."""
    histogram = X.parse_histogram(SEP_2013)
    assert histogram is not None
    assert histogram.years == ("2013", "2014", "2015", "longer run")
    # Nineteen participants in every column, which is the arithmetic check the
    # table itself offers.
    assert [histogram.participants(y) for y in histogram.years] == [19, 19, 19, 19]
    # The 0.50 row has one participant in 2013 and two in 2015 and none in 2014:
    # flattened to "0.50 1 2" that is unreadable.
    assert (0.50, 1) in histogram.counts["2013"]
    assert (0.50, 2) in histogram.counts["2015"]
    assert all(level != 0.50 for level, _n in histogram.counts["2014"])


def test_histogram_medians_are_the_tenth_of_nineteen_dots():
    histogram = X.parse_histogram(SEP_2013)
    assert histogram.medians == {"2013": 0.25, "2014": 0.25, "2015": 1.0, "longer run": 4.0}


def test_median_from_counts_odd_and_even():
    odd = [(0.25, 2), (0.50, 1), (1.00, 2)]  # five dots, the third is 0.50
    assert X.median_from_counts(odd) == 0.50
    even = [(0.25, 2), (1.00, 2)]  # four dots, the mean of the two middle
    assert X.median_from_counts(even) == 0.625
    assert X.median_from_counts([]) is None
    assert X.median_from_counts([(0.25, 0)]) is None


def test_a_negative_level_is_a_level():
    """September 2015 had one participant below zero; dropping the row moves the median."""
    assert X.level_of("-0.125") == -0.125
    with_it = X.median_from_counts([(-0.125, 1), (0.25, 1), (0.50, 1)])
    without = X.median_from_counts([(0.25, 1), (0.50, 1)])
    assert with_it == 0.25 and without == 0.375


def test_bucket_labels_read_as_the_quarter_point_they_contain():
    assert X.level_of("0 - 0.37") == 0.25
    assert X.level_of("0.38 - 0.62") == 0.50
    assert X.level_of("2.38 -2.62") == 2.50  # the page really is missing that space
    assert X.level_of("not a rate") is None


def test_december_2012_buckets_parse_to_quarter_points():
    histogram = X.parse_histogram(SEP_2012_BUCKETS)
    assert histogram is not None
    assert histogram.years == ("2012", "2013", "2014", "2015", "longer run")
    assert [histogram.participants(y) for y in histogram.years] == [19, 19, 19, 19, 19]
    assert histogram.medians["2013"] == 0.25
    assert histogram.medians["longer run"] == 4.0


def test_the_midpoint_layout_is_the_same_table():
    """From 2016 the stub is "Midpoint of target range"; nothing else changes."""
    histogram = X.parse_histogram(SEP_2016)
    assert histogram is not None
    assert histogram.years == ("2016", "2017", "2018", "longer run")
    assert histogram.medians["2016"] == 0.875


def test_a_page_with_no_histogram_answers_none():
    assert X.parse_histogram("<html><body><p>no tables here</p></body></html>") is None


# ------------------------------------------------- printed against the histogram


def test_printed_median_row_is_read_by_the_context_parser():
    printed = X.printed_medians(SEP_2016)
    assert printed == {"2016": 0.9, "2017": 1.9, "2018": 3.0, "longer run": 3.3}


def test_histogram_reproduces_the_printed_median():
    """The overlap check: 0.875 of dots prints as 0.9, and has to match it."""
    histogram = X.parse_histogram(SEP_2016)
    projection = X.Projection(
        date="20160316", url="", ts=0.0, years=histogram.years,
        medians=histogram.medians,
        participants={y: histogram.participants(y) for y in histogram.years},
        printed=X.printed_medians(SEP_2016),
    )
    agreed, compared, notes = projection.agreement()
    assert (agreed, compared) == (4, 4) and notes == []


def test_a_disagreement_is_reported_and_not_swallowed():
    projection = X.Projection(
        date="20160316", url="", ts=0.0, years=("2016", "longer run"),
        medians={"2016": 0.875, "longer run": 3.25}, participants={},
        printed={"2016": 0.9, "longer run": 3.2},
    )
    agreed, compared, notes = projection.agreement()
    assert (agreed, compared) == (1, 2)
    assert notes and "longer run" in notes[0] and "3.2" in notes[0]


def test_rounding_is_half_away_from_zero_like_the_printed_table():
    assert X.round_printed(0.875) == 0.9
    assert X.round_printed(2.5625) == 2.6
    assert X.round_printed(0.125) == 0.1


# ------------------------------------------------------------------ the rule


def projection(date, medians, years=None, printed=None, ts=0.0):
    years = years or tuple(medians)
    return X.Projection(date=date, url=f"u/{date}", ts=ts, years=years, medians=medians,
                        participants={y: 19 for y in years}, printed=printed or {})


NOW = projection("20241218", {"2024": 4.375, "2025": 3.875, "2026": 3.375,
                              "2027": 3.125, "longer run": 3.0},
                 years=("2024", "2025", "2026", "2027", "longer run"),
                 ts=1734548400.0)
PREVIOUS = projection("20240918", {"2024": 4.375, "2025": 3.375, "2026": 2.875,
                                   "2027": 2.875, "longer run": 2.875},
                      years=("2024", "2025", "2026", "2027", "longer run"),
                      ts=1726682400.0)


def test_the_rule_is_next_years_median_against_the_same_year_in_the_previous_sep():
    move = X.move(NOW, PREVIOUS, X.RULE)
    assert move.year == "2025"
    assert move.now == 3.875 and move.previous == 3.375
    assert move.sign == +1  # fewer cuts, a stronger dollar


def test_every_variant_reads_its_own_column():
    assert X.move(NOW, PREVIOUS, "current-year").year == "2024"
    assert X.move(NOW, PREVIOUS, "current-year").sign == 0  # unchanged at 4.375
    assert X.move(NOW, PREVIOUS, "two-years-out").year == "2026"
    assert X.move(NOW, PREVIOUS, "two-years-out").sign == +1
    longer = X.move(NOW, PREVIOUS, "longer-run")
    assert longer.year == "longer run" and longer.sign == +1
    total = X.move(NOW, PREVIOUS, "sum-of-years")
    assert total.now == pytest.approx(4.375 + 3.875 + 3.375 + 3.125)
    assert total.sign == +1
    assert X.move(NOW, PREVIOUS, "not-an-arm") is None


def test_a_year_the_previous_sep_never_had_is_no_signal():
    previous = projection("20240918", {"2024": 4.375, "2025": 3.375, "longer run": 2.875},
                          years=("2024", "2025", "longer run"))
    assert X.move(NOW, previous, "two-years-out") is None
    # ... and the shared-columns sum leaves it out rather than counting it zero.
    total = X.move(NOW, previous, "sum-of-years")
    assert total.now == pytest.approx(4.375 + 3.875)


def test_no_move_is_no_trade():
    same = projection("20240918", dict(NOW.medians), years=NOW.years)
    assert X.move(NOW, same, X.RULE).sign == 0
    assert X.signals(X.records([same, NOW]), "EURUSD") == []


# ------------------------------------------------------- the sign convention


def test_one_convention_for_three_pairs():
    """A hawkish dot plot is a stronger dollar: short EURUSD, short GBPUSD, long USDJPY."""
    assert X.side("EURUSD", +1) == -1
    assert X.side("GBPUSD", +1) == -1
    assert X.side("USDJPY", +1) == +1
    # ... and a lower path is the mirror image, with nothing else changed.
    assert [X.side(p, -1) for p in ("EURUSD", "GBPUSD", "USDJPY")] == [+1, +1, -1]


def test_signals_carry_the_side_and_the_size():
    rows = X.records([PREVIOUS, NOW])
    for pair, expected in (("EURUSD", -1), ("GBPUSD", -1), ("USDJPY", +1)):
        signals = X.signals(rows, pair)
        assert len(signals) == 1
        assert signals[0].sign == expected and signals[0].pair == pair
        # Half a point is two quarter-point steps, half of the four that is full size.
        assert signals[0].strength == pytest.approx(0.5)


def test_strength_saturates_at_four_quarter_points():
    assert X.strength(0.25) == pytest.approx(0.25)
    assert X.strength(-1.00) == pytest.approx(1.0)
    assert X.strength(2.00) == pytest.approx(1.0)


# --------------------------------------------------------------- the split


def test_in_sample_is_a_date_and_not_a_choice():
    assert projection("20150617", {}).in_sample is False
    assert projection(X.FIRST_PRINTED_MEDIAN, {}).in_sample is True
    assert projection("20241218", {}).in_sample is True


def test_records_pair_each_meeting_with_the_previous_one():
    early = projection("20150318", {"2015": 0.625, "2016": 1.875, "longer run": 3.75},
                       years=("2015", "2016", "longer run"))
    middle = projection("20150617", {"2015": 0.625, "2016": 1.625, "longer run": 3.75},
                        years=("2015", "2016", "longer run"))
    late = projection("20150917", {"2015": 0.375, "2016": 1.5, "2017": 2.625,
                                   "longer run": 3.5},
                      years=("2015", "2016", "2017", "longer run"))
    rows = X.records([late, early, middle])  # any order in, date order out
    assert [r.date for r in rows] == ["20150617", "20150917"]
    assert [r.previous_date for r in rows] == ["20150318", "20150617"]
    assert [r.in_sample for r in rows] == [False, True]
    assert rows[0].rule.sign == -1 and rows[1].rule.sign == -1


def test_the_first_meeting_has_no_predecessor_and_no_row():
    assert X.records([NOW]) == []


def test_sign_test_is_two_sided_and_exact():
    assert X.sign_test(24, 31) == pytest.approx(0.0029, abs=5e-4)
    assert X.sign_test(5, 10) == 1.0
    assert X.sign_test(0, 0) == 1.0
    assert X.sign_test(10, 10) == pytest.approx(2 / 1024)


# --------------------------------------------------------------- the calendar


def test_calendar_reads_the_asterisk_the_month_and_the_links():
    meetings = X.parse_calendar(CALENDAR)
    assert [m.date for m in meetings] == [
        "2015-06-17", "2015-07-29", "2015-11-01", "2016-03-16", "2016-08-22",
        "2016-12-14", "2099-06-17"]
    assert [m.projection for m in meetings] == [
        True, False, False, True, False, True, True]
    assert meetings[0].presser == "20150617"
    assert all(m.presser == "" for m in meetings[1:])


def test_a_meeting_that_crosses_a_month_is_dated_by_its_last_day():
    meetings = {m.date: m for m in X.parse_calendar(CALENDAR)}
    assert "2015-11-01" in meetings  # "Oct/Nov 31-1"


def test_the_footer_is_not_a_meeting():
    """The sidebar's link to the most recent conference must not stick to the last row."""
    meetings = X.parse_calendar(CALENDAR)
    assert all(m.presser != "20991231" for m in meetings)


def test_next_projection_meetings_are_the_ones_after_today():
    meetings = X.parse_calendar(CALENDAR)
    after = datetime(2015, 7, 1, tzinfo=timezone.utc).timestamp()
    assert [m.date for m in X.next_projection_meetings(meetings, after)] == [
        "2016-03-16", "2016-12-14", "2099-06-17"]
    later = datetime(2100, 1, 1, tzinfo=timezone.utc).timestamp()
    assert X.next_projection_meetings(meetings, later) == []


def test_the_register_is_appended_to_and_not_recomputed():
    stored = [{"date": "20130619", "moves": {}}, {"date": "20150917", "moves": {}}]
    fresh = [{"date": "20150917", "moves": {"next-year": {"sign": -1}}},
             {"date": "20151216", "moves": {}}]
    merged = X.merge_records(stored, fresh)
    assert [row["date"] for row in merged] == ["20130619", "20150917", "20151216"]
    assert merged[1]["moves"]["next-year"]["sign"] == -1


# --------------------------------------------------------------- the fetch


def test_fetch_falls_through_to_the_minutes_page_for_december_2012(tmp_path):
    asked = []

    def fetch(url, timeout=30.0):
        asked.append(url)
        if url.endswith("fomcminutes20121212epa.htm"):
            return SEP_2012_BUCKETS
        raise OSError(f"404 {url}")

    store = Store(tmp_path)
    found, why = X.probe_projection(store, "20121212", ts=1.0, fetcher=fetch)
    assert why == "" and found is not None
    assert found.medians["2013"] == 0.25 and found.printed == {}
    assert asked[-1].endswith("fomcminutes20121212epa.htm")


def test_a_meeting_with_no_projections_says_so(tmp_path):
    def fetch(url, timeout=30.0):
        raise OSError(f"404 {url}")

    found, why = X.probe_projection(Store(tmp_path), "20241107", fetcher=fetch)
    assert found is None and why == "no projection page"


def test_a_page_that_answers_but_does_not_parse_is_a_different_answer(tmp_path):
    def fetch(url, timeout=30.0):
        return "<html><body><p>projections, but as a picture</p></body></html>"

    found, why = X.probe_projection(Store(tmp_path), "20121212", fetcher=fetch)
    assert found is None and "no funds-rate histogram" in why


def test_horizons_gain_a_one_minute_column_unless_asked_otherwise():
    assert X.horizon_default("5,15,30,60") == (1, 5, 15, 30, 60)
    assert X.horizon_default("15") == (15,)


# --------------------------------------------------------------- end to end


def flat_hour(start, n=60, bid=1.17, spread=0.00002, step=0.0):
    import lzma
    import struct

    body = b"".join(
        struct.Struct(">IIIff").pack(
            int(i * 60_000), int(round((bid + step * i + spread) * 1e5)),
            int(round((bid + step * i) * 1e5)), 1.0, 1.0)
        for i in range(n)
    )
    return lzma.compress(body, format=lzma.FORMAT_ALONE)


ARCHIVE = json.dumps([
    {"d": "6/17/2015 2:00:00 PM", "t": "Federal Reserve issues FOMC statement",
     "pt": "Monetary Policy", "l": "/newsevents/pressreleases/monetary20150617a.htm"},
    {"d": "7/29/2015 2:00:00 PM", "t": "Federal Reserve issues FOMC statement",
     "pt": "Monetary Policy", "l": "/newsevents/pressreleases/monetary20150729a.htm"},
    {"d": "9/17/2015 2:00:00 PM", "t": "Federal Reserve issues FOMC statement",
     "pt": "Monetary Policy", "l": "/newsevents/pressreleases/monetary20150917a.htm"},
    {"d": "12/16/2015 2:00:00 PM", "t": "Federal Reserve issues FOMC statement",
     "pt": "Monetary Policy", "l": "/newsevents/pressreleases/monetary20151216a.htm"},
])

# Three projection tables a quarter apart, with next year's median moving down,
# then up: one out-of-sample record and one in-sample one.
PAGES = {
    "20150617": table(
        ["Target federal funds rate at year-end (Percent)", "2015", "2016", "Longer run"],
        [("0.625", ["9", "", ""]), ("0.875", ["8", "", ""]),
         ("1.625", ["", "9", ""]), ("1.875", ["", "8", ""]),
         ("3.750", ["", "", "17"])]),
    "20150917": table(
        ["Target federal funds rate at year-end (Percent)", "2015", "2016", "Longer run"],
        [("0.375", ["9", "", ""]), ("0.625", ["8", "", ""]),
         ("1.375", ["", "9", ""]), ("1.625", ["", "8", ""]),
         ("3.500", ["", "", "17"])]),
    "20151216": table(
        ["Target federal funds rate at year-end (Percent)", "2015", "2016", "Longer run"],
        [("0.375", ["17", "", ""]),
         ("1.375", ["", "9", ""]), ("1.625", ["", "8", ""]),
         ("3.500", ["", "", "17"])]),
}


def test_cli_dots_runs_end_to_end_with_no_model(tmp_path, monkeypatch, capsys):
    """``fx --dots`` with every fetch stubbed and no provider anywhere near it."""
    from jevtrade import cli
    from jevtrade.fx import documents as docs_module
    from jevtrade.fx import dots as dots_module
    from jevtrade.fx import ticks as ticks_module

    def fed_page(url, timeout=30.0):
        if url.endswith("ne-press.json"):
            return ARCHIVE
        if url.endswith(("ne-speeches.json", "ne-testimony.json")):
            return "[]"
        raise OSError(f"404 {url}")

    def dots_page(url, timeout=30.0):
        if url.endswith("fomccalendars.htm"):
            return CALENDAR
        for date, body in PAGES.items():
            if url.endswith(f"fomcprojtabl{date}.htm"):
                return f"<html><body>{body}</body></html>"
        raise OSError(f"404 {url}")

    def feed(url):
        stamp = url.rsplit("/datafeed/", 1)[-1]
        _symbol, year, month, day, hour = stamp.split("/")
        when = datetime(int(year), int(month) + 1, int(day), int(hour[:2]),
                        tzinfo=timezone.utc)
        return flat_hour(when.timestamp(), bid=1.17 + (when.timestamp() % 997) / 1e6)

    monkeypatch.setattr(docs_module, "get_text", fed_page)
    monkeypatch.setattr(dots_module, "get_text", dots_page)
    monkeypatch.setattr(ticks_module, "fetch_bi5", feed)

    out_file = tmp_path / "dots.json"
    code = cli.main([
        "fx", "--dots", "--since", "2015-06-01", "--until", "2015-12-31",
        "--pairs", "EURUSD,USDJPY,GBPUSD", "--tape", "dukascopy",
        "--horizons", "1,15", "--cache", str(tmp_path / "cache"),
        "--workers-io", "2", "--null-per", "1", "--latency-sweep",
        "--out", str(out_file),
    ])
    assert code == 0
    printed = capsys.readouterr().out
    assert "projection tables: 3 parsed" in printed
    assert "histogram against the printed median" in printed
    assert "EURUSD in-sample" in printed and "EURUSD out-of-sample" in printed
    assert "USDJPY pooled" in printed and "GBPUSD pooled" in printed
    assert "variants on EURUSD -- reported, not chosen" in printed
    assert "latency sweep" in printed
    assert "next projection meetings" in printed
    assert "2099-06-17" in printed

    record = json.loads(out_file.read_text())
    assert record["mode"] == "dots" and record["rule"] == "next-year"
    assert record["usd_side"] == {"EURUSD": -1, "GBPUSD": -1, "USDJPY": +1}
    assert [m["date"] for m in record["meetings"]] == ["20150917", "20151216"]
    assert [m["in_sample"] for m in record["meetings"]] == [True, True]
    assert record["meetings"][0]["moves"]["next-year"]["sign"] == -1
    assert {a["name"] for a in record["arms"]} >= {"EURUSD pooled", "USDJPY pooled"}
    assert {a["name"] for a in record["variant_arms"]} == set(X.VARIANTS)
    assert [r["latency_s"] for r in record["latency_sweep"]] == list(X.LATENCIES)
    assert record["next_projection_meetings"] == ["2099-06-17"]


def test_cli_dots_appends_to_an_existing_register(tmp_path, monkeypatch):
    """A second run over a narrower window keeps the meetings the first one wrote."""
    from jevtrade import cli
    from jevtrade.fx import documents as docs_module
    from jevtrade.fx import dots as dots_module

    out_file = tmp_path / "dots.json"
    out_file.write_text(json.dumps({
        "mode": "dots",
        "meetings": [{"date": "20130619", "moves": {"next-year": {"sign": 1}}}],
    }))

    monkeypatch.setattr(docs_module, "get_text",
                        lambda url, timeout=30.0: ARCHIVE if url.endswith("ne-press.json")
                        else "[]")
    monkeypatch.setattr(dots_module, "get_text", lambda url, timeout=30.0: (
        CALENDAR if url.endswith("fomccalendars.htm")
        else next(f"<html><body>{b}</body></html>" for d, b in PAGES.items()
                  if url.endswith(f"fomcprojtabl{d}.htm"))))

    code = cli.main([
        "fx", "--dots", "--since", "2015-06-01", "--until", "2015-12-31",
        "--pairs", "EURUSD", "--tape", "yahoo", "--horizons", "15",
        "--cache", str(tmp_path / "cache"), "--workers-io", "1",
        "--out", str(out_file),
    ])
    assert code == 0
    dates = [m["date"] for m in json.loads(out_file.read_text())["meetings"]]
    assert dates == ["20130619", "20150917", "20151216"]


def test_cli_dots_refuses_a_pair_it_has_no_convention_for(tmp_path, capsys):
    from jevtrade import cli

    code = cli.main(["fx", "--dots", "--pairs", "EURCHF", "--cache", str(tmp_path)])
    assert code == 1
    assert "unknown pair" in capsys.readouterr().err
