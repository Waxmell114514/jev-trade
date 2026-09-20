"""The dot plot as a number: the funds-rate histogram, its median, and the rule.

The statements-in-context run left one signed number outside two standard
errors of zero. On the 31 projection meetings since September 2015 whose table
prints a funds-rate *median*, the sign of the change in **next year's** median
against the previous SEP was worth +19 +- 6 bp of EURUSD at fifteen minutes,
24 of 31 right. That is one arm of five over 31 observations, which is a
finding to test and not a strategy, so this module exists to test it: out of
sample, on three pairs, at six entry latencies, with the variants reported next
to it rather than chosen between.

**Where the out-of-sample data comes from.** The SEP has printed a funds-rate
median only since September 2015. From January 2012 -- the first meeting with
dots at all -- to June 2015 the same page carries a *histogram* instead: rate
levels down the rows, years across the columns, the number of participants at
each level in the cells. That is the same information; the median is arithmetic
on it. So the fourteen or so meetings before the printed median are a genuine
out-of-sample set for a rule found on the printed one, and the parser is
validated on the overlap: from September 2015 both the printed row and the
histogram exist, and the histogram-derived median has to reproduce the printed
one.

**Three page layouts, all of them met in the archive** (probed, not assumed):

* 2012 spells the histogram's headings in Title Case ("Appropriate Pace of
  Policy Firming", "Target Federal Funds Rate at Year-End"); 2013 to 2015 use
  sentence case. A case-sensitive search for the 2013 wording therefore misses
  the January 2012 page, which is where the brief for this module expected a
  different table and found none.
* From March 2016 the heading is "Number of participants with projected
  midpoint of target range or target level" and the stub column is midpoints
  ("0.875"), because the target became a range. Nothing else changes.
* December 2012 has no ``fomcprojtabl`` page at all. Its SEP was published
  inside the minutes, and its histogram is on the accessible-figures page
  ``fomcminutes20121212epa.htm``, whose stub column is *buckets* ("0.38 -
  0.62") rather than levels. A bucket is read as the single quarter-point value
  it contains.

**What "the median" means here.** The median of the multiset of dots: the
middle one when the number of participants is odd, the mean of the two middle
when it is even. That is also what the Fed's printed median does -- its own
note says the median is "the middle projection when the number of projections
is odd, and the average of the two middle projections when the number ... is
even" -- except that the printed number is rounded to one decimal, so 0.875
prints as 0.9. The agreement check rounds the histogram median the same way
before comparing.

**The rule, and the variants.** The rule is the one the run found and is not
re-chosen here:

    sign(next year's median now - next year's median at the previous SEP)

positive is fewer cuts, a stronger dollar, and short EURUSD. The current-year
median, the two-years-out median, the longer-run dot and the sum of the year
medians are computed and printed **as variants**, so that a reader can see
whether the finding is a fragile pick among five. None of them is ever
promoted: the point of printing them is that they are not the rule.

**The sign convention, one line per pair and tested.** A hawkish dot plot is a
stronger dollar. The dollar is the quote side of EURUSD and GBPUSD and the base
of USDJPY, so hawkish is short EURUSD, short GBPUSD and long USDJPY.
"""

from __future__ import annotations

import html as _html
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from ..listing.store import Store
from .context import SEP_URLS, parse_sep, quarter_steps
from .documents import FED_ROOT, fed_archive, fed_body_text, get_text, local_string
from .study import Signal

# The projection table, in every place the archive puts it. The first two are
# the ordinary page (March 2022 uses the second spelling); the third is the
# accessible-figures page of the minutes, which is where December 2012's dots
# live and nowhere else.
PROJECTION_URLS = SEP_URLS + (FED_ROOT + "/monetarypolicy/fomcminutes{date}epa.htm",)
CALENDAR_URL = FED_ROOT + "/monetarypolicy/fomccalendars.htm"

# The first SEP with a printed funds-rate median row. Everything strictly
# before it is out of sample for a rule found on the printed medians.
FIRST_PRINTED_MEDIAN = "20150917"
# The first SEP with dots at all.
FIRST_DOTS = "20120125"

FOMC_STATEMENT = re.compile(r"fomc statement", re.I)

# Which way a stronger dollar pushes each pair. Spelled out rather than derived
# from a currency table, because getting it backwards inverts the whole study
# while leaving every number plausible. The dollar is the quote side of EURUSD
# and GBPUSD and the base side of USDJPY.
USD_SIDE: dict[str, int] = {"EURUSD": -1, "GBPUSD": -1, "USDJPY": +1}
PAIRS = ("EURUSD", "USDJPY", "GBPUSD")

# The rule, and the four variants that are reported and never chosen.
RULE = "next-year"
VARIANTS = ("current-year", "two-years-out", "longer-run", "sum-of-years")
ARMS = (RULE,) + VARIANTS

# The sweep the brief asks for: five minutes is "can a person who reads the
# table by hand still catch it", 300 s is the same question with the page load.
LATENCIES = (0.0, 1.0, 5.0, 30.0, 120.0, 300.0)
HORIZONS = (1, 5, 15, 30, 60)


# ------------------------------------------------------------------ HTML tables


_TABLE = re.compile(r"(?is)<table\b[^>]*>(.*?)</table\s*>")
_ROW = re.compile(r"(?is)<tr\b[^>]*>(.*?)</tr\s*>")
_CELL = re.compile(r"(?is)<(t[dh])\b[^>]*>(.*?)</\1\s*>")
_TAG = re.compile(r"(?s)<[^>]+>")


def cell_text(fragment: str) -> str:
    """One cell's text, with the tags gone and an empty cell still empty."""
    text = _html.unescape(_TAG.sub(" ", fragment))
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def parse_tables(raw: str) -> list[list[list[str]]]:
    """Every ``<table>`` as rows of cells, blanks included.

    Flattening the page to text loses the blank cells, and with them the column
    a count belongs to: "0.50 1 2" could be one participant in the first year
    and two in the third, or one in the second and two in the fourth. So the
    histogram is read by walking ``<tr>`` and ``<td>``/``<th>`` and keeping the
    positions.
    """
    return [
        [[cell_text(body) for _tag, body in _CELL.findall(row)] for row in _ROW.findall(table)]
        for table in _TABLE.findall(raw)
    ]


_YEAR = re.compile(r"^(19|20)\d{2}$")
_LONGER_RUN = re.compile(r"(?i)^longer[\s\-]*run$")
LONGER_RUN = "longer run"
# The stub column of the funds-rate histogram, in all three of its spellings.
_RATE_STUB = re.compile(
    r"(?i)target\s+federal\s+funds\s+rate|midpoint\s+of\s+target\s+range|target\s+level"
)
# A level can be negative: the September 2015 table carries a "-0.125" row -- one
# participant who projected a funds rate below zero -- and dropping it moves that
# page's 2016 median a whole eighth away from the median the same page prints.
_LEVEL = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")
_BUCKET = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[-‐-―]\s*(-?\d+(?:\.\d+)?)\s*$")
_COUNT = re.compile(r"^\s*(\d+)\s*$")
QUARTER = 0.25


def level_of(label: str) -> float | None:
    """The rate a histogram row stands for, or ``None`` if the row is not one.

    Most tables label the row with the level itself ("0.25", "0.875"). December
    2012 labels it with the bucket the dots were counted into ("0.38 - 0.62"),
    and a bucket holds exactly one quarter-point value, which is the value the
    participants actually wrote down; that is what is returned. A bucket that
    somehow holds none or several falls back to its midpoint, which is an
    approximation and is why the single-value case is tried first.
    """
    plain = _LEVEL.match(label or "")
    if plain:
        return float(plain.group(1))
    bucket = _BUCKET.match(label or "")
    if bucket is None:
        return None
    low, high = float(bucket.group(1)), float(bucket.group(2))
    if high < low:
        low, high = high, low
    # Strictly inside: the December 2012 bins run "0 - 0.37", "0.38 - 0.62",
    # each one drawn around a single quarter point, and the first of them has
    # both 0 and 0.25 on its closed interval and only 0.25 in its interior.
    first = math.floor(low / QUARTER + 1e-9) + 1
    last = math.ceil(high / QUARTER - 1e-9) - 1
    inside = [step * QUARTER for step in range(first, last + 1)]
    return inside[0] if len(inside) == 1 else (low + high) / 2.0


def median_from_counts(pairs: Sequence[tuple[float, int]]) -> float | None:
    """The median of the dots, given ``(level, how many participants)`` rows.

    The middle dot when the count is odd, the mean of the two middle when it is
    even -- which is the same definition the Fed's own note gives for the
    median it started printing in September 2015. It is not rounded here; the
    printed one is rounded to a tenth, so the comparison rounds rather than the
    measurement.
    """
    dots = [level for level, count in pairs for _ in range(max(0, count))]
    if not dots:
        return None
    return float(statistics.median(sorted(dots)))


def round_printed(value: float) -> float:
    """A tenth, rounded half away from zero -- the way the table prints 0.875 as 0.9."""
    scaled = abs(value) * 10.0
    return math.copysign(math.floor(scaled + 0.5) / 10.0, value)


@dataclass(frozen=True)
class Histogram:
    """One meeting's funds-rate dot counts, by column.

    ``years`` are the column labels in printed order: the meeting's own year
    first, then each following year, then ``"longer run"``. ``counts`` maps a
    column to its ``(level, participants)`` rows.
    """

    years: tuple[str, ...]
    counts: dict[str, list[tuple[float, int]]]

    def participants(self, year: str) -> int:
        return sum(n for _level, n in self.counts.get(year, ()))

    def median(self, year: str) -> float | None:
        return median_from_counts(self.counts.get(year, ()))

    @property
    def medians(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for year in self.years:
            value = self.median(year)
            if value is not None:
                out[year] = value
        return out


def parse_histogram(raw: str) -> Histogram | None:
    """The funds-rate histogram on a projection page, or ``None`` if there is none.

    The table is found by its stub header rather than by the prose above it,
    because the prose is "Appropriate pace of policy firming" until 2015 and
    "Number of participants with projected midpoint of target range or target
    level" afterwards, while the stub header always names the funds rate.
    """
    for table in parse_tables(raw):
        if not table:
            continue
        head = table[0]
        if len(head) < 2 or not _RATE_STUB.search(head[0]):
            continue
        years: list[str] = []
        columns: list[int] = []
        for i, label in enumerate(head[1:], start=1):
            if _YEAR.match(label):
                years.append(label)
                columns.append(i)
            elif _LONGER_RUN.match(label):
                years.append(LONGER_RUN)
                columns.append(i)
        if len(years) < 2:
            continue
        counts: dict[str, list[tuple[float, int]]] = {year: [] for year in years}
        for row in table[1:]:
            if not row:
                continue
            level = level_of(row[0])
            if level is None:
                continue
            for year, i in zip(years, columns):
                if i >= len(row):
                    continue
                hit = _COUNT.match(row[i])
                if hit and int(hit.group(1)) > 0:
                    counts[year].append((level, int(hit.group(1))))
        if any(counts.values()):
            return Histogram(years=tuple(years), counts=counts)
    return None


def printed_medians(raw: str) -> dict[str, float]:
    """The funds-rate *median* row as the page prints it, empty before September 2015.

    This is ``context.parse_sep`` -- the same reader the context study uses --
    re-keyed by year, so the two modules cannot disagree about what the printed
    median is.
    """
    parsed = parse_sep(fed_body_text(raw) or re.sub(r"\s+", " ", raw))
    if parsed is None:
        return {}
    out: dict[str, float] = {}
    for year, value in zip(parsed.years, parsed.medians):
        out[LONGER_RUN if _LONGER_RUN.match(year) else year] = value
    return out


# ------------------------------------------------------------------ projections


@dataclass(frozen=True)
class Projection:
    """One projection meeting's dots, from the histogram and from the printed row."""

    date: str  # the meeting's US Eastern date, YYYYMMDD
    url: str
    ts: float  # the statement's release timestamp
    years: tuple[str, ...]
    medians: dict[str, float]  # from the histogram -- the one convention
    participants: dict[str, int]
    printed: dict[str, float] = field(default_factory=dict)

    @property
    def in_sample(self) -> bool:
        """Whether the printed median this rule was found on exists for this meeting."""
        return self.date >= FIRST_PRINTED_MEDIAN

    def year_at(self, offset: int) -> str | None:
        """The column ``offset`` years out; column 0 is always the meeting's own year."""
        years = [y for y in self.years if y != LONGER_RUN]
        return years[offset] if 0 <= offset < len(years) else None

    def agreement(self) -> tuple[int, int, list[str]]:
        """``(agreed, compared, notes)`` between the histogram and the printed row."""
        agreed, compared, notes = 0, 0, []
        for year, value in sorted(self.printed.items()):
            if year not in self.medians:
                notes.append(f"{self.date} {year}: printed {value:.2f}, no histogram column")
                continue
            compared += 1
            if abs(round_printed(self.medians[year]) - value) < 1e-9:
                agreed += 1
            else:
                notes.append(
                    f"{self.date} {year}: printed {value:.2f}, "
                    f"histogram {self.medians[year]:.3f}"
                )
        return agreed, compared, notes


def probe_projection(
    store: Store,
    date: str,
    *,
    ts: float = 0.0,
    fetcher: Callable[[str], str] | None = None,
) -> tuple[Projection | None, str]:
    """``(projection, why not)`` for one meeting date.

    The two failures are different and are reported differently: **no page** is
    the ordinary answer for the five meetings a year that have no projections,
    and **a page with no readable histogram** is this parser losing to a
    layout, which is the thing worth printing. The page cache key is the one
    ``context.dots`` uses, so a run that has already assembled the context pays
    nothing here.
    """
    fetch = fetcher or get_text
    seen = ""
    for template in PROJECTION_URLS:
        url = template.format(date=date)
        try:
            raw = store.cached(f"page:{url}", lambda url=url: fetch(url))
        except Exception:  # noqa: BLE001 -- a 404 is the common answer here
            continue
        seen = url
        histogram = parse_histogram(raw or "")
        medians = histogram.medians if histogram is not None else {}
        if not medians:
            continue
        return Projection(
            date=date, url=url, ts=ts, years=histogram.years, medians=medians,
            participants={y: histogram.participants(y) for y in histogram.years},
            printed=printed_medians(raw or ""),
        ), ""
    return None, (f"page at {seen} carries no funds-rate histogram" if seen
                  else "no projection page")


def fetch_projection(
    store: Store,
    date: str,
    *,
    ts: float = 0.0,
    fetcher: Callable[[str], str] | None = None,
) -> Projection | None:
    """The projection table published with the statement of ``date``, or ``None``."""
    return probe_projection(store, date, ts=ts, fetcher=fetcher)[0]


def statement_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The FOMC statements among archive rows, oldest first, one per meeting date.

    The same title filter the rest of ``fx/`` uses. A meeting day that carries
    two statements -- January 2012's longer-run goals statement, September
    2014's policy-normalisation principles -- keeps the earliest, which is the
    decision.
    """
    kept: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: r["ts"]):
        if not FOMC_STATEMENT.search(row.get("title") or ""):
            continue
        kept.setdefault(local_string(row["ts"], "fed")[:10].replace("-", ""), row)
    return [kept[date] for date in sorted(kept)]


# ------------------------------------------------------------------ the rule


@dataclass(frozen=True)
class Move:
    """What one arm says at one meeting: the two medians, the gap and its sign."""

    arm: str
    year: str
    now: float
    previous: float

    @property
    def delta(self) -> float:
        return self.now - self.previous

    @property
    def sign(self) -> int:
        """+1 a higher path (hawkish for the dollar), -1 lower, 0 unchanged."""
        if abs(self.delta) < 1e-9:
            return 0
        return 1 if self.delta > 0 else -1


def move(now: Projection, previous: Projection, arm: str = RULE) -> Move | None:
    """One arm's reading of a meeting against the previous projection meeting.

    Every arm compares **the same calendar year** in the two tables, which is
    what "the median moved" means: the current table's next-year column against
    the previous table's column for that same year, not against its own
    next-year column. ``None`` when the previous SEP has no cell for it.
    """
    offsets = {RULE: 1, "current-year": 0, "two-years-out": 2}
    if arm in offsets:
        year = now.year_at(offsets[arm])
        if year is None or year not in now.medians or year not in previous.medians:
            return None
        return Move(arm=arm, year=year, now=now.medians[year],
                    previous=previous.medians[year])
    if arm == "longer-run":
        if LONGER_RUN not in now.medians or LONGER_RUN not in previous.medians:
            return None
        return Move(arm=arm, year=LONGER_RUN, now=now.medians[LONGER_RUN],
                    previous=previous.medians[LONGER_RUN])
    if arm == "sum-of-years":
        # The year columns both tables have. The current table gains a year at
        # the meeting that extends the horizon, and that year has no previous
        # value, so it is left out of both sides rather than counted as zero.
        shared = [y for y in now.years
                  if y != LONGER_RUN and y in now.medians and y in previous.medians]
        if len(shared) < 2:
            return None
        return Move(arm=arm, year="+".join(shared),
                    now=sum(now.medians[y] for y in shared),
                    previous=sum(previous.medians[y] for y in shared))
    return None


def side(pair: str, usd_sign: int) -> int:
    """``+1`` long the pair, ``-1`` short it, for a dollar-positive dot move."""
    return USD_SIDE.get(pair, -1) * usd_sign


def strength(delta: float) -> float:
    """A quarter-point move is a quarter of full size; four or more is full.

    The same scaling the context study's ``dots-surprise`` arm used, kept so the
    two runs' numbers mean the same thing.
    """
    return min(1.0, abs(quarter_steps(delta)) / 4.0)


@dataclass
class Record:
    """One meeting as the study sees it: the medians, every arm's sign, the tape."""

    date: str
    ts: float
    url: str
    in_sample: bool
    previous_date: str
    years: tuple[str, ...]
    medians: dict[str, float]
    previous_medians: dict[str, float]
    printed: dict[str, float]
    participants: dict[str, int]
    moves: dict[str, Move]
    outcomes: dict[str, float] = field(default_factory=dict)  # pair -> +15m bps

    @property
    def rule(self) -> Move | None:
        return self.moves.get(RULE)


def records(projections: Sequence[Projection]) -> list[Record]:
    """Every projection meeting that has a predecessor, with all five arms on it."""
    ordered = sorted(projections, key=lambda p: p.date)
    out: list[Record] = []
    for i, now in enumerate(ordered):
        if i == 0:
            continue
        previous = ordered[i - 1]
        moves = {arm: m for arm in ARMS if (m := move(now, previous, arm)) is not None}
        out.append(Record(
            date=now.date, ts=now.ts, url=now.url, in_sample=now.in_sample,
            previous_date=previous.date, years=now.years, medians=dict(now.medians),
            previous_medians=dict(previous.medians), printed=dict(now.printed),
            participants=dict(now.participants), moves=moves,
        ))
    return out


def signals(rows: Sequence[Record], pair: str, *, arm: str = RULE) -> list[Signal]:
    """The arm's trades on one pair: entry at the statement, side from the sign."""
    out: list[Signal] = []
    for row in rows:
        found = row.moves.get(arm)
        if found is None or found.sign == 0 or not row.ts:
            continue
        out.append(Signal(
            code=f"dots:{arm}:{row.date}", ts=row.ts,
            title=f"FOMC projections {row.date}", pair=pair,
            sign=side(pair, found.sign), strength=strength(found.delta),
        ))
    return out


def sign_test(wins: int, n: int) -> float:
    """Two-sided exact sign test against a coin, from ``math.comb``."""
    if n <= 0:
        return 1.0
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


# ------------------------------------------------------------------ the register


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_PANEL = re.compile(r'(?is)<a id="\d+">\s*(\d{4})\s+FOMC\s+Meetings')
_MONTH_CELL = re.compile(r"(?is)fomc-meeting__month[^>]*>\s*<strong>([^<]*)</strong>")
_DATE_CELL = re.compile(r"(?is)fomc-meeting__date[^>]*>([^<]*)<")
_FOOTER = re.compile(r"(?i)<footer\b|<div[^>]*\bclass=\"[^\"]*\bfooter\b")
# "fomcpresconf20240918", and January 2026's "fomcpressconf20260128" -- one
# page on the Fed's own calendar spells it with two s's.
_PRESSER_LINK = re.compile(r"(?i)fomcpres+conf(\d{8})")
_PROJECTION_LINK = re.compile(r"(?i)fomcprojtabl\w*(\d{8})")


@dataclass(frozen=True)
class CalendarMeeting:
    """One row of the FOMC calendar: when it decides, and what it publishes."""

    date: str  # YYYY-MM-DD of the decision day, the last day of the meeting
    projection: bool  # the calendar's asterisk
    presser: str  # the press-conference page's date stamp, "" when there is none

    @property
    def when(self) -> datetime:
        return datetime.strptime(self.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_calendar(raw: str) -> list[CalendarMeeting]:
    """The meeting schedule from ``fomccalendars.htm``, oldest first.

    The decision day is the **last** day of the printed range, which is what
    the statement is dated; a range that crosses a month boundary ("Apr/May
    30-1") takes the second month. An asterisk on the date marks a projection
    meeting, and that mark is the only forward-looking thing on the page: a
    meeting a year out has no links at all, so the asterisk is what says
    whether the rule will have something to trade.
    """
    markers = [(m.start(), "year", m.group(1)) for m in _PANEL.finditer(raw)]
    markers += [(m.start(), "month", m.group(1)) for m in _MONTH_CELL.finditer(raw)]
    # The last meeting of the last panel would otherwise run to the end of the
    # file and pick up the sidebar's link to the most recent press conference.
    markers += [(m.start(), "stop", "") for m in _FOOTER.finditer(raw)]
    markers.sort()
    out: list[CalendarMeeting] = []
    year = 0
    for i, (start, kind, value) in enumerate(markers):
        if kind == "year":
            year = int(value)
            continue
        if kind == "stop":
            year = 0
            continue
        if not year:
            continue
        end = markers[i + 1][0] if i + 1 < len(markers) else len(raw)
        block = raw[start:end]
        date_cell = _DATE_CELL.search(block)
        if date_cell is None:
            continue
        text = _html.unescape(date_cell.group(1))
        days = [int(d) for d in re.findall(r"\d+", text)]
        if not days:
            continue
        months = [_MONTHS[p[:3].lower()] for p in re.split(r"[/\s]+", value.strip())
                  if p[:3].lower() in _MONTHS]
        if not months:
            continue
        # A range that ends on a lower day than it started crosses the month.
        month = months[-1] if (len(months) > 1 and days[-1] < days[0]) else months[0]
        try:
            when = datetime(year, month, days[-1], tzinfo=timezone.utc)
        except ValueError:
            continue
        presser = _PRESSER_LINK.search(block)
        out.append(CalendarMeeting(
            date=when.strftime("%Y-%m-%d"),
            projection="*" in text or bool(_PROJECTION_LINK.search(block)),
            presser=presser.group(1) if presser else "",
        ))
    out.sort(key=lambda m: m.date)
    return out


def calendar(
    store: Store, *, fetcher: Callable[[str], str] | None = None, day: str = ""
) -> list[CalendarMeeting]:
    """The calendar page, cached per day -- it gains a link every few weeks."""
    fetch = fetcher or get_text
    stamp = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = store.cached(f"page:{CALENDAR_URL}:{stamp}", lambda: fetch(CALENDAR_URL))
    return parse_calendar(raw or "")


def next_projection_meetings(
    meetings: Sequence[CalendarMeeting], after: float, *, limit: int = 6
) -> list[CalendarMeeting]:
    """The projection meetings still ahead -- the forward test, as dates."""
    cutoff = datetime.fromtimestamp(after, timezone.utc).strftime("%Y-%m-%d")
    return [m for m in meetings if m.projection and m.date > cutoff][:limit]


RULE_SENTENCE = (
    "The rule: on a projection meeting, take the sign of the change in the median "
    "federal-funds projection for next year against the previous SEP; positive is "
    "fewer cuts, a stronger dollar, so short EURUSD, short GBPUSD and long USDJPY "
    "at the statement, out at the horizon."
)


# ------------------------------------------------------------------ the register file


def record_to_dict(row: Record) -> dict[str, Any]:
    return {
        "date": row.date, "ts": row.ts, "url": row.url, "in_sample": row.in_sample,
        "previous_date": row.previous_date, "years": list(row.years),
        "medians": row.medians, "previous_medians": row.previous_medians,
        "printed": row.printed, "participants": row.participants,
        "moves": {
            arm: {"year": m.year, "now": m.now, "previous": m.previous,
                  "delta": m.delta, "sign": m.sign}
            for arm, m in row.moves.items()
        },
        "outcomes": row.outcomes,
    }


def merge_records(
    stored: Iterable[dict[str, Any]], fresh: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep every meeting a previous ``--out`` file had, replaced where this run has it.

    The register is meant to be appended to: a run over the last two years and a
    run over the whole archive should leave the same file behind, so a later
    invocation after the next projection meeting adds one row rather than
    recomputing fifty.
    """
    by_date = {row.get("date"): row for row in stored if row.get("date")}
    by_date.update({row["date"]: row for row in fresh})
    return [by_date[date] for date in sorted(by_date)]


def previous_window(since: float, *, days: int = 200) -> float:
    """How far before the window the archive has to be read for a previous SEP.

    The rule needs the meeting *before* the first one in the window, and the
    projection meetings are a quarter apart, so two hundred days is one spare
    quarter and is cheap: the archive rows are one cached fetch however wide
    the window.
    """
    return since - days * 86400.0


def eastern_date(ts: float) -> str:
    return local_string(ts, "fed")[:10].replace("-", "")


def day_epoch(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def describe_window(rows: Sequence[Record]) -> str:
    if not rows:
        return "no projection meetings in range"
    return (f"{len(rows)} projection meetings, {pretty_date(rows[0].date)} to "
            f"{pretty_date(rows[-1].date)}")


def pretty_date(stamp: str) -> str:
    """``"20150917"`` -> ``"2015-09-17"``."""
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}" if len(stamp) == 8 else stamp


def horizon_default(given: str, fallback: str = "5,15,30,60") -> tuple[int, ...]:
    """The dots study wants a one-minute column; an explicit ``--horizons`` wins."""
    if given == fallback:
        return HORIZONS
    return tuple(int(h) for h in given.split(","))


def meeting_dates(
    store: Store,
    *,
    since: float,
    until: float,
    fetcher: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Every FOMC statement in the window, as ``{"date", "ts", "title"}``, oldest first."""
    rows, _skipped = fed_archive(store, previous_window(since), until=until, fetcher=fetcher)
    out = []
    for row in statement_rows(rows):
        out.append({"date": eastern_date(row["ts"]), "ts": row["ts"], "title": row["title"]})
    return out


__all__ = [
    "ARMS", "CALENDAR_URL", "CalendarMeeting", "FIRST_DOTS", "FIRST_PRINTED_MEDIAN",
    "HORIZONS", "Histogram", "LATENCIES", "LONGER_RUN", "Move", "PAIRS",
    "PROJECTION_URLS", "Projection", "RULE", "RULE_SENTENCE", "Record", "USD_SIDE",
    "VARIANTS", "calendar", "cell_text", "day_epoch", "describe_window",
    "eastern_date", "fetch_projection", "horizon_default", "level_of",
    "median_from_counts", "meeting_dates", "merge_records", "move",
    "next_projection_meetings", "parse_calendar", "parse_histogram", "parse_tables",
    "pretty_date", "probe_projection", "previous_window", "printed_medians",
    "record_to_dict", "records", "round_printed", "side", "sign_test", "signals",
    "statement_rows", "strength",
]
