"""What the market already had when a statement hit the tape.

The 2009-2026 run measured the reader on 150 FOMC statements and found a coin
flip: the market moves 15 bp at one minute and 37 bp at sixty, and the reader's
direction is right half the time at every horizon. The named failures all have
the same shape. 2024-12-18 was a cut, and the statement reads dovish; the market
took it as hawkish because the dot plot released with it moved 2025 from three
cuts to two. 2022-11-02 reads one way and the press conference an hour later
read the other. 2009-01-28 promised purchases the market had already priced.

A statement is not read in a vacuum. It is read *against* what the market
expected, and the expectation is not in the statement, so a tree asked "is this
a surprise?" has nothing to be surprised against. This module assembles the
expectation, and only the expectation, out of five pieces of public record:

* **the pricing** -- H.15 daily yields the day before, and what the six-month
  bill says about the next six months of the funds rate;
* **the dots** -- the SEP medians released with the statement, against the
  previous SEP's, when the meeting has one;
* **the previous statement** -- the text this one is a diff of;
* **the minutes** -- the last set released before this meeting, sliced to the
  policy-action section and the end of the participants' views;
* **the intermeeting communication** -- what the Chair and everyone else said
  between the two meetings;
* **the drift** -- where EURUSD was 24 hours and 15 minutes before the release.

Two rules are load-bearing and are enforced by code rather than by care:

* **Nothing may post-date the release.** Every item carries the timestamp of
  the thing it is, and ``Context`` refuses to exist if any of them is at or
  after ``released_at``. The one exception is the SEP, which is published at
  the same minute as the statement, is labelled ``concurrent``, and is a real
  part of what the market reacts to.
* **Code owns every number.** The model is handed sentences -- "prices roughly
  two quarter-point cuts within six months" -- and never a spread to subtract,
  in the style of ``discretize.py``. The raw values stay on the dataclasses for
  the audit trail.

What is measured here: the rates, the dots, the release times, the tick drift.
What is assumed: that the H.15 row for the previous business day was known to
the market before the release (it is published at 4:15 p.m. ET the day it
names), and that a document's published timestamp is when its text became
readable -- which is honest for a scheduled statement or minutes and generous
for a speech.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Sequence

from ..listing.store import Store
from .baseline import announced_rate
from .diff import Editions, previous_of
from .documents import FED_ROOT, Document, SPEECH, TESTIMONY, fed_body_text, get_text

# --------------------------------------------------------------------- feeds

H15_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series={series}"
    "&lastobs=&from={start}&to={end}&filetype=csv&label=include&layout=seriescolumn"
    "&type=package"
)
# The Data Download Program addresses a saved selection by an MD5 of its series
# identifiers joined by newlines, and only answers for selections it already
# knows -- a freshly computed hash for an arbitrary basket comes back empty. So
# the package is two known ones: the Treasury constant maturities, and the
# federal funds effective rate on its own. (The brief for this module said the
# first package carried the funds rate too. Probed on 2026-09-20 it does not:
# eleven columns, all of them Treasury yields.)
H15_PACKAGES = {
    "treasuries": "bf17364827e38702b42a58cf8eaa3f78",
    "funds": "646250c87b1afd04cc6774796fc0cec8",
}
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
# The projection table's own page. The March 2022 release is at
# "fomcprojtable" rather than "fomcprojtabl"; both spellings are tried rather
# than leaving one meeting's dots out of the study for a typo.
SEP_URLS = (
    FED_ROOT + "/monetarypolicy/fomcprojtabl{date}.htm",
    FED_ROOT + "/monetarypolicy/fomcprojtable{date}.htm",
)
SEP_URL = SEP_URLS[0]

# H.15 publishes at 4:15 p.m. ET, which is 20:15 or 21:15 UTC; the later one is
# used so the stamp is never optimistic. Any row strictly before the release's
# UTC date is therefore before the release whichever way the clocks fell.
H15_PUBLISHED_UTC_H = 21.25

MISSING = "ND"

# H.15 series descriptions -> the short keys the sentences use.
_SERIES_KEYS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"federal funds effective", re.I), "ff"),
    (re.compile(r"\b1-month\b.*constant maturity", re.I), "1m"),
    (re.compile(r"\b3-month\b.*constant maturity", re.I), "3m"),
    (re.compile(r"\b6-month\b.*constant maturity", re.I), "6m"),
    (re.compile(r"\b1-year\b.*constant maturity", re.I), "1y"),
    (re.compile(r"\b2-year\b.*constant maturity", re.I), "2y"),
    (re.compile(r"\b3-year\b.*constant maturity", re.I), "3y"),
    (re.compile(r"\b5-year\b.*constant maturity", re.I), "5y"),
    (re.compile(r"\b7-year\b.*constant maturity", re.I), "7y"),
    (re.compile(r"\b10-year\b.*constant maturity", re.I), "10y"),
    (re.compile(r"\b20-year\b.*constant maturity", re.I), "20y"),
    (re.compile(r"\b30-year\b.*constant maturity", re.I), "30y"),
)
# The Treasury's own CSV, used only where H.15 has nothing.
_TREASURY_KEYS = {
    "1 mo": "1m", "3 mo": "3m", "6 mo": "6m", "1 yr": "1y", "2 yr": "2y",
    "3 yr": "3y", "5 yr": "5y", "7 yr": "7y", "10 yr": "10y", "20 yr": "20y",
    "30 yr": "30y",
}

STEP_PP = 0.25  # one quarter-point move of the funds rate
ZERO_BOUND_PP = 0.30  # an effective rate below this is the floor, not a level
MEETINGS_PER_SIX_MONTHS = 4.0  # the FOMC meets eight times a year

DATE = "%Y-%m-%d"


class LookaheadError(ValueError):
    """A context item dated at or after the release it is supposed to precede.

    This is not a warning. A study whose "prior context" contains the answer is
    not measuring reading, and the only safe response is to refuse the context.
    """


# ------------------------------------------------------------------ the rates


@dataclass(frozen=True)
class RateTable:
    """Daily rates, in percent, by ``YYYY-MM-DD``. Missing cells are absent.

    ``series`` maps each short key back to the description the source gave it,
    so a run can print what it actually parsed rather than what it hoped for.
    """

    dates: tuple[str, ...]
    rows: dict[str, dict[str, float]]
    series: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.dates)

    def on(self, date: str) -> dict[str, float] | None:
        row = self.rows.get(date)
        return row or None

    def last_on_or_before(self, date: str) -> tuple[str, dict[str, float]] | None:
        for stamp in reversed(self.dates):
            if stamp <= date and self.rows.get(stamp):
                return stamp, self.rows[stamp]
        return None

    def last_before(self, date: str) -> tuple[str, dict[str, float]] | None:
        """The last row *strictly* before a date -- the no-lookahead lookup."""
        for stamp in reversed(self.dates):
            if stamp < date and self.rows.get(stamp):
                return stamp, self.rows[stamp]
        return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace(" ", " ")).strip()


def parse_h15(text: str) -> RateTable:
    """One H.15 download -> a table. ``ND`` is absent, not zero.

    The five header rows are read rather than skipped: the column order of a
    Data Download package is not documented anywhere, so the mapping comes from
    each column's own "Series Description".
    """
    # The byte-order mark has to come off before the CSV reader sees it, or the
    # first field arrives with its quotes still attached and no column is mapped.
    reader = csv.reader(io.StringIO((text or "").lstrip("\ufeff")))
    columns: dict[int, str] = {}
    rows: dict[str, dict[str, float]] = {}
    dates: list[str] = []
    series: dict[str, str] = {}
    for record in reader:
        if not record:
            continue
        head = _normalize(record[0])
        if head == "Series Description":
            for i, cell in enumerate(record[1:], start=1):
                description = _normalize(cell)
                for pattern, key in _SERIES_KEYS:
                    if pattern.search(description):
                        columns[i] = key
                        series[key] = description
                        break
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", head):
            continue  # Unit:, Multiplier:, Currency:, Unique Identifier:, Time Period
        values: dict[str, float] = {}
        for i, cell in enumerate(record[1:], start=1):
            key = columns.get(i)
            raw = _normalize(cell)
            if key is None or not raw or raw.upper() == MISSING:
                continue
            try:
                values[key] = float(raw)
            except ValueError:
                continue
        if head not in rows:
            dates.append(head)
        rows[head] = values
    dates.sort()
    return RateTable(tuple(dates), rows, series)


def parse_treasury(text: str) -> RateTable:
    """The Treasury's own par-yield CSV, the fallback when H.15 is unreachable.

    Its dates are ``MM/DD/YYYY``, newest first, and the 2009 file has no 2- or
    4-month column at all, which is why the columns are read by name here too.
    """
    reader = csv.reader(io.StringIO((text or "").lstrip("\ufeff")))
    try:
        header = next(reader)
    except StopIteration:
        return RateTable((), {}, {})
    columns: dict[int, str] = {}
    series: dict[str, str] = {}
    for i, cell in enumerate(header):
        key = _TREASURY_KEYS.get(_normalize(cell).lower())
        if key is not None:
            columns[i] = key
            series[key] = _normalize(cell)
    rows: dict[str, dict[str, float]] = {}
    for record in reader:
        if not record:
            continue
        try:
            stamp = datetime.strptime(_normalize(record[0]), "%m/%d/%Y").strftime(DATE)
        except ValueError:
            continue
        values: dict[str, float] = {}
        for i, cell in enumerate(record):
            key = columns.get(i)
            raw = _normalize(cell)
            if key is None or not raw:
                continue
            try:
                values[key] = float(raw)
            except ValueError:
                continue
        rows[stamp] = values
    return RateTable(tuple(sorted(rows)), rows, series)


def merge_tables(*tables: RateTable) -> RateTable:
    """Left to right, first writer wins per (date, key) -- H.15 before Treasury."""
    rows: dict[str, dict[str, float]] = {}
    series: dict[str, str] = {}
    for table in tables:
        for key, description in table.series.items():
            series.setdefault(key, description)
        for date, values in table.rows.items():
            target = rows.setdefault(date, {})
            for key, value in values.items():
                target.setdefault(key, value)
    return RateTable(tuple(sorted(rows)), rows, series)


def rates_h15(
    store: Store,
    *,
    start: str = "01/01/2009",
    end: str = "",
    fetcher: Callable[[str], str] | None = None,
) -> RateTable:
    """Both H.15 packages, merged and cached per day.

    The series are revised and extended, so the cache key carries today's date
    rather than pretending the file is immutable the way an archived page is.
    """
    fetch = fetcher or get_text
    end = end or datetime.now(timezone.utc).strftime("%m/%d/%Y")
    day = datetime.now(timezone.utc).strftime(DATE)
    tables: list[RateTable] = []
    for name, package in H15_PACKAGES.items():
        url = H15_URL.format(series=package, start=start, end=end)
        raw = store.cached(f"h15:{name}:{start}:{end}:{day}", lambda url=url: fetch(url))
        tables.append(parse_h15(raw))
    return merge_tables(*tables)


def rates_treasury(
    store: Store,
    years: Iterable[int],
    *,
    fetcher: Callable[[str], str] | None = None,
) -> RateTable:
    """The Treasury fallback, one file per year. No funds rate lives here."""
    fetch = fetcher or get_text
    day = datetime.now(timezone.utc).strftime(DATE)
    tables: list[RateTable] = []
    for year in years:
        url = TREASURY_URL.format(year=year)
        raw = store.cached(f"treasury:{year}:{day}", lambda url=url: fetch(url))
        tables.append(parse_treasury(raw))
    return merge_tables(*tables) if tables else RateTable((), {}, {})


def _utc_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime(DATE)


def _published_ts(date: str) -> float:
    """When the H.15 row for ``date`` was on the wire, in epoch seconds."""
    day = datetime.strptime(date, DATE).replace(tzinfo=timezone.utc)
    return (day + timedelta(hours=H15_PUBLISHED_UTC_H)).timestamp()


@dataclass(frozen=True)
class Rates:
    """The last rates the market had before a release, and the meeting before.

    ``ts`` is when the ``as_of`` row was published, which is what the
    no-lookahead check reads. ``changes`` is in basis points and is computed
    here so no model ever subtracts anything.
    """

    as_of: str
    ts: float
    values: dict[str, float]
    prior_as_of: str = ""
    prior_values: dict[str, float] = field(default_factory=dict)

    @property
    def changes_bp(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in self.values.items():
            before = self.prior_values.get(key)
            if before is not None:
                out[key] = (value - before) * 100.0
        return out

    def change_bp(self, key: str) -> float | None:
        return self.changes_bp.get(key)


def rates_before(
    table: RateTable, released_at: float, *, previous_ts: float | None = None
) -> Rates | None:
    """The last business-day row strictly before the release, plus the last meeting's.

    "Strictly before the release's UTC date" and not "before the release" is
    deliberate: the H.15 row named after the meeting day is published at 4:15
    p.m. ET, two hours *after* a 2 p.m. statement, so the row that shares the
    release's date is a lookahead even though its date is not.
    """
    found = table.last_before(_utc_date(released_at))
    if found is None:
        return None
    as_of, values = found
    prior_as_of, prior_values = "", {}
    if previous_ts is not None:
        earlier = table.last_on_or_before(_utc_date(previous_ts))
        if earlier is not None:
            prior_as_of, prior_values = earlier
    return Rates(as_of=as_of, ts=_published_ts(as_of), values=dict(values),
                 prior_as_of=prior_as_of, prior_values=dict(prior_values))


# -------------------------------------------------------------- the sentences


def quarter_steps(spread_pp: float) -> int:
    """A spread in percentage points -> whole quarter-point moves, rounded."""
    return int(round(spread_pp / STEP_PP))


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _gloss(change_bp: float, quiet_bp: float = 10.0) -> str:
    if change_bp >= quiet_bp:
        return "the market has moved hawkish since then"
    if change_bp <= -quiet_bp:
        return "the market has moved dovish since then"
    return "little changed either way"


def _plausible_target(target: float | None, effective: float | None) -> float | None:
    """Refuse a parsed target range that the effective funds rate contradicts.

    ``baseline.announced_rate`` reads "at 0 to 1/4 percent" as 1.00, because the
    fraction loses its denominator on the way through; that is the wording of
    every statement from December 2008 to December 2015. The effective rate is
    the check: the funds rate trades inside its target range, so a top more than
    60 bp above it or 10 bp below it is a parse, not a policy rate, and the
    sentence falls back to the effective rate and says so.
    """
    if target is None or effective is None:
        return target
    return target if effective - 0.10 <= target <= effective + 0.60 else None


def pricing_sentences(rates: Rates, *, previous_statement: Document | None = None) -> list[str]:
    """Four to six sentences: the level, what is priced, and what moved.

    Numbers become words in the style of ``discretize.py``: the model is told
    "prices roughly two quarter-point cuts within six months", never handed a
    bill yield and a funds rate to subtract. The basis-point changes are spelled
    out because a change of 14 bp and one of 40 bp are different facts and no
    bucket of five labels can carry both; each comes with its own gloss, which
    is the part the model is meant to read.
    """
    out: list[str] = []
    effective = rates.values.get("ff")
    target = None
    if previous_statement is not None:
        target = announced_rate(f"{previous_statement.title}\n{previous_statement.body}")
    target = _plausible_target(target, effective)

    if target is not None:
        level = f"The target range set at the last meeting tops out at {target:.2f} percent"
        if effective is not None:
            level += (f", and the effective funds rate printed {effective:.2f} percent"
                      f" on {rates.as_of}")
        out.append(level + ".")
    elif effective is not None:
        out.append(
            f"The previous statement's target range could not be parsed; the effective "
            f"funds rate printed {effective:.2f} percent on {rates.as_of}."
        )
    else:
        out.append(f"Neither the target range nor the effective funds rate is "
                   f"available for {rates.as_of}.")

    bill = rates.values.get("6m")
    anchor = effective if effective is not None else target
    if bill is not None and anchor is not None:
        steps = quarter_steps(bill - anchor)
        if anchor <= ZERO_BOUND_PP:
            out.append(
                f"Policy is at the zero bound and the six-month bill yields {bill:.2f} percent, "
                f"so there is no room to price cuts; the bill prices "
                + (f"roughly {_plural(abs(steps), 'quarter-point hike')} within six months."
                   if steps > 0 else "no move away from the floor within six months.")
            )
        elif steps == 0:
            out.append(
                f"The six-month bill yields {bill:.2f} percent against an overnight rate of "
                f"{anchor:.2f}: the market prices roughly no change within six months."
            )
        else:
            word = "hike" if steps > 0 else "cut"
            out.append(
                f"The six-month bill yields {bill:.2f} percent against an overnight rate of "
                f"{anchor:.2f}: the market prices roughly "
                f"{_plural(abs(steps), 'quarter-point ' + word)} within six months."
            )
    elif bill is not None:
        out.append(f"The six-month bill yields {bill:.2f} percent; there is nothing "
                   "to price it against.")

    for key, name in (("2y", "two-year yield"), ("10y", "ten-year yield"),
                      ("1y", "one-year yield")):
        value = rates.values.get(key)
        change = rates.change_bp(key)
        if value is None:
            continue
        if change is None:
            out.append(f"The {name} is at {value:.2f} percent; there is no reading "
                       "from the last meeting.")
        else:
            direction = "higher" if change > 0 else ("lower" if change < 0 else "unchanged")
            moved = (f"{abs(change):.0f} bp {direction}" if change else "unchanged")
            tail = f": {_gloss(change)}" if key == "2y" else ""
            out.append(
                f"The {name} is {moved} than on {rates.prior_as_of or 'the last meeting'}, "
                f"at {value:.2f} percent{tail}."
            )
        if len(out) >= 6:
            break
    return out


# ------------------------------------------------------------------- minutes

MINUTES_TITLE = re.compile(r"^\s*Minutes of (?:the )?Federal Open Market Committee\b", re.I)
MINUTES_LINK = re.compile(
    r'href="(?:https?://(?:www\.)?federalreserve\.gov)?(/monetarypolicy/fomcminutes\d{8}\.htm)"',
    re.I,
)
# Section headings, as the Fed has actually written them. 2009 runs the two
# together ("Meeting Participants' Views and Committee Policy Action"); every
# year from 2010 splits them, and "Actions" is sometimes plural.
# Matched case-sensitively on purpose: "participants' views of longer-run
# sustainable rates" appears mid-sentence in the 2008 minutes and is not a
# heading, and slicing a document at a lowercase near-miss loses the section.
POLICY_ACTION = re.compile(r"Committee Policy Action")
PARTICIPANTS_VIEWS = re.compile(r"(?:Meeting )?Participants['\u2019] Views")
# Site chrome that survives tag-stripping at the foot of every Fed page.
MINUTES_TAIL = re.compile(r"(?:Return to top|Back to Top|Last [Uu]pdate:)\b.*$", re.S)

MINUTES_CAP = 4000
VIEWS_TAIL = 2500


@dataclass(frozen=True)
class Excerpt:
    """A dated slice of another document, with the URL it came from."""

    title: str
    url: str
    ts: float
    text: str


def slice_minutes(body: str, *, cap: int = MINUTES_CAP, views_tail: int = VIEWS_TAIL) -> str:
    """The policy-action section plus the tail of the participants' views.

    The rest of a set of minutes is a staff review of data the market has long
    since traded. What is left is the two sections that say what the Committee
    argued about and what it decided, which is the only part a statement can be
    read *against*. A set of minutes whose headings this does not recognise
    falls back to its last ``cap`` characters rather than to nothing.
    """
    text = MINUTES_TAIL.sub("", _normalize(body)).strip()
    if not text:
        return ""
    action = POLICY_ACTION.search(text)
    views = PARTICIPANTS_VIEWS.search(text)
    parts: list[tuple[str, str]] = []
    if views is not None and (action is None or action.start() - views.start() > 200):
        end = action.start() if action is not None else len(text)
        segment = text[views.start():end].strip()
        parts.append(("Participants' views, end of the section", segment[-views_tail:]))
    if action is not None:
        budget = max(cap - sum(len(label) + len(part) + 4 for label, part in parts), 800)
        parts.append(("Committee policy action", text[action.start():].strip()[:budget]))
    if not parts:
        return text[-cap:]
    out = "\n".join(f"{label}: {part}" for label, part in parts)
    return out[:cap]


def minutes_rows(archive_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """FOMC minutes press releases, oldest first. Discount-rate minutes are not these."""
    rows = [r for r in archive_rows if MINUTES_TITLE.match(str(r.get("title") or ""))]
    return sorted(rows, key=lambda r: float(r["ts"]))


def minutes_excerpt(
    store: Store,
    archive_rows: Sequence[dict[str, Any]],
    released_at: float,
    *,
    fetcher: Callable[[str], str] | None = None,
    cap: int = MINUTES_CAP,
) -> Excerpt | None:
    """The last minutes released before ``released_at``, sliced and capped.

    The press release is a stub; the minutes themselves live at a URL named
    after the *meeting's* last day, which cannot be derived from the release
    date, so the link is followed rather than guessed (guessing 404s).
    """
    fetch = fetcher or get_text
    candidates = [r for r in minutes_rows(archive_rows) if float(r["ts"]) < released_at]
    if not candidates:
        return None
    row = candidates[-1]
    try:
        stub = store.cached(f"page:{row['url']}", lambda: fetch(str(row["url"])))
    except Exception:  # noqa: BLE001 -- a missing page is "no minutes", not a crash
        return None
    link = MINUTES_LINK.search(stub or "")
    if link is None:
        return None
    url = FED_ROOT + link.group(1)
    try:
        raw = store.cached(f"page:{url}", lambda: fetch(url))
    except Exception:  # noqa: BLE001
        return None
    text = slice_minutes(fed_body_text(raw or ""), cap=cap)
    if not text:
        return None
    return Excerpt(title=str(row.get("title") or ""), url=url, ts=float(row["ts"]), text=text)


# ------------------------------------------------------- intermeeting talking

# "Chair Jerome H. Powell" now, "Chairman Ben S. Bernanke" before 2014.
CHAIR = re.compile(r"\bchair(?:man|woman|person)?\b", re.I)
VICE = re.compile(r"\bvice\b", re.I)
REMARK_CHARS = 700
MAX_CHAIR = 3
MAX_OTHERS = 8


@dataclass(frozen=True)
class Remark:
    """One speech or testimony between two meetings. ``body`` is empty for the tail."""

    ts: float
    speaker: str
    title: str
    body: str = ""
    chair: bool = False


def is_chair(speaker: str) -> bool:
    """The Chair, not the Vice Chair and not the Vice Chair for Supervision."""
    return bool(CHAIR.search(speaker or "")) and not VICE.search(speaker or "")


def intermeeting_communication(
    documents: Sequence[Document],
    previous_ts: float,
    released_at: float,
    *,
    chairs: int = MAX_CHAIR,
    others: int = MAX_OTHERS,
    body_chars: int = REMARK_CHARS,
) -> list[Remark]:
    """Speeches and testimony strictly inside ``(previous_ts, released_at)``.

    The Chair's words are the ones the market trades, so up to three of them
    carry a body; everything else is a title and a date, which is enough for the
    model to know whether the Committee had been talking and in what volume.
    The window is open at both ends: a speech at the previous meeting's minute
    belongs to that meeting, and one at this release's minute is not context.
    """
    window = [
        d for d in documents
        if d.kind in (SPEECH, TESTIMONY) and previous_ts < d.ts < released_at
    ]
    window.sort(key=lambda d: d.ts)
    chair_docs = [d for d in window if is_chair(d.speaker)][-chairs:] if chairs > 0 else []
    picked = {id(d) for d in chair_docs}
    rest = [d for d in window if id(d) not in picked][-others:] if others > 0 else []
    out = [
        Remark(ts=d.ts, speaker=d.speaker, title=d.title,
               body=_normalize(d.body)[:body_chars], chair=True)
        for d in chair_docs
    ]
    out += [Remark(ts=d.ts, speaker=d.speaker, title=d.title) for d in rest]
    return sorted(out, key=lambda r: r.ts)


# ---------------------------------------------------------------------- dots

# The header runs "Median 1 Central Tendency 2 Range 3" and then either the
# years straight away or the word "Variable" first, depending on the year.
_SEP_YEARS = re.compile(r"Range\s*\d?\s*(?:Variable\s*)?((?:\d{4}\s+){2,6})Longer run", re.I)
# The 2020 tables write a tenth of a percent as ".1", with no leading zero.
_DECIMAL = r"(?:\d+(?:\.\d+)?|\.\d+)"
_SEP_FFR = re.compile(r"Federal funds rate\s+((?:" + _DECIMAL + r"\s+){1,12})")
_SEP_PREVIOUS = re.compile(r"([A-Z][a-z]+)\s+projection\s+((?:" + _DECIMAL + r"\s+){1,12})")
# The longer-run dot is the most stable number in the table: it has never moved
# by three quarter-points between consecutive meetings. An alignment of the
# previous projection's row that does move it by that much is an alignment
# error, not a projection, so it is rejected and a shorter row is tried.
LONGER_RUN_TOLERANCE = 0.75


@dataclass(frozen=True)
class Dots:
    """The SEP's funds-rate medians, this one and the last, as sentences.

    Released at the same minute as the statement, so this is the one piece of
    context that is concurrent rather than prior, and it is labelled as such.
    """

    url: str
    ts: float
    years: tuple[str, ...]
    medians: tuple[float, ...]
    previous_label: str
    # Aligned to ``years``; ``None`` where the previous SEP had no such column.
    previous: tuple[float | None, ...]
    sentences: list[str] = field(default_factory=list)


def _numbers(blob: str, columns: int) -> list[float]:
    """The run of plain decimals, with a leading footnote marker dropped.

    A row can read "Core PCE inflation 4 2.8 2.5 ..." where the 4 is a footnote;
    a bare single digit in front of more values than the table has columns is
    that, not a projection. Everything captured is returned, because where the
    medians stop and the central tendency starts is decided by the caller: a
    central tendency printed as a bare number ("0.1" rather than "0.1-0.4")
    looks exactly like one more median.
    """
    tokens = blob.split()
    while len(tokens) > columns and re.fullmatch(r"[1-9]", tokens[0]):
        tokens = tokens[1:]
    return [float(t) for t in tokens]


def _previous_row(
    tokens: Sequence[float], columns: int, longer_run: float
) -> tuple[float | None, ...]:
    """Pick how many of the captured numbers are the previous projection's.

    The longest alignment that leaves the longer-run dot where it was wins. On a
    table whose central tendency prints bare numbers the greedy read is one
    column too long, and the longer-run check is what catches it.
    """
    for take in range(min(len(tokens), columns), 1, -1):
        aligned = _align_previous(tokens[:take], columns)
        if not aligned:
            continue
        tail = aligned[-1]
        if tail is None or abs(tail - longer_run) <= LONGER_RUN_TOLERANCE:
            return aligned
    return ()


def _align_previous(values: Sequence[float], columns: int) -> tuple[float | None, ...]:
    """Line the previous SEP's medians up with this one's columns.

    The previous projection row is printed under the same headings but has no
    cell for the year this meeting added, and tag-stripping loses empty cells.
    The last value is always the longer run, so a short row fills from the left
    and from the right and leaves the new year blank.
    """
    if len(values) == columns:
        return tuple(values)
    if 2 <= len(values) < columns:
        gap = columns - len(values)
        return tuple(values[:-1]) + (None,) * gap + (values[-1],)
    return ()


def parse_sep(text: str) -> Dots | None:
    """The funds-rate median row and the previous projection's, or ``None``.

    The 2012-and-earlier projection pages have no median row at all -- they
    publish ranges and a histogram -- so "not parseable" is a normal answer for
    a meeting before 2013 and is reported as "no dots context", never as a
    failure.
    """
    flat = _normalize(text)
    years_match = _SEP_YEARS.search(flat)
    ffr = _SEP_FFR.search(flat)
    if years_match is None or ffr is None:
        return None
    years = tuple(years_match.group(1).split()) + ("longer run",)
    captured = _numbers(ffr.group(1), len(years))
    if len(captured) < len(years):
        return None
    medians = captured[: len(years)]
    label, previous = "", ()
    tail = _SEP_PREVIOUS.search(flat, ffr.end() - 1, ffr.end() + 400)
    if tail is not None:
        label = tail.group(1)
        previous = _previous_row(_numbers(tail.group(2), len(years)), len(years), medians[-1])
        if not previous:
            label = ""
    return Dots(url="", ts=0.0, years=years, medians=tuple(medians),
                previous_label=label, previous=previous)


def dots_sentences(dots: Dots) -> list[str]:
    """One sentence per horizon, saying which way the median moved and what that means."""
    out: list[str] = []
    for i, year in enumerate(dots.years):
        now = dots.medians[i]
        where = "the longer run" if year == "longer run" else f"end-{year}"
        was = dots.previous[i] if i < len(dots.previous) else None
        if was is not None:
            if abs(now - was) < 1e-9:
                out.append(f"Median projection for {where} is unchanged at {now:.1f}%.")
                continue
            steps = quarter_steps(abs(now - was))
            if year == "longer run":
                move = "a higher neutral rate" if now > was else "a lower neutral rate"
            else:
                move = "fewer cuts" if now > was else "more cuts"
            size = f" ({_plural(steps, 'quarter-point step')})" if steps else ""
            since = (f" in {dots.previous_label}" if dots.previous_label
                     else " at the last projection")
            out.append(
                f"Median projection for {where} moved to {now:.1f}% from "
                f"{was:.1f}%{since}: {move}{size}."
            )
        else:
            out.append(f"Median projection for {where} is {now:.1f}%.")
    return out


def dots(
    store: Store,
    released_at: float,
    *,
    fetcher: Callable[[str], str] | None = None,
    local_date: str = "",
) -> Dots | None:
    """The projection table published with this statement, or ``None``.

    ``local_date`` is the meeting's US Eastern date as ``YYYYMMDD``; the page is
    named after it. A meeting with no projections answers 404 and that is "no
    dots", not an error.
    """
    fetch = fetcher or get_text
    stamp = local_date or datetime.fromtimestamp(released_at, timezone.utc).strftime("%Y%m%d")
    for template in SEP_URLS:
        url = template.format(date=stamp)
        try:
            raw = store.cached(f"page:{url}", lambda url=url: fetch(url))
        except Exception:  # noqa: BLE001 -- 404 is the common answer here
            continue
        parsed = parse_sep(fed_body_text(raw or "") or _normalize(raw or ""))
        if parsed is not None:
            return Dots(url=url, ts=released_at, years=parsed.years, medians=parsed.medians,
                        previous_label=parsed.previous_label, previous=parsed.previous,
                        sentences=dots_sentences(parsed))
    return None


# --------------------------------------------------------------------- drift


@dataclass(frozen=True)
class Drift:
    """Where the pair was going into the release, from the cached ticks."""

    ts: float
    mid: float
    day_bps: float | None
    quarter_hour_bps: float | None
    sentences: list[str] = field(default_factory=list)


def _log_bps(a: float, b: float) -> float | None:
    import math

    if not (a > 0 and b > 0):
        return None
    return math.log(b / a) * 1e4


def market_drift(tape: Any, released_at: float, *, symbol: str = "EURUSD") -> Drift | None:
    """EURUSD over the 24 hours and the 15 minutes before the release.

    The last quote is taken a millisecond *before* the release so the stamp on
    this item can never equal the release's own. ``None`` means the tape had no
    price -- a holiday, or an hour the feed never served -- which is a normal
    answer and not a reason to drop the statement.
    """
    if tape is None:
        return None
    quote = tape.quote_at(released_at - 0.001)
    if quote is None:
        return None
    mid = quote.mid
    day = tape.mid_at(released_at - 86400.0)
    quarter = tape.mid_at(released_at - 900.0)
    day_bps = _log_bps(day, mid) if day else None
    quarter_bps = _log_bps(quarter, mid) if quarter else None
    sentences: list[str] = []
    if day_bps is not None:
        way = "higher" if day_bps > 0 else "lower"
        sentences.append(
            f"{symbol} is at {mid:.5f}, {abs(day_bps):.0f} bp {way} than 24 hours "
            "before the release."
        )
    else:
        sentences.append(f"{symbol} is at {mid:.5f}; there is no price 24 hours back.")
    if quarter_bps is not None:
        if abs(quarter_bps) < 5:
            mood = "the tape was quiet going in"
        elif abs(quarter_bps) < 15:
            mood = "some positioning ahead of the release"
        else:
            mood = "a large move going in, so something was already being traded"
        way = "up" if quarter_bps > 0 else "down"
        sentences.append(
            f"In the fifteen minutes before the release it moved {abs(quarter_bps):.0f} bp "
            f"{way}: {mood}."
        )
    return Drift(ts=quote.ts, mid=mid, day_bps=day_bps, quarter_hour_bps=quarter_bps,
                 sentences=sentences)


def previous_statement(
    document: Document,
    statements: Sequence[Document],
    *,
    editions: Editions | None = None,
) -> Document | None:
    """The statement this one is read against.

    ``diff.previous_of`` answers first, because that is the edition the diff is
    computed from and the two must agree. It groups by title family, and the Fed
    renamed the release twice ("FOMC statement" to "Federal Reserve issues FOMC
    statement"), which leaves the first statement after each rename with no
    predecessor; the fallback is then simply the last statement before this one,
    which is what a desk would have had open.
    """
    found = previous_of(document, editions if editions is not None else list(statements))
    if found is not None:
        return found
    earlier = [d for d in statements if d.ts < document.ts and d.body]
    return max(earlier, key=lambda d: d.ts) if earlier else None


# ------------------------------------------------------------------- context


@dataclass(frozen=True)
class Source:
    """One item of context and the moment it became public."""

    kind: str
    label: str
    ts: float
    concurrent: bool = False

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def check_no_lookahead(sources: Sequence[Source], released_at: float) -> None:
    """Raise if anything in the context is dated at or after the release.

    The concurrent SEP is allowed to be stamped at the release minute itself
    and nothing is allowed past it.
    """
    for source in sources:
        # A concurrent item may share the release's minute; everything else has
        # to be strictly earlier. The comparison is >= rather than a subtracted
        # epsilon, because an epsilon under a microsecond disappears into the
        # float at 2026 epoch seconds and the check would pass on equality.
        late = source.ts > released_at if source.concurrent else source.ts >= released_at
        if late:
            raise LookaheadError(
                f"{source.kind} ({source.label}) is dated {source.when}, "
                f"at or after the release it is context for"
            )


PREVIOUS_CHARS = 4500


@dataclass
class Context:
    """Everything a desk had before the statement, with a stamp on every piece.

    Construction runs the no-lookahead check, so a ``Context`` that exists is
    one whose every item predates the release (bar the concurrent SEP). The
    parts are kept as objects rather than as one string so a run can report
    coverage -- how many statements had minutes, dots, a Chair speech -- without
    re-parsing prose.
    """

    released_at: float
    pricing: list[str] = field(default_factory=list)
    rates: Rates | None = None
    projections: Dots | None = None
    previous: Document | None = None
    minutes: Excerpt | None = None
    communication: list[Remark] = field(default_factory=list)
    drift: Drift | None = None
    sources: list[Source] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sources = list(self.sources) or self._sources()
        check_no_lookahead(self.sources, self.released_at)

    def _sources(self) -> list[Source]:
        out: list[Source] = []
        if self.rates is not None:
            out.append(Source("rates", f"H.15 close of {self.rates.as_of}", self.rates.ts))
        if self.projections is not None:
            out.append(Source("dots", "SEP released with this statement",
                              self.projections.ts, concurrent=True))
        if self.previous is not None:
            out.append(Source("previous_statement", self.previous.title, self.previous.ts))
        if self.minutes is not None:
            out.append(Source("minutes", self.minutes.title, self.minutes.ts))
        for remark in self.communication:
            kind = "chair_remark" if remark.chair else "remark"
            out.append(Source(kind, f"{remark.speaker}: {remark.title}"[:90], remark.ts))
        if self.drift is not None:
            out.append(Source("drift", "last EURUSD quote before the release", self.drift.ts))
        return out

    @property
    def has(self) -> dict[str, bool]:
        return {
            "rates": self.rates is not None,
            "dots": self.projections is not None,
            "previous": self.previous is not None,
            "minutes": self.minutes is not None,
            "chair": any(r.chair for r in self.communication),
            "communication": bool(self.communication),
            "drift": self.drift is not None,
        }

    def as_text(self, char_budget: int = 12000) -> str:
        """The six blocks, in a fixed order, inside ``char_budget`` characters.

        The order is fixed so the model sees the same shape every time: what is
        priced, what the dots did, the text this one is a diff of, the minutes,
        who said what between the meetings, and where the pair was. The two long
        free-text blocks -- the previous statement and the minutes -- share
        whatever the short ones leave, and the whole thing is truncated as a
        guarantee rather than as a hope.
        """
        when = datetime.fromtimestamp(self.released_at, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        blocks: list[str] = [f"WHAT THE MARKET ALREADY HAD, as of {when}"]
        blocks.append(self._pricing_block())
        blocks.append(self._dots_block())
        # Placeholders for the two long blocks, filled once the short ones are
        # measured, so the budget is spent on text and not on headings.
        blocks.append("")
        blocks.append("")
        blocks.append(self._communication_block())
        blocks.append(self._drift_block())

        spare = max(char_budget - len("\n\n".join(blocks)) - 200, 0)
        blocks[3] = self._previous_block(min(spare // 2, PREVIOUS_CHARS))
        blocks[4] = self._minutes_block(spare // 2)
        return "\n\n".join(b for b in blocks if b)[:char_budget]

    def _pricing_block(self) -> str:
        head = "RATES AND WHAT THEY PRICE" + (f" (H.15 close of {self.rates.as_of})"
                                              if self.rates is not None else "")
        body = "\n".join(f"- {s}" for s in self.pricing) or "- No rates were available."
        return f"{head}\n{body}"

    def _dots_block(self) -> str:
        head = "PROJECTIONS RELEASED WITH THIS STATEMENT"
        if self.projections is not None and self.projections.sentences:
            return (head + " (concurrent, not prior)\n"
                    + "\n".join(f"- {s}" for s in self.projections.sentences))
        return head + "\n- No projections at this meeting, or none this code could read."

    def _previous_block(self, budget: int) -> str:
        if self.previous is None or budget <= 0:
            return "THE PREVIOUS STATEMENT\n- Not found in this window."
        when = datetime.fromtimestamp(self.previous.ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return f"THE PREVIOUS STATEMENT ({when})\n" + _normalize(self.previous.body)[:budget]

    def _minutes_block(self, budget: int) -> str:
        if self.minutes is None or budget <= 0:
            return "THE LAST MINUTES BEFORE THIS MEETING\n- None released in this window."
        when = datetime.fromtimestamp(self.minutes.ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return (f"THE LAST MINUTES BEFORE THIS MEETING (released {when})\n"
                + self.minutes.text[:budget])

    def _communication_block(self) -> str:
        head = "WHAT OFFICIALS SAID BETWEEN THE MEETINGS"
        lines: list[str] = []
        for remark in self.communication:
            day = datetime.fromtimestamp(remark.ts, timezone.utc).strftime("%Y-%m-%d")
            line = f"- {day} {remark.speaker or 'unattributed'}: {remark.title}"
            if remark.body:
                line += f"\n    {remark.body}"
            lines.append(line)
        return head + "\n" + ("\n".join(lines)
                              or "- Nothing on the record between the two meetings.")

    def _drift_block(self) -> str:
        head = "THE PAIR INTO THE RELEASE"
        if self.drift is None or not self.drift.sentences:
            return head + "\n- No ticks around this release."
        return head + "\n" + "\n".join(f"- {s}" for s in self.drift.sentences)


def context_for(
    document: Document,
    *,
    store: Store,
    table: RateTable,
    archive_rows: Sequence[dict[str, Any]],
    documents: Sequence[Document],
    previous: Document | None = None,
    tape: Any = None,
    fetcher: Callable[[str], str] | None = None,
    local_date: str = "",
) -> Context:
    """Assemble one statement's context. Every piece is optional; the stamps are not."""
    previous_ts = previous.ts if previous is not None else document.ts - 45 * 86400
    rates = rates_before(table, document.ts, previous_ts=previous_ts)
    return Context(
        released_at=document.ts,
        pricing=pricing_sentences(rates, previous_statement=previous) if rates else [],
        rates=rates,
        projections=dots(store, document.ts, fetcher=fetcher, local_date=local_date),
        previous=previous,
        minutes=minutes_excerpt(store, archive_rows, document.ts, fetcher=fetcher),
        communication=intermeeting_communication(documents, previous_ts, document.ts),
        drift=market_drift(tape, document.ts) if tape is not None else None,
    )


__all__ = [
    "Context", "Dots", "Drift", "Excerpt", "H15_PACKAGES", "H15_URL", "LookaheadError",
    "MINUTES_CAP", "RateTable", "Rates", "Remark", "SEP_URL", "SEP_URLS", "Source",
    "TREASURY_URL",
    "check_no_lookahead", "context_for", "dots", "dots_sentences", "intermeeting_communication",
    "is_chair", "market_drift", "merge_tables", "minutes_excerpt", "minutes_rows",
    "parse_h15", "parse_sep", "parse_treasury", "previous_statement",
    "pricing_sentences", "quarter_steps",
    "rates_before", "rates_h15", "rates_treasury", "slice_minutes",
]
