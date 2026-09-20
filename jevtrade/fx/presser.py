"""The press conference, thirty minutes after the statement.

The 2009-2026 run's third-worst call was 2022-11-02: the statement read one way
and the press conference read the other, and the tape followed the press
conference. The statement arm has no vocabulary for that, because the press
conference is a different document with a different timestamp, and this module
is the document.

**The sources.** Every press conference since April 2011 has a transcript at
``mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf`` -- checked here on 2011-04-27
(26 pages, "Chairman Bernanke's Press Conference FINAL"), 2015-09-17,
2019-01-30, 2024-09-18 ("Chair Powell's Press Conference FINAL") and 2026-09-16
(15 pages, "Chairman Warsh's Press Conference PRELIMINARY"). Which meetings had
one is on the calendar pages: ``fomccalendars.htm`` carries the current six
years and ``fomchistorical{YYYY}.htm`` one year each before that, and both link
the press-conference page by its date. Counted over 2011 to 2026-09 that is 95
conferences: three or five a year through 2018, every meeting from 2019.

**A transcript is published later than the conference it records**, usually the
same evening for the preliminary text. So this arm measures *"was it worth
listening to"*, not *"could you have traded it"* -- exactly the caveat the
speeches section already carries. What would make it tradeable is a live
speech-to-text feed, and that is not free.

**Reading the PDF** needs ``pypdf``, which is an *optional* dependency
(``pip install 'jev-trade[pdf]'``). The rest of this repository is standard
library only and stays that way: when ``pypdf`` is not importable this module
says so once and the whole arm is reported **dark**, which is a result -- the
same way the surprise bot reports "not available" for a week with no calendar
snapshot -- and never an exception in the middle of a run.

**When it starts** is a rule, and ``presser_start`` documents which parts of it
were checked; see that function.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

from ..listing.store import Store
from .baseline import word_lean
from .documents import FED_ROOT, UA, get_text, local_string
from .reader import DOVISH, HAWKISH, Reading, VS_CONSISTENT, signed_pair
from .study import Signal
from .ticks import TickTape, log_bps

TRANSCRIPT_URL = FED_ROOT + "/mediacenter/files/FOMCpresconf{date}.pdf"
PRESSER_PAGE = FED_ROOT + "/monetarypolicy/fomcpresconf{date}.htm"
CALENDAR_URL = FED_ROOT + "/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = FED_ROOT + "/monetarypolicy/fomchistorical{year}.htm"

# The first press conference of all. Nothing before it exists to be fetched.
FIRST_PRESSER = "20110427"
# What the reader is shown of each half. Measured over the 95 transcripts in
# the archive, the opening remarks run 2,400 to 14,100 characters with a median
# of 7,700, so this cap carries all of most of them and the first half of the
# longest. The Q&A has a median of 41,300, so the cap is a real cut there: what
# the reader gets is the opening exchanges, which are where the questions about
# the path are asked, and not the hour.
REMARKS_CAP = 9000
QA_CAP = 9000


class PdfUnavailable(RuntimeError):
    """``pypdf`` is not installed, so this arm is dark rather than broken."""


# ------------------------------------------------------------------ the dates


_PRESSER_LINK = re.compile(r"(?i)fomcpres+conf(\d{8})")


def presser_links(raw: str) -> set[str]:
    """Every ``fomcpresconf{date}`` on a calendar page.

    The spelling is matched loosely because the Fed's own January 2026 row
    links ``fomcpressconf20260128.htm``, with two s's, and a strict match would
    drop that meeting for a typo.
    """
    return set(_PRESSER_LINK.findall(raw or ""))


def presser_dates(
    store: Store,
    *,
    since: float,
    until: float,
    fetcher: Callable[[str], str] | None = None,
    day: str = "",
) -> list[str]:
    """The dates that had a press conference, ``YYYYMMDD``, oldest first.

    One page per year before the current calendar page's range, and the
    calendar page itself for the rest. The calendar page is cached per day
    because it gains a link every few weeks; the historical pages are finished
    and cached forever.
    """
    fetch = fetcher or get_text
    stamp = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    first = max(2011, datetime.fromtimestamp(since, timezone.utc).year)
    last = datetime.fromtimestamp(until, timezone.utc).year
    found: set[str] = set()
    try:
        raw = store.cached(f"page:{CALENDAR_URL}:{stamp}", lambda: fetch(CALENDAR_URL))
        found |= presser_links(raw or "")
    except Exception:  # noqa: BLE001 -- one unreachable page is not a failed run
        pass
    for year in range(first, last + 1):
        url = HISTORICAL_URL.format(year=year)
        try:
            raw = store.cached(f"page:{url}", lambda url=url: fetch(url))
        except Exception:  # noqa: BLE001 -- the current years have no historical page
            continue
        found |= presser_links(raw or "")
    low = datetime.fromtimestamp(since, timezone.utc).strftime("%Y%m%d")
    high = datetime.fromtimestamp(until, timezone.utc).strftime("%Y%m%d")
    return sorted(d for d in found if low <= d <= high and d >= FIRST_PRESSER)


# ------------------------------------------------------------------ the PDF


def get_bytes(url: str, timeout: float = 60.0) -> bytes:
    """The raw bytes behind a URL -- a PDF, which ``documents.get_text`` refuses."""
    import urllib.request

    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_transcript(
    store: Store, date: str, *, fetcher: Callable[[str], bytes] | None = None
) -> bytes | None:
    """The transcript PDF's raw bytes, cached base64 in the store, or ``None``.

    The bytes are cached rather than the text, because the text depends on
    which extractor is installed and the bytes do not; a run that gains
    ``pypdf`` later re-reads what it already has without fetching again.
    """
    fetch = fetcher or get_bytes
    url = TRANSCRIPT_URL.format(date=date)
    key = f"pdf:{url}"
    found, encoded = store.get(key)
    if not found:
        try:
            raw = fetch(url)
        except Exception:  # noqa: BLE001 -- a missing transcript is not a failed run
            return None
        encoded = base64.b64encode(raw or b"").decode("ascii")
        store.put(key, encoded)
    try:
        return base64.b64decode(encoded or "")
    except (ValueError, TypeError):
        return None


def pdf_text(raw: bytes) -> str:
    """The transcript's text, or ``PdfUnavailable`` when ``pypdf`` is missing.

    Imported here and not at module scope so that importing this module -- and
    therefore the CLI -- never depends on the extra. ``pypdf`` prints font
    warnings on these files; they are harmless and left alone rather than
    silenced globally.
    """
    try:
        import pypdf
    except Exception as exc:  # noqa: BLE001 -- ImportError, or a broken install
        raise PdfUnavailable(
            "pypdf is not installed; the press-conference arm is dark "
            "(pip install 'jev-trade[pdf]')"
        ) from exc
    reader = pypdf.PdfReader(io.BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# The running header every page carries, in both of its shapes: "Page 3 of 26"
# from 2015 on, a bare "3 of 26" in 2011, and the dated title line that names
# the Chair and whether the transcript is FINAL or PRELIMINARY.
_PAGE_NUMBER = re.compile(r"^\s*(?:Page\s+)?\d+\s+of\s+\d+\s*$", re.I)
_RUNNING_TITLE = re.compile(r"^.{0,80}?Press\s+Conference\s+(?:FINAL|PRELIMINARY)\s*$", re.I)


def strip_headers(text: str) -> str:
    """Drop the running header lines, keeping everything that was said."""
    kept = [
        line for line in (text or "").splitlines()
        if not _PAGE_NUMBER.match(line.replace("\xa0", " "))
        and not _RUNNING_TITLE.match(line.replace("\xa0", " "))
    ]
    return "\n".join(kept)


# A speaker turn starts a line with the speaker's name in capitals and a full
# stop. Four shapes appear across the archive and all four are matched here:
# "QUESTION." (the April 2011 conference only), a reporter's name ("STEVE
# LIESMAN."), a moderator's ("MICHELLE SMITH."), and the Chair's own ("CHAIR
# YELLEN.", "CHAIRMAN POWELL.", "CHAIRMAN WARSH.").
_SPEAKER = re.compile(
    r"(?m)^[ \t]*((?:[A-Z][A-Z.'’\-]*)(?:[ ][A-Z][A-Z.'’\-]*){0,3})\.(?=\s)"
)
_CHAIR = re.compile(r"(?i)^chair")
QUESTION_MARKER = "QUESTION"


def speaker_marks(body: str) -> list[tuple[int, int, str]]:
    """``(start, end, name)`` for every speaker turn, in order.

    The gap after the stop is usually two spaces, which is what keeps a line
    beginning "U.S. " or "FOMC. " from looking like a speaker. It is not always:
    the June 2024 transcript opens "CHAIR POWELL. Good afternoon." with one
    space, and a parser that insisted on two took that conference's opening
    remarks to be a thousand characters out of the middle of the Q&A. So a
    single space is accepted for a name of two words or more, and for the bare
    "QUESTION." marker, and nowhere else.
    """
    out: list[tuple[int, int, str]] = []
    for match in _SPEAKER.finditer(body):
        name = match.group(1)
        wide = body[match.end():match.end() + 2] in ("  ", " \n", "\n\n", "\n ")
        if wide or " " in name or name == QUESTION_MARKER:
            out.append((match.start(), match.end(), name))
    return out
# A moderator's whole turn is a name ("Steve."), so anything this short before
# the next speaker is the moderator handing over, not a question.
MODERATOR_CHARS = 120


@dataclass(frozen=True)
class Transcript:
    """One press conference, split at the first question.

    ``style`` names how the Q&A is marked on this transcript -- ``question``
    for the "QUESTION." marker, ``moderator`` when a press officer calls the
    first reporter, ``reporter`` when the first reporter's name is the marker --
    because that is the thing most likely to change under this parser and it
    should be visible when it does. Counted over the archive: ``question`` once
    ever, on 2011-04-27; ``reporter`` 47 times, from 2011-06-22 on; and
    ``moderator`` 47 times, from 2020-04-29 on.
    """

    date: str
    remarks: str
    qa: str
    marker: str
    style: str
    speakers: int

    @property
    def text(self) -> str:
        return f"{self.remarks}\n{self.qa}".strip()


def split_transcript(text: str, date: str = "") -> Transcript:
    """Opening remarks, then the Q&A, split at the first non-Chair speaker.

    The Chair's prepared remarks always come first and always end with an
    invitation to ask questions; the first speaker who is not the Chair is
    therefore the boundary, whichever of the three marker styles the transcript
    uses. A transcript with no second speaker at all -- which none of the
    archive's 95 has -- comes back as remarks with an empty Q&A rather than as
    an error.
    """
    body = strip_headers(text)
    marks = speaker_marks(body)
    names = {name for _s, _e, name in marks if not _CHAIR.match(name)}
    # The remarks begin at the Chair's marker only when the Chair's marker is
    # the *first* one. If it is not, this parser has missed the opening line --
    # the September 2018 transcripts write it "CHAIRMAN POWELL:" -- and starting
    # at the top of the page is right where starting at a later Chair marker
    # would report a few hundred characters of the Q&A as the opening remarks.
    start = marks[0][0] if marks and _CHAIR.match(marks[0][2]) else 0
    cut = next(
        (i for i, (_s, _e, name) in enumerate(marks) if not _CHAIR.match(name)), None)
    if cut is None:
        return Transcript(date=date, remarks=body[start:].strip(), qa="", marker="",
                          style="none", speakers=0)
    marker = marks[cut][2]
    turn_end = marks[cut + 1][0] if cut + 1 < len(marks) else len(body)
    if marker == QUESTION_MARKER:
        style = "question"
    elif turn_end - marks[cut][1] < MODERATOR_CHARS and cut + 1 < len(marks):
        style = "moderator"
    else:
        style = "reporter"
    return Transcript(
        date=date, remarks=body[start:marks[cut][0]].strip(),
        qa=body[marks[cut][0]:].strip(), marker=marker, style=style,
        speakers=len(names),
    )


# ------------------------------------------------------------------ the clock


def presser_start(statement_ts: float) -> float:
    """When the Chair started, from the statement's own timestamp.

    **A rule, in two parts, and only one of them is measured.**

    From 2013 the statement goes out at 2:00 p.m. ET and the press conference
    begins at 2:30, so the start is the statement plus thirty minutes. That the
    statement is at 2:00 is measured -- every FOMC statement row in the archive
    from 2013-03-20 on carries 14:00 US Eastern, and the Fed's own
    2013-03-13 release announces the change -- but the half hour itself is the
    published schedule and is taken on trust.

    In 2011 and 2012 the statement on a press-conference day went out at
    12:30 p.m. ET and the Chair began at 2:15, so the start is 2:15 p.m. on the
    statement's own Eastern date. That one *is* checked: the April 2011
    press-conference page prints "FOMC Meeting Statement (Released April 27,
    2011 at 12:30 p.m.)" beside "Projections Materials ... (Released April 27,
    2011 at 2:15 p.m.)", and the projections were released as the conference
    opened. The archive's own minute for those statements is 12:35 or 12:40,
    the moment the release was posted rather than the moment it was released,
    which is why the early rule is an absolute time of day and not an offset.

    An unscheduled meeting keeps the thirty-minute offset. For the two March
    2020 briefings that is a guess, and the study prints it as one rather than
    trusting it.
    """
    local = local_string(statement_ts, "fed")
    if int(local[:4]) > 2012:
        return statement_ts + 30 * 60.0
    shift = timedelta(hours=14, minutes=15) - timedelta(
        hours=int(local[11:13]), minutes=int(local[14:16])
    )
    return statement_ts + shift.total_seconds()


# ------------------------------------------------------------------ assembly


@dataclass
class Presser:
    """One press conference as the reader sees it."""

    date: str
    ts: float  # when the Chair started
    statement_ts: float
    url: str
    transcript: Transcript
    # The dots sentences ``context.dots`` wrote for the same meeting, when it
    # had projections. The Chair is asked about them within the first two
    # questions on a projection day, so the reader is given them too.
    dots: list[str] = field(default_factory=list)
    note: str = ""

    def state(self, *, remarks_cap: int = REMARKS_CAP, qa_cap: int = QA_CAP) -> dict[str, Any]:
        """The block that rides in the reader's state next to the statement."""
        return {
            "projections_released_with_the_statement": list(self.dots),
            "starts_utc": datetime.fromtimestamp(self.ts, timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"),
            "starts_local": local_string(self.ts, "fed"),
            "minutes_after_the_statement": round((self.ts - self.statement_ts) / 60.0),
            "opening_remarks": self.transcript.remarks[:remarks_cap],
            "question_and_answer": self.transcript.qa[:qa_cap],
            "qa_marker_style": self.transcript.style,
        }


def presser_for(
    store: Store,
    date: str,
    statement_ts: float,
    *,
    dots: Sequence[str] = (),
    fetcher: Callable[[str], bytes] | None = None,
) -> Presser | None:
    """The press conference of ``date``, or ``None`` when there is no transcript.

    Raises ``PdfUnavailable`` when the bytes are there and nothing can read
    them, so the caller can report the arm dark once rather than per day.
    """
    raw = fetch_transcript(store, date, fetcher=fetcher)
    if not raw:
        return None
    transcript = split_transcript(pdf_text(raw), date=date)
    if not transcript.remarks:
        return None
    return Presser(
        date=date, ts=presser_start(statement_ts), statement_ts=statement_ts,
        url=TRANSCRIPT_URL.format(date=date), transcript=transcript, dots=list(dots),
    )


# ------------------------------------------------------------------ the arms


def reader_signals(
    readings: Sequence[Reading],
    threshold: float,
    starts: dict[str, float],
    *,
    pair: str = "EURUSD",
    min_confidence: float = 0.5,
) -> list[Signal]:
    """The tree's trades, entered when the Chair started rather than at the release.

    ``starts`` maps a statement's document id to its press conference's start,
    which is the only thing that differs from ``study.reader_signals``: the
    verdict, the side and the strength are whatever the tree said.
    """
    out: list[Signal] = []
    for reading in readings:
        when = starts.get(reading.document.id)
        if when is None:
            continue
        for verdict in reading.signals(threshold, min_confidence=min_confidence):
            out.append(Signal(
                code=f"presser:{reading.document.id}", ts=when,
                title=reading.document.title, pair=pair, sign=verdict.sign,
                strength=verdict.strength,
            ))
    return out


def keyword_signals(
    pressers: Sequence[Presser], *, pair: str = "EURUSD", currency: str = "USD"
) -> list[Signal]:
    """The incumbent, pointed at the transcript: hawkish words minus dovish ones."""
    out: list[Signal] = []
    for presser in pressers:
        stance, hawks, doves = word_lean(presser.transcript.text)
        if stance not in (HAWKISH, DOVISH):
            continue
        signed = signed_pair(currency, stance)
        if signed is None:
            continue
        total = hawks + doves
        out.append(Signal(
            code=f"pressbot:{presser.date}", ts=presser.ts,
            title=f"press conference {presser.date}", pair=pair, sign=signed[1],
            strength=min(1.0, abs(hawks - doves) / total if total else 0.0),
        ))
    return out


@dataclass
class Reversal:
    """A day the press conference did not say what the statement said.

    ``statement_bps`` is the tape's own move over the half hour between the
    release and the Chair's first word, and ``presser_bps`` over the hour after
    it -- both unsigned EURUSD log returns, so a day where the two have
    opposite signs is visible at a glance. That is the shape of 2022-11-02.
    """

    date: str
    remarks_vs_statement: str
    qa_vs_remarks: str
    presser_stance: str
    statement_bps: float | None
    presser_bps: float | None
    note: str = ""


def reversals(
    readings: Sequence[Reading],
    pressers: dict[str, Presser],
    tape: TickTape | None,
    *,
    after_min: int = 60,
) -> list[Reversal]:
    """Every day the remarks or the Q&A were not *consistent*, with the tape beside them."""
    out: list[Reversal] = []
    for reading in readings:
        presser = pressers.get(reading.document.id)
        if presser is None:
            continue
        if (reading.remarks_vs_statement == VS_CONSISTENT
                and reading.qa_vs_remarks == VS_CONSISTENT):
            continue
        before = after = None
        if tape is not None:
            release = tape.mid_at(presser.statement_ts)
            start = tape.mid_at(presser.ts)
            later = tape.mid_at(presser.ts + after_min * 60)
            before = log_bps(release, start) if release and start else None
            after = log_bps(start, later) if start and later else None
        out.append(Reversal(
            date=presser.date,
            remarks_vs_statement=reading.remarks_vs_statement,
            qa_vs_remarks=reading.qa_vs_remarks,
            presser_stance=reading.presser_stance,
            statement_bps=before, presser_bps=after,
        ))
    out.sort(key=lambda r: r.date)
    return out


@dataclass
class Coverage:
    """What the fetch found, so "no transcripts" and "no pypdf" stay different."""

    dates: list[str] = field(default_factory=list)
    fetched: int = 0
    parsed: int = 0
    dark: str = ""
    per_year: dict[str, int] = field(default_factory=dict)

    @property
    def lit(self) -> bool:
        return not self.dark


def collect(
    store: Store,
    dates: Sequence[str],
    statement_ts: dict[str, float],
    *,
    dots: dict[str, Sequence[str]] | None = None,
    fetcher: Callable[[str], bytes] | None = None,
    workers: int = 4,
) -> tuple[dict[str, Presser], Coverage]:
    """Fetch and split every transcript in ``dates``; report what came back.

    ``statement_ts`` maps a press-conference date to its statement's release,
    because the start time is computed from it. A date with no statement in the
    window is skipped rather than guessed at.
    """
    import concurrent.futures

    coverage = Coverage(dates=list(dates))
    wanted = [d for d in dates if d in statement_ts]

    def one(date: str) -> tuple[str, Presser | None, str]:
        try:
            return date, presser_for(store, date, statement_ts[date],
                                     dots=(dots or {}).get(date, ()), fetcher=fetcher), ""
        except PdfUnavailable as exc:
            return date, None, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        results = list(pool.map(one, wanted))
    out: dict[str, Presser] = {}
    for date, presser, note in results:
        if note:
            coverage.dark = note
            continue
        coverage.fetched += 1
        if presser is None:
            continue
        coverage.parsed += 1
        out[date] = presser
        coverage.per_year[date[:4]] = coverage.per_year.get(date[:4], 0) + 1
    return out, coverage


__all__ = [
    "CALENDAR_URL", "Coverage", "FIRST_PRESSER", "HISTORICAL_URL", "MODERATOR_CHARS",
    "PRESSER_PAGE", "PdfUnavailable", "Presser", "QA_CAP", "QUESTION_MARKER",
    "REMARKS_CAP", "Reversal", "TRANSCRIPT_URL", "Transcript", "collect",
    "fetch_transcript", "get_bytes", "keyword_signals", "pdf_text", "presser_dates",
    "presser_for", "presser_links", "presser_start", "reader_signals", "reversals",
    "split_transcript", "strip_headers",
]
