"""Official text from four central banks, with a timestamp you can trade on.

The Fed is the primary source because it is the only one of the four that
publishes a deep archive with a time of day: three JSON files (press releases,
speeches, testimony) going back to 2006, each row carrying a US Eastern local
timestamp. Everything else here is RSS, which means the most recent fifteen to
fifty items and nothing before them.

Three things are load-bearing and are therefore parsed, not assumed:

* **The clock.** ``ne-press.json`` dates are US Eastern *local* time with no
  offset, so a statement released at 2:00 p.m. is 18:00 UTC in summer and 19:00
  UTC in winter. Getting that wrong moves every measurement by an hour. 468 of
  the 4633 press rows (all of them older than 2010) carry a date with no time at
  all; those are dropped, not guessed at.
* **The body.** The substantive text of a Fed page sits in one
  ``col-xs-12 col-sm-8 col-md-8`` div. It is extracted by balancing ``<div>``
  tags rather than by searching for a closing marker, because the marker the
  older pages used (``last-update``) is not on the current ones.
* **What is policy and what is not.** 1331 of those press rows are enforcement
  actions and 1099 are bank regulatory policy; neither moves a currency. The
  ``pt`` field separates them, so the filter is a field lookup, not a judgment,
  and the model never sees the noise.
"""

from __future__ import annotations

import concurrent.futures
import email.utils
import html
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Sequence

from ..listing.announcements import get_json
from ..listing.store import Store

FED_ROOT = "https://www.federalreserve.gov"
FED_ARCHIVES = {
    "press": f"{FED_ROOT}/json/ne-press.json",
    "speeches": f"{FED_ROOT}/json/ne-speeches.json",
    "testimony": f"{FED_ROOT}/json/ne-testimony.json",
}
RSS_FEEDS = {
    "ecb": "https://www.ecb.europa.eu/rss/press.html",
    "boj": "https://www.boj.or.jp/en/rss/whatsnew.xml",
    "boe": "https://www.bankofengland.co.uk/rss/news",
}
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CALENDAR_INDEX = "calendar:index"

ISSUER_CURRENCY = {"fed": "USD", "ecb": "EUR", "boj": "JPY", "boe": "GBP"}
ISSUERS = tuple(ISSUER_CURRENCY)

UA = {"User-Agent": "Mozilla/5.0 (compatible; jev-trade research)"}

# Kinds. Fed press releases take the slug of their ``pt`` field; speeches and
# testimony are tagged by which archive they came from; RSS items are classified
# by title, since the feeds carry no type field.
MONETARY_POLICY = "monetary_policy"
SPEECH = "speech"
TESTIMONY = "testimony"
PRESS_RELEASE = "press_release"
POLICY_KINDS = (MONETARY_POLICY, SPEECH, TESTIMONY)

_POLICY_TITLE = re.compile(
    r"monetary polic|interest rate|bank rate|policy rate|money market operation|"
    r"statement on|mpc |minutes|summary of opinions|press conference|"
    r"asset purchase|quantitative|decision|rate decision",
    re.I,
)
_SPEECH_TITLE = re.compile(r"speech|interview|remarks|address|lecture|testimon|panel", re.I)


@dataclass(frozen=True)
class Document:
    id: str
    issuer: str
    kind: str
    ts: float  # epoch seconds, UTC
    title: str
    body: str
    url: str
    speaker: str = ""
    currency: str = ""

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.ts, timezone.utc)


@dataclass
class Collection:
    """What ``collect`` found, and what it refused to use.

    The skipped count is part of the result rather than a log line: a study
    that silently drops a tenth of its feed is not measuring what it says.
    """

    documents: list[Document] = field(default_factory=list)
    skipped_no_time: int = 0
    per_issuer: dict[str, int] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)


# ------------------------------------------------------------------- clocks
#
# zoneinfo needs a tz database, which a slim container may not have. The two
# rules below are the only ones this module needs, and they are simple enough
# to state exactly, so the fallback is a fallback and not a guess.


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
    """The ``n``-th ``weekday`` (Monday=0) of a month; ``n=-1`` is the last one."""
    if n > 0:
        first = datetime(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    last = datetime(year, month, 28) + timedelta(days=4)
    last = last - timedelta(days=last.day)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def us_eastern_offset(naive: datetime) -> timedelta:
    """UTC offset for a *wall clock* US Eastern time.

    DST runs from the second Sunday of March at 02:00 local to the first Sunday
    of November at 02:00 local.
    """
    start = _nth_weekday(naive.year, 3, 6, 2) + timedelta(hours=2)
    end = _nth_weekday(naive.year, 11, 6, 1) + timedelta(hours=2)
    return timedelta(hours=-4) if start <= naive < end else timedelta(hours=-5)


def eu_offset(naive: datetime, standard_hours: int) -> timedelta:
    """UTC offset for a wall clock in the EU summer-time regime.

    Summer time runs from the last Sunday of March to the last Sunday of
    October, switching at 01:00 UTC on both dates.
    """
    start = _nth_weekday(naive.year, 3, 6, -1) + timedelta(hours=1 + standard_hours)
    end = _nth_weekday(naive.year, 10, 6, -1) + timedelta(hours=2 + standard_hours)
    summer = start <= naive < end
    return timedelta(hours=standard_hours + (1 if summer else 0))


# issuer -> (IANA zone, fallback offset function)
ZONES: dict[str, tuple[str, Callable[[datetime], timedelta]]] = {
    "fed": ("America/New_York", us_eastern_offset),
    "ecb": ("Europe/Berlin", lambda d: eu_offset(d, 1)),
    "boe": ("Europe/London", lambda d: eu_offset(d, 0)),
    "boj": ("Asia/Tokyo", lambda _d: timedelta(hours=9)),
}


def _localize(naive: datetime, issuer: str) -> datetime:
    """Attach the issuer's zone to a naive wall clock, tz database or not."""
    zone, fallback = ZONES[issuer]
    try:
        from zoneinfo import ZoneInfo

        return naive.replace(tzinfo=ZoneInfo(zone))
    except Exception:
        return naive.replace(tzinfo=timezone(fallback(naive)))


_FED_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})(?:\s+(\d{1,2}):(\d{2}):(\d{2})\s*([AP])M)?$")


def parse_fed_date(text: str, issuer: str = "fed") -> float | None:
    """``"9/18/2026 11:00:00 AM"`` -> epoch seconds. ``None`` without a time."""
    match = _FED_DATE.match((text or "").strip())
    if not match or match.group(4) is None:
        return None
    month, day, year, hour, minute, second, half = match.groups()
    hour = int(hour) % 12 + (12 if half == "P" else 0)
    naive = datetime(int(year), int(month), int(day), hour, int(minute), int(second))
    return _localize(naive, issuer).timestamp()


def parse_pubdate(text: str) -> float | None:
    """RFC 2822 ``pubDate`` -> epoch seconds. Offsets and ``GMT`` both work."""
    try:
        parsed = email.utils.parsedate_to_datetime((text or "").strip())
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:  # "-0000" means "no zone stated": treat as UTC
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def local_string(ts: float, issuer: str) -> str:
    """The issuer's own wall clock, which is how its releases are scheduled."""
    naive = datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=None)
    zone, fallback = ZONES.get(issuer, ("UTC", lambda _d: timedelta(0)))
    try:
        from zoneinfo import ZoneInfo

        return datetime.fromtimestamp(ts, ZoneInfo(zone)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return (naive + fallback(naive)).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------- html to text

_BODY_DIV = re.compile(r'<div\s+class="col-xs-12 col-sm-8 col-md-8"\s*>', re.I)
_LAST_UPDATE = re.compile(r'<div\s+class="col-xs-12 col-sm-8 col-md-8 last-update"', re.I)
_DIV_TAG = re.compile(r"<div\b[^>]*>|</div\s*>", re.I)
_PARAGRAPH = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S | re.I)
_STRIP = re.compile(r"<(script|style)\b.*?</\1\s*>", re.S | re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
# Boilerplate that every CMS wraps its articles in. Dropping it is best effort:
# what survives is what the model is asked to read, so chrome costs tokens and
# buys confusion.
_CHROME = re.compile(
    r"cookie|javascript|enable js|skip to (main )?content|newsletter|subscribe|"
    r"follow us|all rights reserved|search term|privacy (policy|notice)|"
    r"accessibility|sitemap|^share$|^menu$|browser",
    re.I,
)

TAIL_MARKERS = ("Implementation Note issued", "For media inquiries")


def strip_tags(segment: str) -> str:
    segment = _STRIP.sub(" ", segment)
    segment = _COMMENT.sub(" ", segment)
    segment = re.sub(r"<[^>]+>", " ", segment)
    return re.sub(r"\s+", " ", html.unescape(segment)).strip()


def _balanced_div(raw: str, start: int) -> str:
    depth = 1
    for match in _DIV_TAG.finditer(raw, start):
        depth += 1 if match.group(0)[1] != "/" else -1
        if depth == 0:
            return raw[start : match.start()]
    return raw[start:]


def fed_body_text(raw: str) -> str:
    """The article text of a federalreserve.gov page, without the site chrome.

    The current pages have no closing marker, so the div is closed by balancing
    tags. The Implementation Note tail is dropped: it is a separate release, on
    a separate page, and repeating it would make every statement look unchanged
    in the part that matters least.
    """
    match = _BODY_DIV.search(raw)
    if match:
        segment = _balanced_div(raw, match.end())
    else:
        end = _LAST_UPDATE.search(raw)
        segment = raw[: end.start()] if end else raw
    text = strip_tags(segment)
    for marker in TAIL_MARKERS:
        cut = text.find(marker)
        if cut > 0:
            text = text[:cut]
    return text.strip()


def generic_body_text(raw: str) -> str:
    """Paragraph text from any HTML page, with the obvious chrome dropped."""
    parts: list[str] = []
    for chunk in _PARAGRAPH.findall(_STRIP.sub(" ", raw)):
        text = strip_tags(chunk)
        if len(text) >= 40 and not _CHROME.search(text):
            parts.append(text)
    if parts:
        return " ".join(parts)
    return strip_tags(raw)[:20000]


def body_text(url: str, raw: str) -> str:
    return fed_body_text(raw) if "federalreserve.gov" in url else generic_body_text(raw)


# ----------------------------------------------------------------- fetching


def get_text(url: str, timeout: float = 30.0) -> str:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def fetch_body(url: str) -> str:
    """The readable text behind a link, or ``""`` when there is none.

    BoJ links are mostly PDFs and several issuers link to spreadsheets; there is
    no PDF reader in the standard library, so those documents are carried by
    their title alone and the reader is told the body is empty rather than
    handed a blob of binary.
    """
    if re.search(r"\.(pdf|xlsx?|csv|zip|docx?)(\?|$)", url, re.I):
        return ""
    try:
        return body_text(url, get_text(url))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return ""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _doc_id(issuer: str, url: str, ts: float) -> str:
    tail = re.sub(r"[^A-Za-z0-9]+", "-", url.rsplit("/", 1)[-1])[:60]
    return f"{issuer}:{int(ts)}:{tail}"


def rss_kind(title: str) -> str:
    if _SPEECH_TITLE.search(title):
        return SPEECH
    if _POLICY_TITLE.search(title):
        return MONETARY_POLICY
    return PRESS_RELEASE


_ITEM = re.compile(r"<item[\s>](.*?)</item\s*>", re.S | re.I)


def _tag(item: str, name: str) -> str:
    match = re.search(rf"<{name}[^>]*>(.*?)</{name}\s*>", item, re.S | re.I)
    return html.unescape(strip_tags(match.group(1))) if match else ""


def parse_rss(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _ITEM.findall(raw):
        link = _tag(item, "link")
        title = _tag(item, "title")
        ts = parse_pubdate(_tag(item, "pubDate") or _tag(item, "dc:date"))
        rows.append({"title": title, "link": link, "ts": ts})
    return rows


def fed_archive(
    store: Store,
    since: float,
    *,
    until: float | None = None,
    fetcher: Callable[[str], str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Rows from the three Fed JSON archives. Returns ``(rows, skipped)``.

    The archives are re-published whenever something is added, so they are
    cached per day; the pages behind them are immutable and cached forever.
    """
    until = until if until is not None else time.time()
    fetcher = fetcher or get_text
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    skipped = 0
    for archive, url in FED_ARCHIVES.items():
        raw = store.cached(f"fed:{archive}:{day}", lambda url=url: fetcher(url))
        payload = json.loads(raw.lstrip("﻿"))
        for entry in payload:
            if not isinstance(entry, dict) or "l" not in entry:
                continue  # the archives end with an {"updateDate": ...} row
            raw_date = str(entry.get("d") or "")
            ts = parse_fed_date(raw_date)
            if ts is None:
                # Date-only rows are only worth reporting when they fall in the
                # window being studied; the 2006-2010 ones always do not.
                day_only = parse_fed_date(f"{raw_date} 12:00:00 PM")
                if day_only is None or since <= day_only <= until:
                    skipped += 1
                continue
            if not (since <= ts <= until):
                continue
            if archive == "press":
                kind = _slug(str(entry.get("pt") or "")) or PRESS_RELEASE
            else:
                kind = SPEECH if archive == "speeches" else TESTIMONY
            rows.append(
                {
                    "issuer": "fed",
                    "kind": kind,
                    "ts": ts,
                    "title": html.unescape(str(entry.get("t") or "")),
                    "url": FED_ROOT + str(entry["l"]),
                    "speaker": html.unescape(str(entry.get("s") or "")),
                }
            )
    return rows, skipped


def rss(
    store: Store,
    issuer: str,
    url: str | None = None,
    *,
    fetcher: Callable[[str], str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Rows from one issuer's RSS feed. Returns ``(rows, skipped)``.

    RSS is a shallow window -- fifteen items for the ECB, fifty for the BoE --
    so these issuers can only ever contribute the last week or two. The feed is
    cached per hour, because unlike the Fed archives it genuinely changes.
    """
    url = url or RSS_FEEDS[issuer]
    fetch = fetcher or get_text
    hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    raw = store.cached(f"rss:{issuer}:{hour}", lambda: fetch(url))
    rows: list[dict[str, Any]] = []
    skipped = 0
    for row in parse_rss(raw):
        if row["ts"] is None or not row["link"]:
            skipped += 1
            continue
        rows.append(
            {
                "issuer": issuer,
                "kind": rss_kind(row["title"]),
                "ts": row["ts"],
                "title": row["title"],
                "url": row["link"],
                "speaker": "",
            }
        )
    return rows, skipped


def collect(
    store: Store,
    *,
    since: float,
    until: float | None = None,
    issuers: Sequence[str] = ISSUERS,
    kinds: Iterable[str] | None = POLICY_KINDS,
    limit: int = 0,
    workers: int = 8,
    fed_fetcher: Callable[[str], str] | None = None,
    rss_fetcher: Callable[[str], str] | None = None,
    body_fetcher: Callable[[str], str] | None = None,
) -> Collection:
    """Every policy-relevant document from ``issuers`` between the bounds.

    ``kinds=None`` keeps everything, which is how the enforcement-action arm of
    a sanity check would be run; the default keeps only the three kinds that can
    plausibly move a currency.
    """
    until = until if until is not None else time.time()
    body_fetcher = body_fetcher or fetch_body
    wanted = set(kinds) if kinds is not None else None
    rows: list[dict[str, Any]] = []
    skipped = 0
    for issuer in issuers:
        if issuer == "fed":
            found, missed = fed_archive(
                store, since, until=until, fetcher=fed_fetcher
            )
        else:
            found, missed = rss(store, issuer, fetcher=rss_fetcher)
            found = [r for r in found if since <= r["ts"] <= until]
        skipped += missed
        rows.extend(found)

    rows = [r for r in rows if wanted is None or r["kind"] in wanted]
    rows.sort(key=lambda r: r["ts"])
    if limit:
        rows = rows[-limit:]

    def body(url: str) -> str:
        return store.cached(f"body:{url}", lambda: body_fetcher(url))

    urls = list(dict.fromkeys(r["url"] for r in rows))
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        bodies = dict(zip(urls, pool.map(body, urls)))

    out = Collection(skipped_no_time=skipped)
    for row in rows:
        document = Document(
            id=_doc_id(row["issuer"], row["url"], row["ts"]),
            issuer=row["issuer"],
            kind=row["kind"],
            ts=row["ts"],
            title=row["title"],
            body=bodies.get(row["url"], ""),
            url=row["url"],
            speaker=row["speaker"],
            currency=ISSUER_CURRENCY.get(row["issuer"], ""),
        )
        out.documents.append(document)
        out.per_issuer[document.issuer] = out.per_issuer.get(document.issuer, 0) + 1
    return out


# ----------------------------------------------------------------- calendar


def fetch_calendar() -> list[dict[str, Any]]:
    return get_json(CALENDAR_URL)


def week_key(ts: float) -> str:
    year, week, _ = datetime.fromtimestamp(ts, timezone.utc).isocalendar()
    return f"calendar:{year}-W{week:02d}"


def _calendar_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    try:
        when = datetime.fromisoformat(str(raw["date"]))
    except (KeyError, TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return {
        "title": str(raw.get("title") or ""),
        "country": str(raw.get("country") or ""),
        "impact": str(raw.get("impact") or ""),
        "forecast": str(raw.get("forecast") or ""),
        "previous": str(raw.get("previous") or ""),
        "ts": when.timestamp(),
    }


def calendar_snapshot(
    store: Store,
    *,
    now: float | None = None,
    fetcher: Callable[[], list[dict[str, Any]]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Store this week's calendar under a key for this ISO week.

    Only *this* week is available -- ``lastweek`` and ``nextweek`` both 404 --
    so the numeric-surprise baseline can only ever be live for weeks in which
    somebody ran this. Nothing here reconstructs a calendar for a past week: a
    forecast invented after the fact is not a forecast.
    """
    fetch = fetcher or fetch_calendar
    rows = [r for r in (_calendar_row(r) for r in fetch()) if r is not None]
    key = week_key(now if now is not None else time.time())
    store.put(key, rows)
    found, index = store.get(CALENDAR_INDEX)
    keys = list(index) if found and isinstance(index, list) else []
    if key not in keys:
        keys.append(key)
        store.put(CALENDAR_INDEX, keys)
    return key, rows


def calendar_rows(store: Store) -> list[dict[str, Any]]:
    """Every snapshot ever taken into this cache, oldest first."""
    found, index = store.get(CALENDAR_INDEX)
    if not found or not isinstance(index, list):
        return []
    seen: dict[tuple[str, str, float], dict[str, Any]] = {}
    for key in index:
        got, rows = store.get(key)
        if not got or not isinstance(rows, list):
            continue
        for row in rows:
            seen[(row["title"], row["country"], row["ts"])] = row
    return sorted(seen.values(), key=lambda r: r["ts"])


_RATE_ROW = re.compile(r"\brate\b", re.I)
_NOT_RATE_ROW = re.compile(r"votes?|statement|projections|press conference|minutes|summary|speaks", re.I)


def match_calendar(
    document: Document, rows: list[dict[str, Any]], *, tolerance_s: float = 900.0
) -> dict[str, Any] | None:
    """The calendar row this document *is*, if the snapshot covers its week.

    A decision publishes several rows at the same minute ("Official Bank Rate",
    "MPC Official Bank Rate Votes", "Monetary Policy Summary"); the one whose
    forecast is a rate wins, so "3-0-6" never gets read as three percent.
    """
    near = [
        row for row in rows
        if row["country"] == document.currency and abs(row["ts"] - document.ts) <= tolerance_s
    ]
    if not near:
        return None

    def rank(row: dict[str, Any]) -> tuple[int, int, int, float]:
        title = row.get("title") or ""
        forecast = row.get("forecast") or ""
        is_rate = bool(_RATE_ROW.search(title)) and not _NOT_RATE_ROW.search(title)
        return (
            0 if is_rate and "%" in forecast else 1,
            0 if "%" in forecast else 1,
            0 if forecast else 1,
            abs(row["ts"] - document.ts),
        )

    return min(near, key=rank)


__all__ = [
    "Collection", "Document", "ISSUERS", "ISSUER_CURRENCY", "MONETARY_POLICY",
    "POLICY_KINDS", "PRESS_RELEASE", "SPEECH", "TESTIMONY", "calendar_rows",
    "calendar_snapshot", "collect", "eu_offset", "fed_archive", "fed_body_text",
    "fetch_body", "generic_body_text", "local_string", "match_calendar",
    "parse_fed_date", "parse_pubdate", "parse_rss", "rss", "rss_kind",
    "us_eastern_offset", "week_key",
]
