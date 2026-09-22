"""A corpus somebody else collected, read into the same ``Article`` the wire uses.

The wire adapter next door is finished and idle, because investinglive.com's
``robots.txt`` disallows AI agents by name and this repository obeys it. Nothing
about the rest of that machinery -- the ``w1`` tree, the minute-candle judge,
the arms, the breakdowns -- depends on where the posts came from, only on their
being ``Article``s with a timestamp. So a corpus the user collected under their
own licence and handed over as a file drops straight in.

The first one is the **Truth Social archive for 2025**, scraped from
trumpstruth.org. It arrives in two captures of the same thing and a CSV of the
first, and this module normalises all three:

* the *posts* shape -- ``status_id``, ``text``, ``date_published`` (ISO, UTC, to
  the second), ``original_url``, ``archive_url``;
* the *statuses* shape -- ``status_id``, ``article_body``, ``headline``
  (``Donald J. Trump: "…"``), ``date_published``, ``url``, ``original_url``;
* the CSV, which is the posts shape with a BOM on the first header and every
  value a string.

What was measured about the files rather than assumed (counted 2026-09-22, and
three of these contradict what the import was specified from):

* The posts file is **6,683 lines but only 6,120 distinct ``status_id``** -- 563
  ids appear exactly twice. The duplicate rows are identical in ``text``, so the
  dedupe is a straight de-duplication and loses nothing.
* The statuses file is 6,120 lines, one per id, and covers **exactly** the same
  ids. It is a deduplicated second capture, not extra coverage.
* **It cannot fill any missing text.** Both files have the same 1,305 empty
  posts -- the media-only ones and the reposts -- and they are the same 1,305
  ids. The fill below is implemented because it is the right rule for two
  captures of one archive; on this corpus it fires zero times, and ``filled``
  says so rather than the code pretending otherwise.
* So the corpus is **4,815 posts with text**, 2025-01-01 to 2026-01-10, of which
  one is in 2026.

A post with no text is dropped rather than read as an empty document: a reader
handed nothing will answer something, and that answer would be about the
reader, not about the post.

**The weekend is the caveat this corpus carries and the wire does not.** These
posts land whenever they land, and spot FX is shut from about Friday 21:00 to
Sunday 21:00 UTC. A post in that gap cannot be graded at all -- not scored zero,
not carried to Monday -- so the study counts them separately from the ones it
traded, because "the tape was shut" and "the reader stayed out" are different
findings and only one of them is about the reader.
"""

from __future__ import annotations

import csv
import io
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..listing.store import Store
from .wire import CACHE_BODY_CHARS, Article, parse_iso_ts

SECTION = "truth_social"
# The cache namespace. Deliberately not ``wire:article:{url}``: these are not
# wire articles, nobody fetched them, and mixing the two would make a cache
# filled from a file indistinguishable from one filled from the network.
KEY = "wire:posts:{status_id}"
INDEX_KEY = "wire:posts:index"

HEADLINE_CHARS = 120

# ``Donald J. Trump: "…"`` on a text post, and a handful of other names where
# the archive captured a repost. The quotes are the archive's, not the poster's.
_HEADLINE_PREFIX = re.compile(r'^\s*[^":]{1,60}:\s*"?')
_HEADLINE_SUFFIX = re.compile(r'"\s*$')
# ``Donald J. Trump image post from January 1, 2025`` -- the archive's stand-in
# for a post that has no text. It is a description of an absence, not a
# headline, so it is never used as one.
_MEDIA_HEADLINE = re.compile(r"^\s*.{1,60}\s+(image|video|media)\s+post\s+from\s", re.I)


@dataclass
class PostCoverage:
    """What the import read and what it refused to use.

    Every count here is part of the result rather than a log line. A corpus
    where a fifth of the rows carry no text is a different corpus from one where
    none do, and a study that does not say which it had is not reporting its
    sample.
    """

    files: int = 0
    lines: int = 0
    distinct: int = 0
    duplicates: int = 0  # rows merged into an id already seen, within or across files
    filled: int = 0  # texts recovered from a second capture of the same id
    dropped_empty: int = 0
    loaded: int = 0
    median_chars: int = 0
    per_month: dict[str, int] = field(default_factory=dict)
    no_timestamp: int = 0
    unreadable: int = 0

    def summary(self) -> str:
        months = ", ".join(f"{k} {v}" for k, v in sorted(self.per_month.items()))
        return (
            f"{self.loaded} posts with text from {self.lines:,} lines in "
            f"{self.files} file(s): {self.distinct} distinct ids "
            f"({self.duplicates} rows merged into an id already seen), "
            f"{self.filled} texts "
            f"filled from a second capture, {self.dropped_empty} dropped for "
            f"having no text, {self.no_timestamp} for having no timestamp, "
            f"{self.unreadable} unreadable rows; median {self.median_chars:,} "
            f"characters\n  per month: {months}"
        )


# ----------------------------------------------------------------- reading


def read_rows(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Every row of one file, and the number of lines that would not parse.

    JSON Lines or CSV, chosen by suffix and then by what the first character
    turns out to be, because a ``.txt`` full of JSON objects is still JSON
    Lines. The CSV is opened ``utf-8-sig``: this one carries a BOM on the first
    header, and without that the first column is named ``\\ufeffpost_id`` and
    every lookup of ``post_id`` silently misses.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    head = raw.lstrip()[:1]
    if path.suffix.lower() == ".csv" or (head and head not in "[{"):
        return list(csv.DictReader(io.StringIO(raw))), 0
    if head == "[":  # a whole-file JSON array, which some exports are
        try:
            payload = json.loads(raw)
        except ValueError:
            return [], 1
        return [r for r in payload if isinstance(r, dict)], 0
    rows: list[dict[str, Any]] = []
    bad = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(item, dict):
            rows.append(item)
        else:
            bad += 1
    return rows, bad


def strip_headline(headline: str) -> str:
    """``Donald J. Trump: "Happy New Year"`` -> ``Happy New Year``.

    The author prefix is stripped generically rather than by name: 7 of the
    6,120 headlines in this archive are somebody else's, because the capture
    kept a repost under the account that posted it.
    """
    text = (headline or "").strip()
    if not text or _MEDIA_HEADLINE.match(text):
        return ""
    text = _HEADLINE_PREFIX.sub("", text, count=1)
    return _HEADLINE_SUFFIX.sub("", text).strip()


def headline_of(text: str, given: str = "", limit: int = HEADLINE_CHARS) -> str:
    """The first ~120 characters of the post, cut at a word.

    The post's own text wins over the archive's ``headline`` field, because that
    field is itself a truncation of the same text with the author's name glued
    on the front -- all 4,808 quoted headlines in this archive are shorter than
    the body they quote. The given headline is the fallback for a row that
    arrived with no body.
    """
    body = re.sub(r"\s+", " ", (text or "").strip())
    if not body:
        return strip_headline(given)
    if len(body) <= limit:
        return body
    cut = body[:limit]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip() + "..."


@dataclass
class _Row:
    status_id: str
    text: str = ""
    url: str = ""
    published: str = ""
    headline: str = ""


def normalise(row: dict[str, Any]) -> _Row | None:
    """One row of any of the three shapes -> the four fields that matter.

    ``None`` when there is no ``status_id`` to key on, which is the only field
    the merge cannot do without.
    """
    status_id = str(row.get("status_id") or row.get("post_id") or "").strip()
    if not status_id:
        return None
    text = str(row.get("text") or row.get("article_body") or "").strip()
    url = str(row.get("original_url") or row.get("archive_url")
              or row.get("url") or "").strip()
    return _Row(
        status_id=status_id, text=text, url=url,
        published=str(row.get("date_published") or "").strip(),
        headline=str(row.get("headline") or "").strip(),
    )


def load_posts(
    paths: Sequence[str | Path],
    *,
    body_chars: int = CACHE_BODY_CHARS,
    headline_chars: int = HEADLINE_CHARS,
) -> tuple[list[Article], PostCoverage]:
    """Every post with text, oldest first, with the count of everything dropped.

    Files are merged on ``status_id`` across all of them, and the **non-empty
    text wins** wherever two captures disagree -- which is the rule that would
    let a second capture rescue a media-only row, and which on this archive
    rescues none, because both captures are missing the same 1,305.
    """
    coverage = PostCoverage(files=len(paths))
    merged: dict[str, _Row] = {}
    for path in paths:
        rows, bad = read_rows(path)
        coverage.lines += len(rows)
        coverage.unreadable += bad
        for raw in rows:
            row = normalise(raw)
            if row is None:
                coverage.unreadable += 1
                continue
            seen = merged.get(row.status_id)
            if seen is None:
                merged[row.status_id] = row
                continue
            coverage.duplicates += 1
            if not seen.text and row.text:
                coverage.filled += 1
                seen.text = row.text
            seen.url = seen.url or row.url
            seen.headline = seen.headline or row.headline
            seen.published = seen.published or row.published
    coverage.distinct = len(merged)

    out: list[Article] = []
    lengths: list[int] = []
    for row in merged.values():
        if not row.text:
            coverage.dropped_empty += 1
            continue
        when = parse_iso_ts(row.published)
        if when is None:
            coverage.no_timestamp += 1
            continue
        out.append(Article(
            url=row.url or f"https://www.trumpstruth.org/statuses/{row.status_id}",
            published_ts=when,
            headline=headline_of(row.text, row.headline, headline_chars),
            section=SECTION, keywords=(), body=row.text[:body_chars],
            source="posts-file", source_id=row.status_id,
        ))
        lengths.append(len(row.text))
    out.sort(key=lambda a: (a.published_ts, a.source_id))
    coverage.loaded = len(out)
    coverage.median_chars = int(statistics.median(lengths)) if lengths else 0
    for article in out:
        key = f"{article.when:%Y-%m}"
        coverage.per_month[key] = coverage.per_month.get(key, 0) + 1
    return out, coverage


# ------------------------------------------------------------------- cache


def _to_dict(article: Article) -> dict[str, Any]:
    return {
        "url": article.url, "ts": article.published_ts, "headline": article.headline,
        "section": article.section, "keywords": list(article.keywords),
        "body": article.body[:CACHE_BODY_CHARS], "source": article.source,
        "source_id": article.source_id,
    }


def _from_dict(data: dict[str, Any]) -> Article:
    return Article(
        url=str(data.get("url", "")), published_ts=float(data.get("ts") or 0.0),
        headline=str(data.get("headline", "")), section=str(data.get("section", "")),
        keywords=tuple(data.get("keywords") or ()),
        body=str(data.get("body") or "")[:CACHE_BODY_CHARS],
        source=str(data.get("source", "posts-file")),
        source_id=str(data.get("source_id", "")),
    )


def store_posts(store: Store, articles: Sequence[Article]) -> int:
    """Write the corpus into the cache under its own namespace, and index it.

    The index exists so a later run needs neither the files nor ``--posts``: the
    corpus is in the cache like any other, and every downstream step reads it
    the same way it reads a cached wire corpus. Nothing here ever touches the
    uploaded files again, and nothing here is a fetch.
    """
    ids: list[str] = []
    for article in articles:
        status_id = article.source_id or str(int(article.published_ts))
        store.put(KEY.format(status_id=status_id), _to_dict(article))
        ids.append(status_id)
    found, known = store.get(INDEX_KEY)
    if found and isinstance(known, list):
        ids = list(dict.fromkeys([str(i) for i in known] + ids))
    store.put(INDEX_KEY, ids)
    return len(ids)


def cached_posts(store: Store, since: float, until: float) -> list[Article]:
    """The imported corpus, filtered to a window, oldest first. No file, no fetch."""
    found, ids = store.get(INDEX_KEY)
    if not (found and isinstance(ids, list)):
        return []
    out: list[Article] = []
    for status_id in ids:
        ok, data = store.get(KEY.format(status_id=status_id))
        if not (ok and isinstance(data, dict)):
            continue
        article = _from_dict(data)
        if article.published_ts and since <= article.published_ts <= until:
            out.append(article)
    out.sort(key=lambda a: (a.published_ts, a.source_id))
    return out


def import_posts(
    store: Store,
    paths: Sequence[str | Path],
    since: float,
    until: float,
    *,
    body_chars: int = CACHE_BODY_CHARS,
    progress: bool = True,
) -> tuple[list[Article], PostCoverage]:
    """Read the files, cache the corpus, hand back the window. Never fetches."""
    articles, coverage = load_posts(paths, body_chars=body_chars)
    store_posts(store, articles)
    if progress:
        print(f"  imported {len(articles)} posts into the cache", file=sys.stderr,
              flush=True)
    window = [a for a in articles if since <= a.published_ts <= until]
    return window, coverage


# ------------------------------------------------------------- the weekend


def is_weekend(ts: float) -> bool:
    """Saturday or Sunday in UTC. A rule, and a coarse one.

    Spot FX is shut from about Friday 21:00 to Sunday 21:00 UTC, which is not
    the same span, so this label is *when the post was written* and not *whether
    the tape was open*. The second question is answered by the tape itself --
    a signal the judge could not measure -- and the two are reported side by
    side rather than conflated.
    """
    return datetime.fromtimestamp(ts, timezone.utc).weekday() >= 5


def weekday_label(ts: float) -> str:
    return "weekend" if is_weekend(ts) else "weekday"


def per_weekday(articles: Iterable[Article]) -> dict[str, int]:
    out: dict[str, int] = {"weekday": 0, "weekend": 0}
    for article in articles:
        out[weekday_label(article.published_ts)] += 1
    return out


__all__ = [
    "HEADLINE_CHARS", "INDEX_KEY", "KEY", "PostCoverage", "SECTION",
    "cached_posts", "headline_of", "import_posts", "is_weekend", "load_posts",
    "normalise", "per_weekday", "read_rows", "store_posts", "strip_headline",
    "weekday_label",
]
