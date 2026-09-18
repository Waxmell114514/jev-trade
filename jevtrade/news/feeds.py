"""Collect crypto headlines with timestamps from public RSS feeds.

No key required, and deliberately many sources: the distribution question is
about what actually arrives on a desk's feed, which is a mix of wire copy,
market recaps, opinion and press releases, not a curated event list.
"""

from __future__ import annotations

import concurrent.futures
import email.utils
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

FEEDS = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "bitcoinmagazine": "https://bitcoinmagazine.com/feed",
    "cryptoslate": "https://cryptoslate.com/feed/",
    "newsbtc": "https://www.newsbtc.com/feed/",
    "bitcoinist": "https://bitcoinist.com/feed/",
    "ambcrypto": "https://ambcrypto.com/feed/",
    "utoday": "https://u.today/rss",
    "theblock": "https://www.theblock.co/rss.xml",
    "beincrypto": "https://beincrypto.com/feed/",
    "cryptobriefing": "https://cryptobriefing.com/feed/",
}

UA = {"User-Agent": "Mozilla/5.0 (compatible; jev-trade research)"}
ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class Headline:
    ts: float  # epoch seconds, UTC
    title: str
    source: str

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.ts, timezone.utc)


def _parse_time(text: str) -> float | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return email.utils.parsedate_to_datetime(text).timestamp()
    except (TypeError, ValueError):
        pass
    try:  # ISO-8601, as Atom uses
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _clean(title: str) -> str:
    title = re.sub(r"<[^>]+>", "", title or "")
    return re.sub(r"\s+", " ", title).strip()


def fetch_feed(name: str, url: str, timeout: float = 25.0) -> list[Headline]:
    request = urllib.request.Request(url, headers=UA)
    raw = urllib.request.urlopen(request, timeout=timeout).read()
    root = ET.fromstring(raw)

    out: list[Headline] = []
    # RSS <item> and Atom <entry> both carry one story each; parsing per entry
    # keeps a title paired with its own timestamp (a flat scan would pick up
    # the channel title too).
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        title = stamp = None
        for child in node:
            ctag = child.tag.split("}")[-1]
            if ctag == "title" and title is None:
                title = _clean("".join(child.itertext()))
            elif ctag in ("pubDate", "published", "updated", "date") and stamp is None:
                stamp = _parse_time("".join(child.itertext()))
        if title and stamp:
            out.append(Headline(ts=stamp, title=title, source=name))
    return out


def _normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def collect(feeds: dict[str, str] | None = None) -> tuple[list[Headline], dict[str, str]]:
    """Fetch every feed in parallel and de-duplicate across sources."""
    feeds = feeds or FEEDS
    errors: dict[str, str] = {}
    gathered: list[Headline] = []

    def one(item):
        name, url = item
        try:
            return name, fetch_feed(name, url), ""
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return name, [], f"{type(exc).__name__}: {exc}"[:80]

    with concurrent.futures.ThreadPoolExecutor(len(feeds)) as pool:
        for name, rows, err in pool.map(one, feeds.items()):
            if err:
                errors[name] = err
            gathered.extend(rows)

    seen: set[str] = set()
    unique: list[Headline] = []
    for headline in sorted(gathered, key=lambda h: h.ts):
        key = _normalise(headline.title)
        # Wire copy gets rewritten across sites; the opening words survive.
        short = " ".join(key.split()[:7])
        if not key or key in seen or short in seen:
            continue
        seen.add(key)
        seen.add(short)
        unique.append(headline)
    return unique, errors
