"""The retail FX wire: every headline a scalper actually sees, and who may read it.

The central-bank study graded one issuer's scheduled text because the Fed
archive was the only free source with minute timestamps and depth. That is not
the stream a scalper reads. **investinglive.com** (formerly ForexLive) is: data
prints, every central bank's speakers, intervention talk, tariffs and politics,
geopolitics, order-flow notes -- 81 to 560 articles a week since 2008.

What the source is (probed, not assumed):

* ``investinglive.com/articles-sitemap-index.xml`` is an index of ~946 weekly
  sitemaps, ``/sitemaps/news/articles/{YYYY}-W{WW}.xml``, one per ISO week from
  2008-W33. Each weekly file is ``<url><loc>…</loc><lastmod>…</lastmod></url>``.
* An article page carries ``<script type="application/ld+json">`` with
  ``@type: NewsArticle`` and the keys ``headline``, ``alternativeHeadline``,
  ``datePublished`` (``2025-04-06T23:51:08.4325180Z`` -- UTC, to the second, with
  seven fractional digits ``datetime.fromisoformat`` will not take),
  ``dateModified``, ``articleSection``, ``keywords``, ``genre``, ``articleBody``
  and ``text``.
* Only the extracted fields are cached, never the HTML. Twenty-one thousand
  article pages is about a gigabyte of markup to keep a few hundred characters
  of each.

**The lateness caveat, which is the whole scope of the result.** This wire runs
seconds to minutes behind Reuters and Bloomberg. Grading from its post time
therefore measures what a reader of *this* wire could have done, not what the
event was worth. The ``pre`` column -- how far the pair moved in the fifteen
minutes before the post -- is that lateness, measured rather than argued about.

**Who is allowed to read it is checked before anything is fetched.** See
``permission`` below: ``robots.txt`` is parsed, and the groups that bind an
AI-driven crawler are the ones this module obeys, whatever ``User-Agent`` header
it would send. As of 2026-09-21 that answer is *no* for this repository, and
``collect`` raises rather than scraping. The adapter is complete and the code
path is exercised offline; what it will not do is send the requests.
"""

from __future__ import annotations

import concurrent.futures
import html
import json
import re
import statistics
import sys
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Sequence

from ..listing.store import Store
from .http import get_text

HOST = "https://investinglive.com"
ROBOTS_URL = f"{HOST}/robots.txt"
SITEMAP_INDEX = f"{HOST}/articles-sitemap-index.xml"
WEEK_SITEMAP = HOST + "/sitemaps/news/articles/{year:04d}-W{week:02d}.xml"

# The first week the archive has, and therefore the floor on ``--since``.
FIRST_WEEK = (2008, 33)

# How much of an article body is kept on disk, and how much of it the model is
# shown by default. The cache cap is generous because re-fetching is the
# expensive half; the state cap is tight because tokens are the other half.
CACHE_BODY_CHARS = 8_000
STATE_BODY_CHARS = 3_000

TRIES = 4
BACKOFF_S = 1.5
# How long a fetched ``robots.txt`` is trusted for. A day, because a site that
# tightened its rules overnight must not be crawled on yesterday's answer.
ROBOTS_TTL_S = 86_400.0
PROGRESS_EVERY = 500

# ``User-Agent`` tokens whose ``robots.txt`` group binds this scraper. The
# wildcard group binds everyone; the three after it are the names a site uses to
# say "no AI agents", and this *is* an AI agent driving a bulk fetch whose
# output is read by a model. Sending a browser's ``User-Agent`` would evade the
# rule rather than satisfy it, so the rule is applied to the work being done and
# not to the header being sent.
AGENT_TOKENS = ("*", "ClaudeBot", "anthropic-ai", "Claude-Web")


class WireError(RuntimeError):
    """A transient fetch failure -- retried, then reported, never cached."""


class WireForbidden(RuntimeError):
    """``robots.txt`` says this client may not have the page. Not retried."""


# ------------------------------------------------------------------- robots


@dataclass(frozen=True)
class Rule:
    allow: bool
    pattern: str

    @property
    def length(self) -> int:
        return len(self.pattern)


@dataclass
class Permission:
    """Whether one path may be fetched, and which line in ``robots.txt`` said so.

    ``blocked_by`` is the agent token whose group matched, so a refusal can say
    *which* rule it is obeying -- a site that disallows everybody and a site that
    disallows AI agents by name are different findings and the caller prints the
    difference.
    """

    allowed: bool
    agent: str = "*"
    rule: str = ""
    blocked_by: str = ""

    def __bool__(self) -> bool:
        return self.allowed


def parse_robots(raw: str) -> dict[str, list[Rule]]:
    """``robots.txt`` -> ``{lowercased user-agent: [Rule, …]}``.

    Consecutive ``User-agent`` lines share the group that follows them, which is
    how every real file writes "these four bots, all disallowed". Comments,
    blank lines and directives that are not Allow/Disallow (``Sitemap``,
    ``Crawl-delay``) are ignored.
    """
    groups: dict[str, list[Rule]] = {}
    current: list[str] = []
    starting = True
    for line in raw.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field_name, _, value = line.partition(":")
        field_name = field_name.strip().lower()
        value = value.strip()
        if field_name == "user-agent":
            if not starting:
                current = []
                starting = True
            current.append(value.lower())
            groups.setdefault(value.lower(), [])
        elif field_name in ("allow", "disallow") and current:
            starting = False
            if field_name == "disallow" and not value:
                continue  # "Disallow:" with nothing after it allows everything
            for agent in current:
                groups[agent].append(Rule(field_name == "allow", value))
    return groups


def _matches(pattern: str, path: str) -> bool:
    """The ``robots.txt`` glob: ``*`` is any run, ``$`` anchors the end."""
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    expr = "".join(".*" if part == "*" else re.escape(part)
                   for part in re.split(r"(\*)", body))
    return re.match(expr + ("$" if anchored else ""), path) is not None


def _verdict(rules: Sequence[Rule], path: str) -> Rule | None:
    """The longest matching rule wins, and Allow wins a tie -- the usual order."""
    hits = [r for r in rules if _matches(r.pattern, path)]
    if not hits:
        return None
    return max(hits, key=lambda r: (r.length, r.allow))


def permission(raw: str, path: str, *, agents: Sequence[str] = AGENT_TOKENS) -> Permission:
    """May this client fetch ``path``? Every group in ``agents`` has to say yes.

    A site that disallows ``ClaudeBot`` and allows ``*`` has said something
    specific about this kind of client, and the specific thing is the one that
    binds: the check is an AND over the groups rather than the single
    best-matching group a browser would use.
    """
    groups = parse_robots(raw)
    for agent in agents:
        rules = groups.get(agent.lower())
        if rules is None:
            continue
        rule = _verdict(rules, path)
        if rule is not None and not rule.allow:
            return Permission(False, agent, f"Disallow: {rule.pattern or '/'}", agent)
    return Permission(True)


def fetch_robots(store: Store, *, fetcher: Callable[[str], str] | None = None,
                 ttl_s: float = ROBOTS_TTL_S) -> str:
    """``robots.txt``, cached for a day. Unreachable is not permission, so it raises.

    The cache has a short life on purpose: a site that has *tightened* its rules
    since yesterday must not be crawled on yesterday's answer. A cache entry
    stored as a bare string rather than a stamped record is honoured without
    expiry -- that is a hand-placed override, used by the tests and by a cache
    somebody else filled, and it is never written by this function.
    """
    found, cached = store.get("wire:robots")
    if found and isinstance(cached, str):
        return cached
    if (found and isinstance(cached, dict)
            and time.time() - float(cached.get("ts") or 0.0) < ttl_s):
        return str(cached.get("raw") or "")
    raw = (fetcher or get_text)(ROBOTS_URL)
    store.put("wire:robots", {"ts": time.time(), "raw": raw})
    return raw


def check(store: Store, path: str = "/news/",
          *, fetcher: Callable[[str], str] | None = None) -> Permission:
    return permission(fetch_robots(store, fetcher=fetcher), path)


def require(store: Store, path: str, *, fetcher: Callable[[str], str] | None = None) -> None:
    verdict = check(store, path, fetcher=fetcher)
    if not verdict.allowed:
        raise WireForbidden(
            f"{HOST}{path}: robots.txt disallows it for '{verdict.blocked_by}' "
            f"({verdict.rule}). Not fetched."
        )


# ------------------------------------------------------------------- fetching


def fetch(url: str, *, timeout: float = 30.0, tries: int = TRIES,
          fetcher: Callable[[str], str] | None = None) -> str:
    """One page, retried with jittered backoff. 404 is a settled empty answer.

    Everything else that fails every try is a ``WireError``: the caller counts
    it as a failure and never caches it, because a cached bad minute of network
    is a permanent hole in the archive.
    """
    get = fetcher or get_text
    last = ""
    for attempt in range(max(1, tries)):
        try:
            return get(url)
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return ""
            last = f"HTTP {exc.code}"
        except Exception as exc:  # noqa: BLE001 -- transport, not logic
            last = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < tries:
            time.sleep(BACKOFF_S * 2**attempt * (0.5 + (attempt % 3) / 3.0))
    raise WireError(f"{url}: {last}")


# ------------------------------------------------------------------- sitemaps

_LOC = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.I | re.S)
_URL_BLOCK = re.compile(r"<url\b.*?</url>", re.I | re.S)
_LASTMOD = re.compile(r"<lastmod>\s*(.*?)\s*</lastmod>", re.I | re.S)
_WEEK_NAME = re.compile(r"/(\d{4})-W(\d{2})\.xml", re.I)


@dataclass(frozen=True)
class Link:
    url: str
    lastmod: float = 0.0


def parse_index(raw: str) -> list[tuple[int, int, str]]:
    """The sitemap index -> ``[(year, week, url), …]``, oldest first.

    The index is read for the weeks it actually has rather than for its URLs:
    a week the archive never wrote is a 404 nobody should ask for, and the
    ``{YYYY}-W{WW}`` name is the only thing that says which weeks those are.
    """
    out: list[tuple[int, int, str]] = []
    for url in _LOC.findall(raw or ""):
        url = html.unescape(url.strip())
        match = _WEEK_NAME.search(url)
        if match:
            out.append((int(match.group(1)), int(match.group(2)), url))
    return sorted(set(out))


def parse_week(raw: str) -> list[Link]:
    """One weekly sitemap -> its article links, with ``lastmod`` kept.

    ``lastmod`` is not the publication time -- an article edited a year later
    carries the edit -- so it is kept as metadata and never used as a timestamp.
    The timestamp comes from the article's own JSON-LD or nowhere.
    """
    out: list[Link] = []
    for block in _URL_BLOCK.findall(raw or ""):
        loc = _LOC.search(block)
        if not loc:
            continue
        stamp = _LASTMOD.search(block)
        out.append(Link(html.unescape(loc.group(1).strip()),
                        parse_iso_ts(stamp.group(1)) or 0.0 if stamp else 0.0))
    return out


def weeks_covering(since: float, until: float) -> list[tuple[int, int]]:
    """Every ISO week that could hold an article in ``[since, until]``.

    The week the archive files an article under is its publication week, so the
    weeks that overlap the window are exactly the files to fetch; a day-by-day
    walk gets the year boundaries right without a calendar rule (2021-01-01 is
    2020-W53), which is where hand-rolled ISO week arithmetic goes wrong.
    """
    if until < since:
        since, until = until, since
    first = datetime.fromtimestamp(since, timezone.utc).date()
    last = datetime.fromtimestamp(until, timezone.utc).date()
    out: list[tuple[int, int]] = []
    day = first
    while day <= last:
        year, week, _ = day.isocalendar()
        if (year, week) not in out:
            out.append((year, week))
        day += timedelta(days=1)
    # The last week of the window is only partly inside it, and so is the first;
    # both are fetched whole and the articles are filtered by their own stamp.
    return out


def weekly_sitemaps(store: Store, *, fetcher: Callable[[str], str] | None = None,
                    offline: bool = False) -> list[tuple[int, int, str]]:
    """The index, cached: every ``(year, week, url)`` the archive publishes."""
    found, cached = store.get("wire:index")
    if found and isinstance(cached, list):
        return [(int(y), int(w), str(u)) for y, w, u in cached]
    if offline:
        return []
    require(store, "/articles-sitemap-index.xml", fetcher=fetcher)
    weeks = parse_index(fetch(SITEMAP_INDEX, fetcher=fetcher))
    store.put("wire:index", [list(w) for w in weeks])
    return weeks


def week_url(year: int, week: int, index: Sequence[tuple[int, int, str]] = ()) -> str:
    for y, w, url in index:
        if (y, w) == (year, week):
            return url
    return WEEK_SITEMAP.format(year=year, week=week)


def article_urls(
    store: Store,
    since: float,
    until: float,
    *,
    fetcher: Callable[[str], str] | None = None,
    workers: int = 4,
    offline: bool = False,
) -> list[Link]:
    """Every article link in the weeks overlapping ``[since, until]``, deduplicated.

    The window is applied to the *weeks*, not to the links: a sitemap carries no
    publication time, only ``lastmod``, and filtering on ``lastmod`` would drop
    an article published in the window and edited outside it. The filtering by
    time happens after the article's own stamp is read.
    """
    if not offline:
        require(store, "/sitemaps/news/articles/", fetcher=fetcher)
    index = weekly_sitemaps(store, fetcher=fetcher, offline=offline)
    have = {(y, w) for y, w, _ in index} if index else set()
    wanted = [wk for wk in weeks_covering(since, until)
              if wk >= FIRST_WEEK and (not have or wk in have)]

    def one(week: tuple[int, int]) -> list[Link]:
        year, number = week
        key = f"wire:week:{year:04d}-W{number:02d}"
        found, cached = store.get(key)
        if found and isinstance(cached, list):
            return [Link(str(u), float(m)) for u, m in cached]
        if offline:
            return []
        links = parse_week(fetch(week_url(year, number, index), fetcher=fetcher))
        store.put(key, [[link.url, link.lastmod] for link in links])
        return links

    out: list[Link] = []
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        for group in pool.map(one, wanted):
            out.extend(group)
    seen: dict[str, Link] = {}
    for link in out:
        seen.setdefault(link.url, link)
    return list(seen.values())


# ------------------------------------------------------------------- articles

_LD_BLOCK = re.compile(
    r"<script[^>]+type\s*=\s*[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
    re.I | re.S,
)
_META = re.compile(
    r"<meta[^>]+(?:property|name)\s*=\s*[\"']([^\"']+)[\"'][^>]*content\s*=\s*"
    r"[\"'](.*?)[\"'][^>]*>", re.I | re.S)
_META_REVERSED = re.compile(
    r"<meta[^>]+content\s*=\s*[\"'](.*?)[\"'][^>]*(?:property|name)\s*=\s*"
    r"[\"']([^\"']+)[\"'][^>]*>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_ISO = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.(\d+))?"
    r"\s*(Z|[+-]\d{2}:?\d{2})?", re.I)


def parse_iso_ts(text: str) -> float | None:
    """ISO-8601 -> epoch seconds, or None. Written out because the wire's is not.

    ``datetime.fromisoformat`` refuses ``2025-04-06T23:51:08.4325180Z``: seven
    fractional digits, where it takes three or six. The fraction is dropped
    rather than rounded -- the study's finest horizon is a minute -- and a stamp
    with no offset is read as UTC, which is what this source publishes.
    """
    match = _ISO.search(text or "")
    if not match:
        return None
    year, month, day, hour, minute, second, _frac, zone = match.groups()
    try:
        when = datetime(int(year), int(month), int(day), int(hour), int(minute),
                        int(second or 0), tzinfo=timezone.utc)
    except ValueError:
        return None
    if zone and zone.upper() != "Z":
        sign = 1 if zone[0] == "+" else -1
        digits = zone[1:].replace(":", "")
        when -= timedelta(minutes=sign * (int(digits[:2]) * 60 + int(digits[2:4])))
    return when.timestamp()


def _text(value: Any) -> str:
    if isinstance(value, str):
        return html.unescape(_TAG.sub(" ", value))
    if isinstance(value, list):
        return " ".join(_text(v) for v in value)
    if isinstance(value, dict):
        return _text(value.get("name") or value.get("@value") or "")
    return ""


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip()


def _keywords(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[,;|]", value)]
    elif isinstance(value, list):
        parts = [_clean(v) for v in value]
    else:
        parts = []
    return tuple(p for p in parts if p)


def _ld_objects(raw: str) -> list[dict[str, Any]]:
    """Every JSON-LD object on the page, ``@graph`` and top-level lists flattened."""
    out: list[dict[str, Any]] = []
    for block in _LD_BLOCK.findall(raw or ""):
        try:
            payload = json.loads(block.strip())
        except ValueError:
            continue
        stack = payload if isinstance(payload, list) else [payload]
        while stack:
            item = stack.pop(0)
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                stack.extend(graph)
            out.append(item)
    return out


def _meta_tags(raw: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for key, value in _META.findall(raw or ""):
        tags.setdefault(key.lower(), html.unescape(value))
    for value, key in _META_REVERSED.findall(raw or ""):
        tags.setdefault(key.lower(), html.unescape(value))
    return tags


@dataclass(frozen=True)
class Article:
    """One wire post: when it was published, what it said, and what it was filed as."""

    url: str
    published_ts: float
    headline: str
    section: str = ""
    keywords: tuple[str, ...] = ()
    body: str = ""
    source: str = "json-ld"  # or "meta": the fallback, counted separately

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.published_ts, timezone.utc)

    @property
    def id(self) -> str:
        tail = re.sub(r"[^A-Za-z0-9]+", "-", self.url.rstrip("/").rsplit("/", 1)[-1])[:60]
        return f"wire:{int(self.published_ts)}:{tail}"


def parse_article(url: str, raw: str) -> Article | None:
    """The JSON-LD ``NewsArticle``, or the meta tags when there is none.

    Returns None only when neither a headline nor a timestamp can be found, so
    "the page was not an article" and "the page failed to fetch" stay different
    answers upstream.
    """
    for item in _ld_objects(raw):
        kinds = item.get("@type")
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if not any(isinstance(k, str) and k.endswith("Article") for k in kinds):
            continue
        published = parse_iso_ts(str(item.get("datePublished") or ""))
        headline = _clean(item.get("headline") or item.get("alternativeHeadline"))
        if published is None and headline == "":
            continue
        body = _clean(item.get("articleBody") or item.get("text"))
        return Article(
            url=url, published_ts=published or 0.0, headline=headline,
            section=_clean(item.get("articleSection") or item.get("genre")),
            keywords=_keywords(item.get("keywords")),
            body=body[:CACHE_BODY_CHARS], source="json-ld",
        )
    tags = _meta_tags(raw)
    published = parse_iso_ts(tags.get("article:published_time", "")
                             or tags.get("datepublished", ""))
    headline = _clean(tags.get("og:title") or tags.get("twitter:title") or tags.get("title"))
    if published is None and not headline:
        return None
    return Article(
        url=url, published_ts=published or 0.0, headline=headline,
        section=_clean(tags.get("article:section") or ""),
        keywords=_keywords(tags.get("news_keywords") or tags.get("keywords") or ""),
        body=_clean(tags.get("og:description") or "")[:CACHE_BODY_CHARS],
        source="meta",
    )


def _to_dict(article: Article) -> dict[str, Any]:
    return {
        "url": article.url, "ts": article.published_ts, "headline": article.headline,
        "section": article.section, "keywords": list(article.keywords),
        "body": article.body[:CACHE_BODY_CHARS], "source": article.source,
    }


def _from_dict(data: dict[str, Any]) -> Article:
    return Article(
        url=str(data.get("url", "")), published_ts=float(data.get("ts") or 0.0),
        headline=str(data.get("headline", "")), section=str(data.get("section", "")),
        keywords=tuple(data.get("keywords") or ()),
        body=str(data.get("body") or "")[:CACHE_BODY_CHARS],
        source=str(data.get("source", "json-ld")),
    )


def fetch_article(
    store: Store, url: str, *, fetcher: Callable[[str], str] | None = None,
    offline: bool = False,
) -> Article | None:
    """One article, cached **as the extracted fields and never as the HTML**.

    Twenty-one thousand pages of markup is about a gigabyte on disk to keep a
    few hundred characters of each, so the page is parsed once and thrown away.
    A page that parses to nothing is cached as nothing, because it will parse to
    nothing again; a page that failed to *fetch* raises and is not cached. In
    ``offline`` mode nothing is fetched at all and a cache miss raises
    ``KeyError``, which is how a run against somebody else's cache stays a run
    that sends no requests.
    """
    key = f"wire:article:{url}"
    found, cached = store.get(key)
    if found:
        return _from_dict(cached) if isinstance(cached, dict) else None
    if offline:
        raise KeyError(url)
    require(store, "/news/", fetcher=fetcher)
    raw = fetch(url, fetcher=fetcher)
    article = parse_article(url, raw) if raw else None
    store.put(key, _to_dict(article) if article is not None else None)
    return article


# ------------------------------------------------------------------- collect


@dataclass
class Coverage:
    """What ``collect`` found, and what it refused to use.

    The refusals are part of the result and not a log line: a study that quietly
    drops a tenth of its feed is not measuring the feed it names.
    """

    urls: int = 0
    articles: int = 0
    per_section: dict[str, int] = field(default_factory=dict)
    median_body: int = 0
    fallbacks: int = 0  # parsed from meta tags because the JSON-LD was missing
    failures: int = 0  # never reached after every retry; not cached
    unparsed: int = 0  # fetched, but neither JSON-LD nor meta tags made an article
    no_timestamp: int = 0  # parsed, but with no ``datePublished`` to grade from
    out_of_window: int = 0
    uncached: int = 0  # offline mode only: links this cache has never held

    def summary(self) -> str:
        top = sorted(self.per_section.items(), key=lambda kv: -kv[1])[:8]
        return (
            f"{self.articles} articles from {self.urls} links; median body "
            f"{self.median_body:,} chars; {self.fallbacks} from meta tags, "
            f"{self.unparsed} unparseable, {self.no_timestamp} with no timestamp, "
            f"{self.out_of_window} outside the window, {self.failures} failures"
            + (f", {self.uncached} not in the cache" if self.uncached else "") + "\n"
            "  sections: " + (", ".join(f"{k or 'none'} {v}" for k, v in top) or "none")
        )


def collect(
    store: Store,
    since: float,
    until: float,
    *,
    workers: int = 4,
    fetcher: Callable[[str], str] | None = None,
    limit: int = 0,
    progress_every: int = PROGRESS_EVERY,
    offline: bool = False,
) -> tuple[list[Article], Coverage]:
    """Every article published in ``[since, until]``, oldest first, with the count
    of everything dropped on the way.

    At most four connections, one kept open per worker. Articles with no
    timestamp are dropped rather than placed at the start of the epoch, and the
    number dropped is reported: a study cannot grade what it cannot time.

    ``offline`` reads the cache and never the network -- the mode for a run
    against a cache filled by somebody the wire permits. A link the cache does
    not hold is counted in ``uncached`` and skipped.
    """
    links = article_urls(store, since, until, fetcher=fetcher, workers=min(workers, 4),
                         offline=offline)
    links.sort(key=lambda link: (link.lastmod, link.url))
    if limit > 0:
        links = links[-limit:]
    coverage = Coverage(urls=len(links))
    started = time.perf_counter()
    counter = {"done": 0}
    guard = threading.Lock()

    def one(link: Link) -> Article | None:
        try:
            article = fetch_article(store, link.url, fetcher=fetcher, offline=offline)
        except KeyError:
            with guard:
                coverage.uncached += 1
            return None
        except WireError as exc:
            with guard:
                coverage.failures += 1
            print(f"  {exc}", file=sys.stderr)
            article = None
        except WireForbidden:
            raise
        with guard:
            counter["done"] += 1
            done = counter["done"]
            if progress_every and (done % progress_every == 0 or done == len(links)):
                print(f"  wire {done}/{len(links)} articles, "
                      f"{coverage.failures} failures, "
                      f"{time.perf_counter() - started:.0f}s",
                      file=sys.stderr, flush=True)
        return article

    with concurrent.futures.ThreadPoolExecutor(max(1, min(workers, 4))) as pool:
        fetched = list(pool.map(one, links))

    out: list[Article] = []
    lengths: list[int] = []
    for article in fetched:
        if article is None:
            coverage.unparsed += 1
            continue
        if article.source == "meta":
            coverage.fallbacks += 1
        if not article.published_ts:
            coverage.no_timestamp += 1
            continue
        if not (since <= article.published_ts <= until):
            coverage.out_of_window += 1
            continue
        out.append(article)
        lengths.append(len(article.body))
        coverage.per_section[article.section] = coverage.per_section.get(article.section, 0) + 1
    # ``unparsed`` counted a failed fetch once already; a failure is not an
    # unparseable page and the two are reported separately.
    coverage.unparsed = max(0, coverage.unparsed - coverage.failures - coverage.uncached)
    out.sort(key=lambda a: (a.published_ts, a.url))
    coverage.articles = len(out)
    coverage.median_body = int(statistics.median(lengths)) if lengths else 0
    return out, coverage


def sessions(ts: float) -> str:
    """Which trading session a UTC moment sits in. A rule, stated once.

    Asia 00-07, London 07-13, New York 13-21, late 21-24, all UTC. The
    boundaries are conventional and coarse -- they do not move with daylight
    saving -- and they are here so the breakdown by session is one definition
    rather than one per table.
    """
    hour = datetime.fromtimestamp(ts, timezone.utc).hour
    if hour < 7:
        return "asia"
    if hour < 13:
        return "london"
    if hour < 21:
        return "new_york"
    return "late"


SESSIONS = ("asia", "london", "new_york", "late")


def local_times(ts: float) -> dict[str, str]:
    """The post's moment in the three places that trade it, as fixed offsets.

    New York -5, London +0 and Tokyo +9 are **standard time** and do not follow
    daylight saving: ``zoneinfo`` needs a tz database a slim container may not
    have, and an hour of error in a label the model reads as "the Tokyo
    afternoon" is not worth a dependency. The UTC stamp is the exact one.
    """
    when = datetime.fromtimestamp(ts, timezone.utc)
    return {
        "new_york": (when - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"),
        "london": when.strftime("%Y-%m-%d %H:%M"),
        "tokyo": (when + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M"),
    }


def day_epoch(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def iso_week_of(when: date) -> tuple[int, int]:
    year, week, _ = when.isocalendar()
    return year, week


def per_week(articles: Iterable[Article]) -> dict[str, int]:
    out: dict[str, int] = {}
    for article in articles:
        year, week = iso_week_of(article.when.date())
        key = f"{year:04d}-W{week:02d}"
        out[key] = out.get(key, 0) + 1
    return out


# ------------------------------------------------------------------- lexicon
#
# The wire's own vocabulary, in one place, because two things read it: the
# ``keyword-bot`` arm -- the incumbent, and the thing a reader has to beat --
# and the offline mock that stands in for the model. Keeping one copy is what
# makes "the model did better than the words" a comparable claim; keeping two
# would let the mock quietly be a different bot from the baseline.
#
# These are rules. They cannot read. "The RBA will not hike" counts as a hike.

CURRENCY_WORDS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("JPY", re.compile(r"(?i)\byen\b|\bjpy\b|bank of japan|\bboj\b|\bueda\b|"
                       r"japan(ese)?\b|\bmof\b|usd/?jpy")),
    ("EUR", re.compile(r"(?i)\beuro\b|\beur\b|\becb\b|lagarde|euro ?zone|euro ?area|"
                       r"germany|german|france|french|italy|eur/?usd")),
    ("GBP", re.compile(r"(?i)sterling|\bgbp\b|\bpound\b|bank of england|\bboe\b|"
                       r"bailey|\buk\b|britain|british|gbp/?usd")),
    ("AUD", re.compile(r"(?i)\baud\b|aussie|australia|\brba\b|bullock|aud/?usd")),
    ("NZD", re.compile(r"(?i)\bnzd\b|\bkiwi\b|new zealand|\brbnz\b|nzd/?usd")),
    ("CAD", re.compile(r"(?i)\bcad\b|loonie|canada|canadian|bank of canada|\bboc\b|"
                       r"macklem|usd/?cad")),
    ("CHF", re.compile(r"(?i)\bchf\b|\bfranc\b|swiss|switzerland|\bsnb\b|usd/?chf")),
    ("CNY", re.compile(r"(?i)\bcny\b|\bcnh\b|\byuan\b|renminbi|\bpboc\b|china|chinese")),
    ("USD", re.compile(r"(?i)\bdollar\b|\busd\b|\bfed\b|\bfomc\b|powell|federal reserve|"
                       r"united states|\bus\b|\bu\.s\.|greenback|treasury")),
)

HAWKISH_WIRE = re.compile(
    r"(?i)\bhawkish\b|\bhike[sd]?\b|\bhiking\b|\braise[sd]? rates\b|\btighten\w*|"
    r"\bbeat[s]?\b|\bstronger\b|\bstrong\b|\bhigher than (expected|forecast)|"
    r"\btops? (forecast|estimate)|\bupside surprise|\bjumps?\b|\bsurges?\b|"
    r"\baccelerat\w*|\brebound\w*|\bhotter\b")
DOVISH_WIRE = re.compile(
    r"(?i)\bdovish\b|\bcut[s]?\b|\bcutting\b|\blower rates\b|\bease?[sd]?\b|"
    r"\beasing\b|\bmiss(es|ed)?\b|\bweaker\b|\bweak\b|\blower than (expected|forecast)|"
    r"\bbelow (forecast|estimate)|\bdownside surprise|\bfalls?\b|\bslumps?\b|"
    r"\bslow(s|ed|ing|down)?\b|\bcooler\b|\bcontract\w*")

# Posts that are not about a currency at all, which on this wire is a real
# fraction of the feed: it also covers crypto, equities and chart levels.
NOT_FX = re.compile(
    r"(?i)\bbitcoin\b|\bbtc\b|\bethereum\b|\beth\b|\bcrypto\w*|\baltcoin|"
    r"\bnasdaq\b|\bs&p\b|\bdow\b|\bearnings\b|\bshares?\b|\bstocks?\b|"
    r"\bnikkei\b|\bdax\b|\bftse\b")

CATEGORY_WORDS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("fx_official_or_intervention",
     re.compile(r"(?i)interven\w*|rate check|excessive|one-?sided|disorderly|"
                r"\bfix\b|verbal intervention")),
    ("central_bank_decision",
     re.compile(r"(?i)rate decision|policy decision|minutes|statement|holds? rates|"
                r"leaves? rates|raises? rates|cuts? rates|\bsep\b|projections")),
    ("central_bank_speaker",
     re.compile(r"(?i)\bspeech\b|speaks|says|comments|testimony|press conference|"
                r"powell|lagarde|bailey|ueda|macklem|bullock")),
    ("data_release",
     re.compile(r"(?i)\bcpi\b|\bppi\b|\bpmi\b|payrolls|\bgdp\b|unemployment|"
                r"retail sales|trade balance|inflation rate|\bifo\b|\bzew\b|"
                r"jobless claims|industrial production|consumer confidence")),
    ("politics_or_trade",
     re.compile(r"(?i)tariff\w*|trade deal|trade war|election|budget|parliament|"
                r"congress|president|prime minister|legislation|shutdown")),
    ("geopolitics",
     re.compile(r"(?i)\bwar\b|sanction\w*|missile|strike[sd]?\b|invasion|ceasefire|"
                r"\bopec\b|attack\w*|conflict")),
    ("orderflow_or_positioning",
     re.compile(r"(?i)option expir\w*|expiries|\bbarrier\w*|\bcftc\b|positioning|"
                r"month-?end|\bflows?\b|\bbids?\b at|\boffers?\b at")),
    ("market_commentary_or_technical",
     re.compile(r"(?i)support|resistance|technical|outlook|preview|wrap|forecast|"
                r"analysts?|strategist|\bchart\b|\blevels?\b")),
)

ALREADY_MOVED_WIRE = re.compile(
    r"(?i)(jumped|surged|slumped|tumbled|rallied|sold off|spiked|plunged|"
    r"already priced|priced in|reacted|knee-?jerk)")
SCHEDULED_WIRE = re.compile(
    r"(?i)\bcpi\b|\bppi\b|\bpmi\b|payrolls|\bgdp\b|rate decision|"
    r"policy decision|minutes|\bspeech\b|press conference|jobless claims|"
    r"retail sales|due at|scheduled")
NUMBER_WIRE = re.compile(r"(?i)\d[\d.,]*\s*%|vs\.? ?(exp|expected|forecast)|"
                         r"\bprior\b|\bconsensus\b|\bestimate\b")


def currency_of(text: str) -> str:
    """The first currency the text names, in a fixed precedence. A rule, not a reading.

    The order matters and is deliberate: almost every post on this wire mentions
    the dollar somewhere, so USD is last and only wins when nothing else is
    named. That is a rule with a known bias and it is the baseline's bias, not
    the reader's.
    """
    for code, pattern in CURRENCY_WORDS:
        if pattern.search(text or ""):
            return code
    return "none"


def direction_of(text: str) -> str:
    """Hawkish/beat words against dovish/miss ones -> stronger / weaker / none."""
    up = len(HAWKISH_WIRE.findall(text or ""))
    down = len(DOVISH_WIRE.findall(text or ""))
    if up == down:
        return "none"
    return "stronger" if up > down else "weaker"


def category_of(text: str) -> str:
    """The first category whose words appear, in a fixed precedence."""
    for name, pattern in CATEGORY_WORDS:
        if pattern.search(text or ""):
            return name
    return "other"


def about_fx_of(text: str) -> bool:
    """Whether a rule would call this an FX post: a currency named and no crypto."""
    if NOT_FX.search(text or ""):
        return False
    return currency_of(text) != "none"


__all__ = [
    "AGENT_TOKENS", "ALREADY_MOVED_WIRE", "Article", "CACHE_BODY_CHARS", "ROBOTS_TTL_S",
    "CATEGORY_WORDS", "CURRENCY_WORDS", "Coverage", "DOVISH_WIRE", "FIRST_WEEK",
    "HAWKISH_WIRE", "HOST", "NOT_FX", "NUMBER_WIRE", "SCHEDULED_WIRE",
    "about_fx_of", "category_of", "currency_of", "direction_of",
    "Link", "Permission", "ROBOTS_URL", "Rule", "SESSIONS", "SITEMAP_INDEX",
    "STATE_BODY_CHARS", "WEEK_SITEMAP", "WireError", "WireForbidden", "article_urls",
    "check", "collect", "day_epoch", "fetch", "fetch_article", "fetch_robots",
    "iso_week_of", "local_times", "parse_article", "parse_index", "parse_iso_ts",
    "parse_robots", "parse_week", "per_week", "permission", "require", "sessions",
    "week_url", "weekly_sitemaps", "weeks_covering",
]
