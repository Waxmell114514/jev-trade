"""Binance's announcement feed, the way a sniping bot reads it.

The public CMS endpoint is what listing bots poll. Each article carries a
millisecond ``releaseDate``, which is the moment the page went live and so the
moment anyone could have acted -- a much better clock than an RSS ``pubDate``.

Three catalogues are read, and all of them, not a hand-picked subset: new
listings, delistings, and the general news catalogue that is mostly noise
(promotions, maintenance, fee changes). A reader that only ever sees listings
has not been tested on the hard part, which is telling a listing from the
notice next to it that merely mentions the same token.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from .store import Store

LIST_URL = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
    "?type=1&pageNo={page}&pageSize={size}&catalogId={catalog}"
)
DETAIL_URL = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query"
    "?articleCode={code}"
)
CATALOGS = {48: "new listings", 161: "delistings", 49: "latest news"}
UA = {"User-Agent": "Mozilla/5.0 (compatible; jev-trade research)"}


@dataclass(frozen=True)
class Announcement:
    id: int
    code: str
    catalog: int
    ts: float  # epoch seconds, UTC: the moment the page went live
    title: str
    body: str

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.ts, timezone.utc)


def get_json(url: str, timeout: float = 30.0, attempts: int = 4) -> Any:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
                last = exc
                time.sleep(1.5 * (2**attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (2**attempt))
                continue
            raise
    raise last or RuntimeError("unreachable")  # pragma: no cover


# ------------------------------------------------------------------ body text

_BLOCK_TAGS = {"p", "li", "br", "h1", "h2", "h3", "h4", "tr", "div", "td", "th"}


def _walk(node: Any, out: list[str]) -> None:
    if isinstance(node, list):
        for child in node:
            _walk(child, out)
        return
    if not isinstance(node, dict):
        return
    if node.get("node") == "text":
        out.append(str(node.get("text", "")))
    if node.get("tag") in _BLOCK_TAGS:
        out.append("\n")
    _walk(node.get("child") or [], out)


def body_text(body: str) -> str:
    """Flatten the CMS body -- a JSON node tree, or HTML on older articles."""
    body = body or ""
    try:
        tree = json.loads(body)
    except ValueError:
        text = re.sub(r"<br\s*/?>|</p>|</li>|</tr>", "\n", body)
        text = re.sub(r"<[^>]+>", " ", text)
    else:
        parts: list[str] = []
        _walk(tree, parts)
        text = "".join(parts)
    text = text.replace("&nbsp;", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ---------------------------------------------------------------- fetching

def fetch_page(catalog: int, page: int, size: int = 20) -> list[dict[str, Any]]:
    payload = get_json(LIST_URL.format(page=page, size=size, catalog=catalog))
    catalogs = (payload.get("data") or {}).get("catalogs") or []
    rows: list[dict[str, Any]] = []
    for entry in catalogs:
        for article in entry.get("articles") or []:
            rows.append(
                {
                    "id": int(article["id"]),
                    "code": str(article["code"]),
                    "title": str(article["title"]),
                    "releaseDate": int(article["releaseDate"]),
                }
            )
    return rows


def fetch_body(code: str) -> str:
    payload = get_json(DETAIL_URL.format(code=code))
    data = payload.get("data") or {}
    return body_text(str(data.get("body") or ""))


def collect(
    store: Store,
    *,
    since: float,
    until: float | None = None,
    catalogs: tuple[int, ...] = (48, 161, 49),
    page_size: int = 20,
    max_pages: int = 400,
    workers: int = 8,
    page_fetcher: Callable[[int, int, int], list[dict[str, Any]]] = fetch_page,
    body_fetcher: Callable[[str], str] = fetch_body,
) -> list[Announcement]:
    """Every announcement in the catalogues between ``since`` and ``until``."""
    until = until or time.time()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    index: dict[str, dict[str, Any]] = {}
    for catalog in catalogs:
        for page in range(1, max_pages + 1):
            # Pages shift as new articles land, so the page cache is per day;
            # bodies are immutable and cached forever.
            key = f"page:{catalog}:{page}:{page_size}:{day}"
            rows = store.cached(key, lambda: page_fetcher(catalog, page, page_size))
            if not rows:
                break
            for row in rows:
                ts = row["releaseDate"] / 1000.0
                if since <= ts <= until:
                    index[row["code"]] = {**row, "catalog": catalog, "ts": ts}
            if min(r["releaseDate"] for r in rows) / 1000.0 < since:
                break

    def body(code: str) -> str:
        return store.cached(f"body:{code}", lambda: body_fetcher(code))

    codes = list(index)
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        bodies = dict(zip(codes, pool.map(body, codes)))

    out = [
        Announcement(
            id=row["id"], code=code, catalog=row["catalog"], ts=row["ts"],
            title=row["title"], body=bodies[code],
        )
        for code, row in index.items()
    ]
    return sorted(out, key=lambda a: a.ts)
