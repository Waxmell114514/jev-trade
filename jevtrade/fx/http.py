"""One persistent TLS connection per worker thread, and a text fetch on top of it.

This is the fetch ``ticks.py`` grew and every other FX source now shares. It is
here rather than there because the reason for it is not the tick feed, it is the
environment: **measured from this sandbox, the first request on a fresh TLS
connection costs nine to sixteen seconds** (handshake, through a proxy), and
every request after it on the same connection about 0.2 s. A fresh connection
per file therefore caps a fetch at roughly five files a minute, and feeds answer
5xx when several connections are opened at once. One connection per worker,
kept open and dropped only on a transport error, does hundreds a minute.

What is measured and what is assumed:

* **Measured** -- the handshake cost and the keep-alive cost above, and that the
  Dukascopy feed returns 503 under a burst of new connections.
* **Assumed** -- that a server honours ``Connection: keep-alive``. When it does
  not, the next request on the dead connection raises, the connection is
  dropped, and the caller's retry opens a new one. That is a slow path, not a
  wrong one.

Nothing here caches, retries or backs off. Those are the caller's policy,
because "404 means this hour is empty" is true of the tick feed and false of an
article page.
"""

from __future__ import annotations

import base64
import http.client
import os
import ssl
import threading
import urllib.error
import urllib.parse

UA = {"User-Agent": "Mozilla/5.0 (compatible; jev-trade research)"}

_local = threading.local()


def _ssl_context() -> ssl.SSLContext:
    cafile = (os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
              or os.environ.get("CURL_CA_BUNDLE") or None)
    return ssl.create_default_context(cafile=cafile)


def _connection(host: str, timeout: float) -> http.client.HTTPSConnection:
    """One persistent TLS connection per thread, tunnelled through the proxy if set.

    Measured from this environment: the first request on a connection to the
    feed costs 9-16 s (the handshake), every request after it about 0.2 s. A
    fresh connection per file therefore caps the fetch at ~5 files a minute and
    trips the feed's 503s under concurrency; one connection per worker, kept
    open, does ~300 a minute.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "host", None) == host:
        conn.timeout = timeout
        return conn
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        parsed = urllib.parse.urlparse(proxy)
        conn = http.client.HTTPSConnection(
            parsed.hostname or "", parsed.port, timeout=timeout, context=_ssl_context(),
        )
        headers = {}
        if parsed.username:
            token = base64.b64encode(
                f"{parsed.username}:{parsed.password or ''}".encode()
            ).decode("ascii")
            headers["Proxy-Authorization"] = f"Basic {token}"
        conn.set_tunnel(host, 443, headers=headers)
    else:
        conn = http.client.HTTPSConnection(host, 443, timeout=timeout, context=_ssl_context())
    _local.conn, _local.host = conn, host
    return conn


def _drop_connection() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _local.conn, _local.host = None, None


def _get(url: str, timeout: float = 30.0) -> bytes:
    """GET over the thread's persistent connection; ``HTTPError`` on a non-200."""
    parts = urllib.parse.urlparse(url)
    conn = _connection(parts.hostname or "", timeout)
    try:
        conn.request("GET", parts.path or "/", headers={**UA, "Connection": "keep-alive"})
        response = conn.getresponse()
        body = response.read()
    except (http.client.HTTPException, OSError):
        _drop_connection()
        raise
    if response.status != 200:
        if not response.getheader("Connection", "").lower() == "keep-alive":
            _drop_connection()
        raise urllib.error.HTTPError(url, response.status, response.reason, response.headers, None)
    return body


def get_text(url: str, timeout: float = 30.0, *, query: str = "") -> str:
    """An HTML or XML page as text, over the same kept-open connection.

    ``query`` is appended to the path because ``_get`` sends only the path and a
    sitemap index is sometimes served with one. Decoding is ``utf-8`` with
    replacement: a wire page with one bad byte in a smart quote is still a
    readable article, and refusing it would lose the article over the quote.
    """
    return _get(url + query, timeout=timeout).decode("utf-8", "replace")


__all__ = ["UA", "_connection", "_drop_connection", "_get", "get_text"]
