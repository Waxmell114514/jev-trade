"""The retail FX wire: the robots gate, the adapter, the minute judge, the tree.

Everything here is offline. The two fetchers -- ``wire.get_text`` for HTML and
``candles.fetch_bi5`` for the price files -- are replaced, so the whole path
from a sitemap to a breakdown table runs from fixtures in a temporary
directory and the tests never touch either host.
"""

from __future__ import annotations

import lzma
import struct
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from jevtrade.fx import candles as C
from jevtrade.fx import reader as R
from jevtrade.fx import wire as W
from jevtrade.fx import wirestudy as WS
from jevtrade.fx.mock import MockWireClient
from jevtrade.fx.study import Signal
from jevtrade.listing.store import Store

FIXTURES = Path(__file__).parent / "fixtures" / "fx"

REAL_ROBOTS = (FIXTURES / "investinglive-robots.txt").read_text()
SITEMAP_INDEX = (FIXTURES / "wire-sitemap-index.xml").read_text()
WEEK = (FIXTURES / "wire-week.xml").read_text()
ARTICLE = (FIXTURES / "wire-article.html").read_text()
ARTICLE_META = (FIXTURES / "wire-article-meta.html").read_text()

OPEN_ROBOTS = "User-agent: *\nDisallow: /search-results\n"


def ts(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(
        tzinfo=timezone.utc).timestamp()


# ------------------------------------------------------------------- robots


def test_the_live_robots_file_disallows_ai_agents_by_name():
    """The verdict this whole mode is gated on, pinned to the file itself.

    The wildcard group lets any crawler have the articles and the sitemaps; the
    three groups after it name AI agents and disallow the entire site. This
    scraper is one, so ``permission`` has to answer no -- and it has to answer
    yes for a plain crawler, or the test would pass for the wrong reason.
    """
    assert W.permission(REAL_ROBOTS, "/news/", agents=("*",)).allowed
    assert W.permission(REAL_ROBOTS, "/articles-sitemap-index.xml", agents=("*",)).allowed

    verdict = W.permission(REAL_ROBOTS, "/news/japan-nikkei-20250406/")
    assert not verdict.allowed
    assert verdict.blocked_by == "ClaudeBot"
    assert verdict.rule == "Disallow: /"
    assert not W.permission(REAL_ROBOTS, "/sitemaps/news/articles/2025-W14.xml").allowed


def test_the_wildcard_rules_are_read_for_paths_and_suffixes():
    assert not W.permission(REAL_ROBOTS, "/search-results", agents=("*",)).allowed
    assert not W.permission(REAL_ROBOTS, "/tag/search-results?q=eur", agents=("*",)).allowed
    # "Disallow: /*.pdf$" is anchored: a path that merely contains .pdf is fine.
    assert not W.permission(REAL_ROBOTS, "/x/y.pdf", agents=("*",)).allowed
    assert W.permission(REAL_ROBOTS, "/x/y.pdf.html", agents=("*",)).allowed


def test_consecutive_user_agent_lines_share_one_group():
    groups = W.parse_robots("User-agent: a\nUser-agent: b\nDisallow: /x\n\nUser-agent: c\n")
    assert [r.pattern for r in groups["a"]] == ["/x"]
    assert [r.pattern for r in groups["b"]] == ["/x"]
    assert groups["c"] == []


def test_an_empty_disallow_allows_everything():
    assert W.permission("User-agent: *\nDisallow:\n", "/anything").allowed


def test_an_allow_beats_a_shorter_disallow():
    raw = "User-agent: *\nDisallow: /news\nAllow: /news/fx\n"
    assert not W.permission(raw, "/news/crypto", agents=("*",)).allowed
    assert W.permission(raw, "/news/fx/eur", agents=("*",)).allowed


def test_a_stale_robots_cache_is_refetched_and_a_seeded_one_is_honoured(tmp_path):
    """A day-old answer is not permission; a hand-placed one is an override."""
    store = Store(tmp_path)
    asked: list[str] = []

    def fetcher(url: str) -> str:
        asked.append(url)
        return OPEN_ROBOTS

    assert W.fetch_robots(store, fetcher=fetcher) == OPEN_ROBOTS
    assert W.fetch_robots(store, fetcher=fetcher) == OPEN_ROBOTS
    assert len(asked) == 1  # still fresh
    assert W.fetch_robots(store, fetcher=fetcher, ttl_s=-1.0) == OPEN_ROBOTS
    assert len(asked) == 2  # expired, so asked again
    store.put("wire:robots", REAL_ROBOTS)
    assert W.fetch_robots(store, fetcher=fetcher, ttl_s=-1.0) == REAL_ROBOTS
    assert len(asked) == 2  # a bare string never expires


def test_collect_refuses_rather_than_scraping_when_robots_says_no(tmp_path):
    store = Store(tmp_path)
    store.put("wire:robots", REAL_ROBOTS)

    def explode(_url: str) -> str:
        raise AssertionError("a fetch was attempted after robots.txt said no")

    with pytest.raises(W.WireForbidden):
        W.collect(store, ts("2025-04-01 00:00"), ts("2025-04-07 23:59"), fetcher=explode)


# ------------------------------------------------------------------ sitemaps


def test_the_sitemap_index_is_read_for_the_weeks_it_has():
    weeks = W.parse_index(SITEMAP_INDEX)
    assert weeks[0][:2] == (2008, 33)
    assert (2025, 14, "https://investinglive.com/sitemaps/news/articles/2025-W14.xml") in weeks
    assert len(weeks) == 3


def test_a_weekly_sitemap_keeps_the_links_and_their_lastmod():
    links = W.parse_week(WEEK)
    assert len(links) == 3
    assert links[0].url.endswith("japan-nikkei-225-futures-point-lower-20250406/")
    assert links[0].lastmod == ts("2025-04-06 23:51") + 8
    # The old-style "/news/!/…" URL is kept: it is an article, oddly spelled.
    assert links[2].url.endswith("japan-monetary-base-old-style-20170903")
    assert links[2].lastmod == 0.0


def test_the_weeks_of_a_window_cross_the_year_boundary_by_the_iso_calendar():
    # 2020-12-28 is 2020-W53 and 2021-01-01 is still 2020-W53; 2021-W01 starts
    # on the 4th. Hand-rolled week arithmetic gets this wrong.
    weeks = W.weeks_covering(ts("2020-12-28 00:00"), ts("2021-01-05 00:00"))
    assert weeks == [(2020, 53), (2021, 1)]
    assert W.weeks_covering(ts("2025-01-01 00:00"), ts("2025-12-31 23:00"))[0] == (2025, 1)


def test_article_urls_fetches_only_the_weeks_the_window_touches(tmp_path):
    store = Store(tmp_path)
    store.put("wire:robots", OPEN_ROBOTS)
    asked: list[str] = []

    def fetcher(url: str) -> str:
        asked.append(url)
        return SITEMAP_INDEX if url.endswith("articles-sitemap-index.xml") else WEEK

    links = W.article_urls(store, ts("2025-04-01 00:00"), ts("2025-04-06 23:59"),
                           fetcher=fetcher, workers=1)
    weeks = [u for u in asked if "2025-W" in u]
    assert weeks == ["https://investinglive.com/sitemaps/news/articles/2025-W14.xml"]
    assert len(links) == 3
    # Second call: everything is cached, nothing is fetched again.
    asked.clear()
    W.article_urls(store, ts("2025-04-01 00:00"), ts("2025-04-06 23:59"),
                   fetcher=fetcher, workers=1)
    assert asked == []


# ------------------------------------------------------------------ articles


def test_the_json_ld_article_is_read_whole():
    article = W.parse_article("https://investinglive.com/news/x-20250406/", ARTICLE)
    assert article is not None
    assert article.source == "json-ld"
    assert article.headline == "Japan - Nikkei 225 futures point lower at the open"
    assert article.published_ts == ts("2025-04-06 23:51") + 8
    assert article.section == "News"
    assert article.keywords == ("JPY", "Nikkei", "Japan")
    assert "lower open in Tokyo" in article.body


def test_seven_fractional_digits_and_a_trailing_z_are_parsed():
    # ``datetime.fromisoformat`` refuses this exact string, which is the one the
    # wire publishes, so the parser is written out rather than borrowed.
    when = W.parse_iso_ts("2025-04-06T23:51:08.4325180Z")
    assert when == ts("2025-04-06 23:51") + 8
    assert W.parse_iso_ts("2025-04-07T11:15:00+02:00") == ts("2025-04-07 09:15")
    assert W.parse_iso_ts("") is None
    assert W.parse_iso_ts("not a date") is None


def test_the_meta_tags_are_the_fallback_when_there_is_no_json_ld():
    article = W.parse_article("https://investinglive.com/central-banks/y/", ARTICLE_META)
    assert article is not None
    assert article.source == "meta"
    assert article.headline == "ECB's Lagarde: inflation risks are to the upside"
    assert article.published_ts == ts("2025-04-07 09:15")
    assert article.section == "Central banks"
    assert article.keywords == ("EUR", "ECB", "Lagarde")


def test_a_page_that_is_not_an_article_parses_to_nothing():
    assert W.parse_article("https://investinglive.com/x/", "<html><body>hi</body></html>") is None


def test_the_body_is_capped_in_the_cache_and_again_in_the_state():
    long_body = "word " * 5000
    raw = ARTICLE.replace("Nikkei 225 futures are pointing to a lower open in Tokyo. "
                          "The yen is a touch firmer against the dollar after the "
                          "weekend headlines.", long_body)
    article = W.parse_article("https://investinglive.com/news/x/", raw)
    assert article is not None
    assert len(article.body) == W.CACHE_BODY_CHARS
    state = R.build_wire_state(article, round_no=1, body_chars=3000)
    assert len(state["body"]) == 3000


def test_an_article_is_cached_as_its_fields_and_never_as_the_html(tmp_path):
    store = Store(tmp_path)
    store.put("wire:robots", OPEN_ROBOTS)
    url = "https://investinglive.com/news/x-20250406/"
    calls: list[str] = []

    def fetcher(asked: str) -> str:
        calls.append(asked)
        return ARTICLE

    first = W.fetch_article(store, url, fetcher=fetcher)
    second = W.fetch_article(store, url, fetcher=fetcher)
    assert calls == [url]
    assert first == second
    stored = store.get(f"wire:article:{url}")[1]
    assert set(stored) == {"url", "ts", "headline", "section", "keywords", "body",
                           "source", "source_id"}
    assert "<script" not in "".join(
        path.read_text() for path in Path(tmp_path).glob("*.json"))


def test_collect_reports_what_it_dropped_and_sorts_by_publication(tmp_path):
    store = Store(tmp_path)
    store.put("wire:robots", OPEN_ROBOTS)
    undated = ARTICLE.replace('"datePublished":"2025-04-06T23:51:08.4325180Z",', "") \
                     .replace('<meta property="article:published_time" '
                              'content="2025-04-06T23:51:08Z">', "")

    def fetcher(url: str) -> str:
        if url.endswith("articles-sitemap-index.xml"):
            return SITEMAP_INDEX
        if "2025-W" in url:
            return WEEK
        if "ecb-lagarde" in url:
            return ARTICLE_META
        if "japan-monetary-base" in url:
            return undated
        return ARTICLE

    articles, coverage = W.collect(store, ts("2025-04-01 00:00"), ts("2025-04-07 23:59"),
                                   workers=1, fetcher=fetcher, progress_every=0)
    assert coverage.urls == 3
    assert coverage.articles == 2
    assert coverage.fallbacks == 1
    assert coverage.no_timestamp == 1
    assert coverage.failures == 0
    assert [a.published_ts for a in articles] == sorted(a.published_ts for a in articles)
    assert coverage.per_section == {"News": 1, "Central banks": 1}
    assert coverage.median_body > 0


def test_a_fetch_that_never_succeeds_is_counted_and_not_cached(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.put("wire:robots", OPEN_ROBOTS)
    monkeypatch.setattr(W.time, "sleep", lambda _s: None)

    def fetcher(url: str) -> str:
        if url.endswith("articles-sitemap-index.xml"):
            return SITEMAP_INDEX
        if "2025-W" in url:
            return WEEK
        raise TimeoutError("the wire is down")

    articles, coverage = W.collect(store, ts("2025-04-01 00:00"), ts("2025-04-07 23:59"),
                                   workers=1, fetcher=fetcher, progress_every=0)
    assert articles == []
    assert coverage.failures == 3
    assert coverage.unparsed == 0
    assert store.get("wire:article:"
                     "https://investinglive.com/central-banks/ecb-lagarde-speaks-20250407/"
                     )[0] is False


# ------------------------------------------------------------------- candles


def day_bytes(opens: list[int], *, step: int = 0) -> bytes:
    """A day-file body: 24-byte big-endian records, one a minute from 00:00."""
    return lzma.compress(b"".join(
        struct.pack(">IIIIIf", i * 60, value, value + step, value - 5, value + 5, 100.0)
        for i, value in enumerate(opens)))


def test_the_candle_record_decodes_at_both_price_scales():
    raw = day_bytes([107947])
    assert C.decode_candles(raw, C.DEFAULT_SCALE, 1000.0)[0] == C.Bar(
        1000.0, 1.07947, 1.07947, 1.07942, 1.07952, 100.0)
    jpy = C.decode_candles(day_bytes([149757]), C.JPY_SCALE, 0.0)[0]
    assert jpy.open == pytest.approx(149.757)
    assert C.decode_candles(b"", C.DEFAULT_SCALE) == []


def test_the_month_in_a_candle_url_is_zero_based():
    assert C.day_url("EURUSD", date(2025, 4, 2), "BID").endswith(
        "/EURUSD/2025/03/02/BID_candles_min_1.bi5")
    assert C.day_url("usdjpy", date(2025, 1, 15), "ask").endswith(
        "/USDJPY/2025/00/15/ASK_candles_min_1.bi5")


def feed(days: dict[str, tuple[list[int], list[int]]]):
    """A stand-in for ``fetch_bi5``: ``{iso date: (bid opens, ask opens)}``."""

    def fetch(url: str) -> bytes:
        parts = url.rstrip("/").split("/")
        year, month, day = int(parts[-4]), int(parts[-3]) + 1, int(parts[-2])
        key = f"{year:04d}-{month:02d}-{day:02d}"
        if key not in days:
            return b""
        bid, ask = days[key]
        return day_bytes(ask if parts[-1].startswith("ASK") else bid)

    return fetch


def flat_day(bid: int, ask: int, minutes: int = 1440) -> tuple[list[int], list[int]]:
    return [bid] * minutes, [ask] * minutes


def tape_for(tmp_path, days, symbol: str = "EURUSD") -> C.MinuteTape:
    return C.MinuteTape(Store(tmp_path), symbol, fetcher=feed(days))


def test_minute_lookups_work_across_a_day_boundary(tmp_path):
    days = {"2025-04-02": flat_day(100000, 100020),
            "2025-04-03": flat_day(200000, 200020)}
    tape = tape_for(tmp_path, days)
    last = ts("2025-04-02 23:59")
    assert tape.bar_at(last + 30).ts == last
    # The minute after the last one of a day is the first of the next.
    assert tape.next_bar_open(last + 30).ts == ts("2025-04-03 00:00")
    assert tape.next_bar_open(last + 30).bid.open == 2.0
    assert tape.bar_at(ts("2025-04-04 12:00")) is None


def test_a_long_pays_the_ask_open_and_a_short_is_filled_at_the_bid_open(tmp_path):
    # Flat 1.00000/1.00020 in the morning, 1.00100/1.00120 from 12:00 on.
    bid = [100000] * 720 + [100100] * 720
    ask = [100020] * 720 + [100120] * 720
    tape = tape_for(tmp_path, {"2025-04-02": (bid, ask)})
    when = ts("2025-04-02 10:00")

    long = C.forward_returns_minutes(tape, when, +1, latency_s=1.0, horizons_min=(180,))
    short = C.forward_returns_minutes(tape, when, -1, latency_s=1.0, horizons_min=(180,))
    assert long is not None and short is not None
    assert long.entry_px == pytest.approx(1.00020)   # the ask
    assert short.entry_px == pytest.approx(1.00000)  # the bid
    assert long.spread_bps == pytest.approx(2.0, abs=0.01)
    # The mid rose by 10 pips: the long makes it net of the half spread, the
    # short loses it and pays the half spread on top.
    assert long.fwd_bps[180] > 0 > short.fwd_bps[180]
    assert long.fwd_mid_bps[180] > long.fwd_bps[180]
    assert short.fwd_mid_bps[180] > short.fwd_bps[180]


def test_the_latency_rounds_up_to_the_next_bar_boundary(tmp_path):
    opens = [100000 + i for i in range(1440)]
    tape = tape_for(tmp_path, {"2025-04-02": (opens, [v + 20 for v in opens])})
    at = ts("2025-04-02 10:00")
    # A post stamped mid-minute fills on the NEXT minute's open, whatever the
    # latency is, so 1 s and 59 s are the same trade and 300 s is five later.
    for latency in (1.0, 20.0, 39.0):
        out = C.forward_returns_minutes(tape, at + 20, +1, latency_s=latency,
                                        horizons_min=(5,))
        assert out.entry_ts == at + 60
    late = C.forward_returns_minutes(tape, at + 20, +1, latency_s=300.0, horizons_min=(5,))
    assert late.entry_ts == at + 360
    # A post stamped exactly on a boundary fills on that bar at 0 s.
    exact = C.forward_returns_minutes(tape, at, +1, latency_s=0.0, horizons_min=(5,))
    assert exact.entry_ts == at


def test_a_weekend_and_a_horizon_into_a_missing_day_both_come_back_none(tmp_path):
    tape = tape_for(tmp_path, {"2025-04-04": flat_day(100000, 100020)})
    # Saturday: no file at all.
    assert C.forward_returns_minutes(tape, ts("2025-04-05 10:00"), +1,
                                     horizons_min=(15,)) is None
    # Friday 23:30 with a 60-minute horizon reaches into a day the feed has not
    # got, which would be a 51-hour return in disguise.
    assert C.forward_returns_minutes(tape, ts("2025-04-04 23:30"), +1,
                                     horizons_min=(60,)) is None
    assert C.forward_returns_minutes(tape, ts("2025-04-04 12:00"), +1,
                                     horizons_min=(60,)) is not None
    # And the fifteen minutes before the post have to exist too.
    assert C.forward_returns_minutes(tape, ts("2025-04-04 00:05"), +1,
                                     horizons_min=(15,)) is None


def test_a_day_file_is_cached_once_and_not_refetched(tmp_path):
    store = Store(tmp_path)
    calls: list[str] = []

    def counting(url: str) -> bytes:
        calls.append(url)
        return feed({"2025-04-02": flat_day(100000, 100020)})(url)

    tape = C.MinuteTape(store, "EURUSD", fetcher=counting)
    tape.bars(date(2025, 4, 2))
    again = C.MinuteTape(store, "EURUSD", fetcher=counting)
    again.bars(date(2025, 4, 2))
    assert len(calls) == 2  # BID and ASK, once between the two tapes
    have, want = C.coverage(store, ["EURUSD"], ts("2025-04-02 00:00"),
                            ts("2025-04-02 23:00"))["EURUSD"]
    assert (have, want) == (2, 2)


def padded_day(real: int, price: int = 100000) -> tuple[list[int], list[int]]:
    """``real`` minutes with volume, then the feed's flat zero-volume padding."""
    bid = [price] * 1440
    ask = [price + 20] * 1440
    return bid, ask


def zero_volume_body(opens: list[int], live: int) -> bytes:
    return lzma.compress(b"".join(
        struct.pack(">IIIIIf", i * 60, value, value, value, value,
                    100.0 if i < live else 0.0)
        for i, value in enumerate(opens)))


def padded_feed(days: dict[str, tuple[list[int], list[int], int]]):
    def fetch(url: str) -> bytes:
        parts = url.rstrip("/").split("/")
        year, month, day = int(parts[-4]), int(parts[-3]) + 1, int(parts[-2])
        key = f"{year:04d}-{month:02d}-{day:02d}"
        if key not in days:
            return b""
        bid, ask, live = days[key]
        return zero_volume_body(ask if parts[-1].startswith("ASK") else bid, live)

    return fetch


def test_the_feeds_zero_volume_weekend_padding_is_not_a_minute(tmp_path):
    """The one place this format will silently corrupt a study.

    Unlike the hourly tick files, a Saturday day-file is not empty: it is 1,440
    records at Friday's close with volume zero, and Friday's own file pads
    21:00-23:59 the same way. Taken at face value a weekend "trade" returns a
    guaranteed 0 bp and a Friday-evening +60m becomes a 51-hour return.
    """
    bid, ask = padded_day(1440)
    days = {"2025-04-04": (bid, ask, 1260),   # Friday: live to 20:59, padded after
            "2025-04-05": (bid, ask, 0)}      # Saturday: all padding
    tape = C.MinuteTape(Store(tmp_path), "EURUSD", fetcher=padded_feed(days))

    assert len(tape.bars(date(2025, 4, 4))) == 1260
    assert tape.bars(date(2025, 4, 5)) == []
    assert tape.bar_at(ts("2025-04-04 22:00")) is None
    assert tape.bar_at(ts("2025-04-04 20:30")) is not None
    # 20:30 + 60 minutes lands in the padding, so the trade is dropped whole.
    assert C.forward_returns_minutes(tape, ts("2025-04-04 20:30"), +1,
                                     horizons_min=(60,)) is None
    assert C.forward_returns_minutes(tape, ts("2025-04-04 20:30"), +1,
                                     horizons_min=(15,)) is not None
    # And a signal at 20:55 has no bar to enter on at all: the next live minute
    # is Sunday evening, which ``next_bar_open`` deliberately will not reach.
    assert tape.next_bar_open(ts("2025-04-04 20:59") + 30) is None
    assert not C.has_bars(tape, ts("2025-04-05 10:00"), horizons_min=(15,))


def test_only_the_minutes_both_sides_have_survive(tmp_path):
    # A bid file with three minutes and an ask file with two: the third minute
    # has no spread and therefore no price to enter at.
    days = {"2025-04-02": ([100000, 100001, 100002], [100020, 100021])}
    tape = tape_for(tmp_path, days)
    assert [bar.ts for bar in tape.bars(date(2025, 4, 2))] == [
        ts("2025-04-02 00:00"), ts("2025-04-02 00:01")]


# ------------------------------------------------------------ the sign table


def test_the_sign_table_covers_all_seven_pairs_in_both_directions():
    assert R.wire_signed_pair("USD", R.STRONGER) == ("EURUSD", -1)
    assert R.wire_signed_pair("USD", R.WEAKER) == ("EURUSD", +1)
    assert R.wire_signed_pair("EUR", R.STRONGER) == ("EURUSD", +1)
    assert R.wire_signed_pair("JPY", R.STRONGER) == ("USDJPY", -1)
    assert R.wire_signed_pair("JPY", R.WEAKER) == ("USDJPY", +1)
    assert R.wire_signed_pair("GBP", R.STRONGER) == ("GBPUSD", +1)
    assert R.wire_signed_pair("AUD", R.STRONGER) == ("AUDUSD", +1)
    assert R.wire_signed_pair("CAD", R.STRONGER) == ("USDCAD", -1)
    assert R.wire_signed_pair("CHF", R.STRONGER) == ("USDCHF", -1)
    assert R.wire_signed_pair("NZD", R.STRONGER) == ("NZDUSD", +1)


def test_the_currencies_with_no_pair_in_the_judge_never_trade():
    for currency in ("CNY", "other", "none", "SEK", ""):
        assert R.wire_signed_pair(currency, R.STRONGER) is None
    for direction in (R.NO_DIRECTION, "", "sideways"):
        assert R.wire_signed_pair("EUR", direction) is None


def test_every_pair_the_table_names_is_one_the_judge_fetches():
    assert {pair for pair, _ in R.WIRE_PAIRS.values()} <= set(C.SYMBOLS)


# ------------------------------------------------------------------ the tree


def article(headline: str, *, when: str = "2025-04-02 10:00", body: str = "",
            section: str = "News", keywords: tuple[str, ...] = ()) -> W.Article:
    return W.Article(url=f"https://investinglive.com/news/{abs(hash(headline)) % 10**8}/",
                     published_ts=ts(when), headline=headline, section=section,
                     keywords=keywords, body=body or headline)


def test_round_one_asks_the_nine_questions_by_name():
    questions = R.wire_questions()
    assert set(questions) == {
        R.ABOUT_FX, R.CURRENCY, R.DIRECTION, R.CATEGORY, R.MAGNITUDE,
        R.NEW_INFORMATION, R.SCHEDULED, R.ALREADY_MOVED, R.IS_NUMBER,
    }
    assert set(questions[R.CURRENCY]["criteria"]) == set(R.WIRE_CURRENCIES)
    assert set(questions[R.DIRECTION]["criteria"]) == set(R.DIRECTIONS)
    assert set(questions[R.CATEGORY]["criteria"]) == set(R.CATEGORIES)
    assert len(questions[R.MAGNITUDE]["criteria"]) == 4
    for key in (R.ABOUT_FX, R.NEW_INFORMATION, R.SCHEDULED, R.ALREADY_MOVED, R.IS_NUMBER):
        assert questions[key]["type"] == "noul"


def test_round_two_asks_about_the_pair_and_not_about_the_currency():
    questions = R.wire_round_two_questions("USDJPY", "JPY")
    assert set(questions) == {R.HOLDER_UNAFFECTED, R.SIDE_OF_PAIR, R.HORIZON}
    assert "USDJPY" in questions[R.SIDE_OF_PAIR]["instructions"]
    assert set(questions[R.SIDE_OF_PAIR]["criteria"]) == {
        R.LONG_PAIR, R.SHORT_PAIR, R.NEITHER}


def test_the_wire_tree_has_its_own_version_and_does_not_share_a_cache_tag():
    assert R.WIRE_VERSION == "w1"
    assert R.tree_version(R.WIRE) == "w1"
    assert R.WIRE_VERSION not in (R.TREE_VERSION, R.CONTEXT_VERSION, R.PRESSER_VERSION)
    assert R.WIRE not in R.MODES  # it reads an Article, not a Document


def test_round_two_runs_only_when_the_post_is_about_fx_and_decisive():
    reader = R.WireReader(MockWireClient())
    decisive = reader.read(article("ECB's Lagarde: we will hike again, inflation is too high"))
    assert decisive.rounds == 2
    assert decisive.currency == "EUR"
    assert decisive.direction == R.STRONGER
    assert decisive.verdict.pair == "EURUSD" and decisive.verdict.sign == +1

    crypto = reader.read(article("Bitcoin surges through 100k as ETF flows build"))
    assert crypto.rounds == 1
    assert crypto.about_fx < 0.5
    assert crypto.verdict is None or crypto.verdict.strength == 0.0

    flat = reader.read(article("EURUSD technical: support at 1.0800, resistance at 1.0900"))
    assert flat.rounds == 1  # no direction, so no confirmation round


def test_no_second_round_means_no_trade():
    reader = R.WireReader(MockWireClient())
    reading = reader.read(article("EURUSD technical: support at 1.0800"))
    assert reading.confirm == 0.0
    assert reading.signals(0.01) == []


def test_the_strength_is_arithmetic_and_already_moved_cancels_it():
    assert R.wire_strength(1.0, 1.0, 1.0, 1.0, 0.0) == 1.0
    assert R.wire_strength(0.8, 0.5, 0.5, 1.0, 0.0) == pytest.approx(0.2)
    assert R.wire_strength(1.0, 1.0, 1.0, 1.0, 1.0) == 0.0
    assert R.wire_strength(1.0, 1.0, 1.0, 0.0, 0.0) == 0.0


def test_the_state_carries_the_clock_in_three_places():
    when = ts("2025-04-02 10:00")
    local = W.local_times(when)
    state = R.build_wire_state(article("x", when="2025-04-02 10:00"),
                               round_no=1, body_chars=100, local=local)
    assert state["published_utc"] == "2025-04-02 10:00:00 UTC"
    assert state["published_local"]["new_york"] == "2025-04-02 05:00"
    assert state["published_local"]["tokyo"] == "2025-04-02 19:00"
    assert state["round"] == 1


# ------------------------------------------------------------- the arms


def test_the_keyword_bot_names_a_currency_and_lets_the_words_sign_it():
    posts = [
        article("ECB's Lagarde: we will hike again"),           # EUR stronger
        article("BOJ cuts its forecast, yen weaker on the day"),  # JPY weaker
        article("Bitcoin surges through 100k"),                  # not FX
        article("China PBOC sets the yuan fix stronger"),         # CNY: no pair
        article("EURUSD daily technical outlook"),                # no direction
    ]
    signals = WS.keyword_signals(posts)
    assert [(s.pair, s.sign) for s in signals] == [("EURUSD", +1), ("USDJPY", +1)]


def test_the_sample_arm_is_seeded_and_reproducible():
    posts = [article(f"ECB's Lagarde hikes number {i}", when="2025-04-02 10:00")
             for i in range(50)]
    first = WS.sample_signals(posts, 10)
    again = WS.sample_signals(posts, 10)
    assert [s.code for s in first] == [s.code for s in again]
    assert len(first) == 10
    assert all(s.code.startswith("sample:") for s in first)
    # A sample larger than the corpus is the corpus.
    assert len(WS.sample_signals(posts, 500)) == 50


def test_the_session_buckets_are_one_rule():
    assert W.sessions(ts("2025-04-02 00:00")) == "asia"
    assert W.sessions(ts("2025-04-02 06:59")) == "asia"
    assert W.sessions(ts("2025-04-02 07:00")) == "london"
    assert W.sessions(ts("2025-04-02 12:59")) == "london"
    assert W.sessions(ts("2025-04-02 13:00")) == "new_york"
    assert W.sessions(ts("2025-04-02 20:59")) == "new_york"
    assert W.sessions(ts("2025-04-02 21:00")) == "late"
    assert W.sessions(ts("2025-04-02 23:59")) == "late"


def outcome(code: str, when: str, pair: str, sign: int, fwd: dict[int, float],
            pre: float = 0.0):
    from jevtrade.fx.study import Outcome

    return Outcome(signal=Signal(code, ts(when), code, pair, sign), symbol=pair,
                   pre_bps=pre, release_bar_bps=0.0, fwd_bps=fwd)


def test_the_breakdown_groups_and_signs_pre_by_the_side_taken():
    rows = WS.breakdown(
        [outcome("a", "2025-04-02 08:00", "EURUSD", +1, {15: 10.0}, pre=4.0),
         outcome("b", "2025-04-02 09:00", "EURUSD", -1, {15: -6.0}, pre=4.0),
         outcome("c", "2025-04-02 15:00", "USDJPY", +1, {15: 2.0}, pre=1.0)],
        WS.by_session, horizons=(15,))
    london, new_york = rows[0], rows[1]
    assert (london.key, london.n) == ("london", 2)
    assert london.fwd[15].mean == pytest.approx(2.0)
    assert london.hit[15] == pytest.approx(0.5)
    # +4 taken long and +4 taken short average to zero: pre is signed.
    assert london.pre.mean == pytest.approx(0.0)
    assert (new_york.key, new_york.n) == ("new_york", 1)


def test_the_breakdown_can_key_on_what_the_reader_said():
    reader = R.WireReader(MockWireClient())
    post = article("ECB's Lagarde: we will hike again")
    reading = reader.read(post)
    rows = WS.breakdown([outcome(post.id, "2025-04-02 10:00", "EURUSD", +1, {15: 3.0})],
                        WS.by_reading([], [reading], "category"), horizons=(15,))
    assert rows[0].key == reading.category
    boolean = WS.breakdown([outcome(post.id, "2025-04-02 10:00", "EURUSD", +1, {15: 3.0})],
                           WS.by_reading([], [reading], "is_number"), horizons=(15,))
    assert boolean[0].key in ("is_number=yes", "is_number=no")


def test_the_abstention_rate_is_per_category():
    reader = R.WireReader(MockWireClient())
    readings = [reader.read(article("ECB's Lagarde: we will hike again")),
                reader.read(article("EURUSD technical: support at 1.0800"))]
    rows = {a.category: a for a in WS.abstentions(readings, 0.05)}
    assert sum(a.read for a in rows.values()) == 2
    technical = rows[R.COMMENTARY]
    assert technical.traded == 0 and technical.rate == 1.0


def test_a_reading_survives_the_cache_round_trip():
    reader = R.WireReader(MockWireClient())
    post = article("ECB's Lagarde: we will hike again")
    reading = reader.read(post)
    back = WS.reading_from_dict(post, WS.reading_to_dict(reading))
    assert back.currency == reading.currency
    assert back.direction == reading.direction
    assert back.verdict.strength == pytest.approx(reading.verdict.strength)


def test_the_cache_key_changes_when_the_body_budget_does():
    post = article("x", body="y" * 5000)
    assert WS.state_hash(post, body_chars=1000) != WS.state_hash(post, body_chars=3000)


# --------------------------------------------------------------------- CLI


def weekday_feed(start: date, days: int):
    """Every weekday in the range priced, every weekend empty."""
    table: dict[str, tuple[list[int], list[int]]] = {}
    for offset in range(days):
        day = start + timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        opens = [100000 + (offset * 1440 + i) for i in range(1440)]
        table[day.isoformat()] = (opens, [v + 20 for v in opens])
    return feed(table)


def wire_pages(count: int = 6):
    """A sitemap index, one week, and ``count`` articles with real timestamps."""
    urls = [f"https://investinglive.com/news/post-{i}-20250402/" for i in range(count)]
    heads = [
        "ECB's Lagarde: we will hike again, inflation is too high",
        "BOJ leaves policy unchanged, the yen is weaker",
        "US CPI beats forecast at 3.2% vs 3.0% expected",
        "Bitcoin surges through 100k",
        "EURUSD technical: support at 1.0800",
        "Bank of England's Bailey says rates may need to rise",
    ]
    week = "<urlset>" + "".join(
        f"<url><loc>{u}</loc><lastmod>2025-04-02T1{i}:00:00+00:00</lastmod></url>"
        for i, u in enumerate(urls)) + "</urlset>"
    pages = {
        "https://investinglive.com/articles-sitemap-index.xml":
            "<sitemapindex><sitemap><loc>https://investinglive.com/sitemaps/news/"
            "articles/2025-W14.xml</loc></sitemap></sitemapindex>",
        "https://investinglive.com/sitemaps/news/articles/2025-W14.xml": week,
    }
    import json as _json

    for i, url in enumerate(urls):
        payload = _json.dumps({
            "@type": "NewsArticle", "headline": heads[i % len(heads)],
            "datePublished": f"2025-04-02T1{i}:00:00.1234567Z",
            "articleSection": "News", "keywords": "FX",
            "articleBody": heads[i % len(heads)],
        })
        pages[url] = ('<html><head><script type="application/ld+json">'
                      f"{payload}</script></head><body></body></html>")
    return pages


def cli(monkeypatch, tmp_path, *extra: str, robots: str = OPEN_ROBOTS):
    from jevtrade.cli import main

    pages = wire_pages()
    monkeypatch.setattr(W, "get_text", lambda url, timeout=30.0, query="": pages[url])
    monkeypatch.setattr(C, "fetch_bi5", weekday_feed(date(2025, 3, 25), 20))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    store = Store(tmp_path / "cache")
    store.put("wire:robots", robots)
    return main([
        "fx", "--wire", "--provider", "mock", "--cache", str(tmp_path / "cache"),
        "--since", "2025-04-01", "--until", "2025-04-04", *extra,
    ])


def test_cli_collect_only_scrapes_and_stops(tmp_path, monkeypatch, capsys):
    assert cli(monkeypatch, tmp_path, "--collect-only") == 0
    out = capsys.readouterr().out
    assert "robots.txt: articles allowed" in out
    assert "6 articles from 6 links" in out
    assert "ISO weeks" in out
    assert "reader (" not in out  # it stopped before reading anything


def test_cli_refuses_the_whole_run_when_robots_disallows(tmp_path, monkeypatch, capsys):
    code = cli(monkeypatch, tmp_path, "--collect-only", robots=REAL_ROBOTS)
    out = capsys.readouterr().out
    assert code == 3
    assert "DISALLOWED for ClaudeBot" in out
    assert "nothing will be fetched" in out
    assert "No cached corpus either" in out


def test_an_offline_collect_reads_the_cache_and_sends_nothing(tmp_path):
    """The mode for a run against a cache somebody the wire permits filled.

    Nothing is fetched, so nothing needs permission; a link the cache does not
    hold is counted rather than requested.
    """
    store = Store(tmp_path)
    store.put("wire:robots", OPEN_ROBOTS)

    def fetcher(url: str) -> str:
        if url.endswith("articles-sitemap-index.xml"):
            return SITEMAP_INDEX
        if "2025-W" in url:
            return WEEK
        if "ecb-lagarde" in url:
            return ARTICLE_META
        raise TimeoutError("not cached in this test")

    # Fill the cache with the week and one of its three articles.
    W.article_urls(store, ts("2025-04-01 00:00"), ts("2025-04-07 23:59"),
                   fetcher=fetcher, workers=1)
    W.fetch_article(store, "https://investinglive.com/central-banks/"
                    "ecb-lagarde-speaks-20250407/", fetcher=fetcher)
    store.put("wire:robots", REAL_ROBOTS)  # the wire now says no

    def explode(_url: str) -> str:
        raise AssertionError("offline mode sent a request")

    articles, coverage = W.collect(store, ts("2025-04-01 00:00"), ts("2025-04-07 23:59"),
                                   workers=1, fetcher=explode, offline=True,
                                   progress_every=0)
    assert len(articles) == 1
    assert coverage.uncached == 2
    assert coverage.unparsed == 0


def test_an_offline_collect_with_an_empty_cache_finds_nothing(tmp_path):
    store = Store(tmp_path)
    store.put("wire:robots", REAL_ROBOTS)
    articles, coverage = W.collect(store, ts("2025-04-01 00:00"), ts("2025-04-07 23:59"),
                                   workers=1, offline=True, progress_every=0)
    assert articles == [] and coverage.urls == 0


def test_cli_warms_the_candles_without_asking_the_wire(tmp_path, monkeypatch, capsys):
    # The price feed is a different host with no such rule, so this path runs
    # even when the wire has said no.
    code = cli(monkeypatch, tmp_path, "--warm-candles", robots=REAL_ROBOTS)
    out = capsys.readouterr().out
    assert code == 0
    assert "day-files on disk" in out
    assert "EURUSD: 8/8 files on disk" in out


def test_cli_runs_the_whole_wire_study_on_the_mock(tmp_path, monkeypatch, capsys):
    code = cli(monkeypatch, tmp_path, "--horizons", "5,15,60", "--sample", "4",
               "--null-per", "1", "--out", str(tmp_path / "run.json"))
    out = capsys.readouterr().out
    assert code == 0
    assert "6 posts" in out
    assert "reader (mock, w1)" in out
    assert "keyword-bot" in out
    assert "wire-sample" in out
    assert "where the reader stayed out" in out
    assert "reader, by category" in out
    assert "reader, by session (UTC)" in out
    assert "latency sweep" in out

    import json

    record = json.loads((tmp_path / "run.json").read_text())
    assert record["tree"] == "w1"
    assert record["robots"]["articles_allowed"] is True
    assert {a["name"] for a in record["arms"]} >= {"keyword-bot", "wire-sample"}
    assert len(record["readings"]) == 6
    assert any(t["table"] == "reader, by category" for t in record["breakdowns"])
    assert record["arms"][0]["outcomes"][0]["session"] in W.SESSIONS
