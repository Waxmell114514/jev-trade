"""A corpus the user brought: parsing it, merging it, and running the study on it.

Every fixture here is written inline. The uploaded archive is the user's file,
it is read from where it lies and never copied into this repository, so nothing
in these tests depends on it being present.

The load-bearing assertion is ``test_a_posts_run_never_consults_robots_txt``:
``--posts`` is a file the user already has, not a fetch, so the wire's
``robots.txt`` must not be looked at even once. The test proves it by making any
wire fetch raise.
"""

from __future__ import annotations

import json
import lzma
import struct
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from jevtrade.fx import candles as C
from jevtrade.fx import posts as P
from jevtrade.fx import wire as W
from jevtrade.fx import wirestudy as WS
from jevtrade.fx.study import Signal
from jevtrade.listing.store import Store


def ts(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(
        tzinfo=timezone.utc).timestamp()


# The two shapes the archive arrives in, three ids: one plain, one whose text is
# missing from the posts capture and present in the statuses one, and one that
# is empty in both because it was a picture.
POSTS_ROWS = [
    {"post_id": "1", "status_id": "1", "author": "Donald J. Trump",
     "handle": "@realDonaldTrump", "published_at": "January 2, 2025, 10:00 AM",
     "has_search_text": True, "text_characters": 61,
     "archive_url": "https://www.trumpstruth.org/statuses/1",
     "original_url": "https://truthsocial.com/@realDonaldTrump/111",
     "text": "Our Interest Rates are TOO HIGH, and Too Late Powell knows it!",
     "date_published": "2025-01-02T15:00:00+00:00",
     "capture_date": "2026-09-18T10:51:31+00:00", "fetched": True},
    {"post_id": "2", "status_id": "2", "author": "Donald J. Trump",
     "handle": "@realDonaldTrump", "published_at": "January 3, 2025, 11:00 AM",
     "has_search_text": False, "text_characters": 0,
     "archive_url": "https://www.trumpstruth.org/statuses/2",
     "original_url": "https://truthsocial.com/@realDonaldTrump/222",
     "text": "", "date_published": "2025-01-03T16:00:00+00:00",
     "capture_date": "2026-09-18T10:51:31+00:00", "fetched": True},
    {"post_id": "3", "status_id": "3", "author": "Donald J. Trump",
     "handle": "@realDonaldTrump", "published_at": "January 4, 2025, 12:00 PM",
     "has_search_text": False, "text_characters": 0,
     "archive_url": "https://www.trumpstruth.org/statuses/3",
     "original_url": "https://truthsocial.com/@realDonaldTrump/333",
     "text": "", "date_published": "2025-01-04T17:00:00+00:00",
     "capture_date": "2026-09-18T10:51:31+00:00", "fetched": True},
    # The same id again, exactly as the real posts file repeats 563 of them.
    {"post_id": "1", "status_id": "1", "author": "Donald J. Trump",
     "handle": "@realDonaldTrump", "published_at": "January 2, 2025, 10:00 AM",
     "has_search_text": True, "text_characters": 61,
     "archive_url": "https://www.trumpstruth.org/statuses/1",
     "original_url": "https://truthsocial.com/@realDonaldTrump/111",
     "text": "Our Interest Rates are TOO HIGH, and Too Late Powell knows it!",
     "date_published": "2025-01-02T15:00:00+00:00",
     "capture_date": "2026-09-18T10:51:31+00:00", "fetched": True},
]

STATUS_ROWS = [
    {"status_id": "1", "url": "https://www.trumpstruth.org/statuses/1",
     "headline": 'Donald J. Trump: "Our Interest Rates are TOO HIGH, and Too Late..."',
     "description": "Our Interest Rates are TOO HIGH",
     "date_published": "2025-01-02T15:00:00+00:00", "author_name": "Donald J. Trump",
     "author_url": "https://truthsocial.com/@realDonaldTrump",
     "original_url": "https://truthsocial.com/@realDonaldTrump/111",
     "capture_date": "2026-09-15T18:40:52+00:00",
     "article_body": "Our Interest Rates are TOO HIGH, and Too Late Powell knows it!"},
    {"status_id": "2", "url": "https://www.trumpstruth.org/statuses/2",
     "headline": 'Donald J. Trump: "Tariffs on China are making us RICH!"',
     "description": "Tariffs on China", "date_published": "2025-01-03T16:00:00+00:00",
     "author_name": "Donald J. Trump",
     "author_url": "https://truthsocial.com/@realDonaldTrump",
     "original_url": "https://truthsocial.com/@realDonaldTrump/222",
     "capture_date": "2026-09-15T18:40:52+00:00",
     "article_body": "Tariffs on China are making us RICH!"},
    {"status_id": "3", "url": "https://www.trumpstruth.org/statuses/3",
     "headline": "Donald J. Trump image post from January 4, 2025",
     "description": "", "date_published": "2025-01-04T17:00:00+00:00",
     "author_name": "Donald J. Trump",
     "author_url": "https://truthsocial.com/@realDonaldTrump",
     "original_url": "https://truthsocial.com/@realDonaldTrump/333",
     "capture_date": "2026-09-15T18:40:52+00:00", "article_body": ""},
]

CSV_FIELDS = ["post_id", "published_at", "author", "handle", "has_search_text",
              "text_characters", "archive_url", "status_id", "text", "original_url",
              "date_published", "capture_date", "fetched"]


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def write_csv(path: Path, rows, *, bom: bool = True) -> Path:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    path.write_text(("﻿" if bom else "") + buf.getvalue(), encoding="utf-8")
    return path


# ------------------------------------------------------------------ parsing


def test_the_posts_shape_loads(tmp_path):
    articles, coverage = P.load_posts([write_jsonl(tmp_path / "posts.jsonl", POSTS_ROWS)])
    assert coverage.lines == 4
    assert coverage.distinct == 3
    assert coverage.duplicates == 1
    assert coverage.dropped_empty == 2
    assert coverage.loaded == 1
    article = articles[0]
    assert article.section == "truth_social"
    assert article.source == "posts-file"
    assert article.source_id == "1"
    assert article.keywords == ()
    assert article.url == "https://truthsocial.com/@realDonaldTrump/111"
    assert article.published_ts == ts("2025-01-02 15:00")
    assert article.body.startswith("Our Interest Rates are TOO HIGH")


def test_the_statuses_shape_loads_from_its_own_field_names(tmp_path):
    articles, coverage = P.load_posts([write_jsonl(tmp_path / "s.jsonl", STATUS_ROWS)])
    assert coverage.loaded == 2 and coverage.dropped_empty == 1
    assert {a.source_id for a in articles} == {"1", "2"}
    assert articles[1].body == "Tariffs on China are making us RICH!"


def test_the_csv_with_a_bom_loads_the_same_posts(tmp_path):
    from_json, _ = P.load_posts([write_jsonl(tmp_path / "p.jsonl", POSTS_ROWS)])
    from_csv, coverage = P.load_posts([write_csv(tmp_path / "p.csv", POSTS_ROWS)])
    assert coverage.loaded == 1
    # Without utf-8-sig the first column is named "﻿post_id" and every
    # lookup of post_id silently misses, so this is the assertion that matters.
    assert [a.id for a in from_csv] == [a.id for a in from_json]
    assert from_csv[0].source_id == "1"


def test_a_second_capture_fills_a_text_the_first_one_is_missing(tmp_path):
    """The rule for two captures of one archive -- which this archive never needs.

    Both real captures are missing the same 1,305 posts, so ``filled`` is 0 on
    the live corpus; the fixture makes the case the rule exists for, and the
    count reports it either way rather than the code pretending it fired.
    """
    articles, coverage = P.load_posts([
        write_jsonl(tmp_path / "p.jsonl", POSTS_ROWS),
        write_jsonl(tmp_path / "s.jsonl", STATUS_ROWS),
    ])
    assert coverage.filled == 1  # id "2" was empty in posts and present in statuses
    assert coverage.loaded == 2
    assert coverage.dropped_empty == 1  # id "3" is a picture in both
    bodies = {a.source_id: a.body for a in articles}
    assert bodies["2"] == "Tariffs on China are making us RICH!"


def test_the_merge_does_not_care_which_file_comes_first(tmp_path):
    forward = P.load_posts([write_jsonl(tmp_path / "p.jsonl", POSTS_ROWS),
                            write_jsonl(tmp_path / "s.jsonl", STATUS_ROWS)])[0]
    backward = P.load_posts([tmp_path / "s.jsonl", tmp_path / "p.jsonl"])[0]
    assert [(a.source_id, a.body) for a in forward] == \
           [(a.source_id, a.body) for a in backward]


def test_posts_are_sorted_oldest_first_and_counted_by_month(tmp_path):
    rows = [dict(POSTS_ROWS[0], status_id=str(i), post_id=str(i),
                 date_published=f"2025-0{1 + i % 3}-1{i}T0{i}:00:00+00:00")
            for i in range(1, 6)]
    articles, coverage = P.load_posts([write_jsonl(tmp_path / "p.jsonl", rows)])
    assert [a.published_ts for a in articles] == sorted(a.published_ts for a in articles)
    assert sum(coverage.per_month.values()) == coverage.loaded == 5
    assert set(coverage.per_month) == {"2025-01", "2025-02", "2025-03"}


def test_a_row_with_no_timestamp_or_no_id_is_dropped_and_counted(tmp_path):
    rows = [dict(POSTS_ROWS[0], status_id="9", post_id="9", date_published=""),
            {"text": "no id at all", "date_published": "2025-01-02T15:00:00+00:00"}]
    _articles, coverage = P.load_posts([write_jsonl(tmp_path / "p.jsonl", rows)])
    assert coverage.no_timestamp == 1
    assert coverage.unreadable == 1
    assert coverage.loaded == 0


def test_an_unparseable_line_is_counted_and_does_not_stop_the_file(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps(POSTS_ROWS[0]) + "\n{ not json\n\n[1,2]\n", encoding="utf-8")
    articles, coverage = P.load_posts([path])
    assert coverage.loaded == 1 and coverage.unreadable == 2
    assert articles[0].source_id == "1"


def test_the_timestamp_is_the_iso_field_and_not_the_prose_one(tmp_path):
    # ``published_at`` is "January 2, 2025, 10:00 AM" with no zone at all; the
    # graded moment comes from ``date_published`` or the post is dropped.
    articles, _ = P.load_posts([write_jsonl(tmp_path / "p.jsonl", POSTS_ROWS)])
    assert articles[0].when.strftime("%Y-%m-%d %H:%M UTC") == "2025-01-02 15:00 UTC"


# ----------------------------------------------------------------- headlines


def test_the_headline_is_the_first_120_characters_cut_at_a_word():
    long = ("The United States has set a World Record on investments being made into a "
            "Country, and it is Trillions of Dollars more than number two, China.")
    head = P.headline_of(long)
    assert len(head) <= P.HEADLINE_CHARS + 3
    assert head.endswith("...")
    assert not head[:-3].endswith(" ")
    assert long.startswith(head[:-3])
    short = "Happy New Year to all."
    assert P.headline_of(short) == short  # no ellipsis when it already fits
    assert P.headline_of("  ragged\n\n  whitespace  ") == "ragged whitespace"


def test_the_archive_headline_is_only_a_fallback_and_loses_its_author_prefix():
    assert P.strip_headline('Donald J. Trump: "Happy New Year to all."') == \
        "Happy New Year to all."
    assert P.strip_headline('Marjorie Taylor Greene: "RT: something"') == "RT: something"
    # A media post's headline describes an absence; it is never a headline.
    assert P.strip_headline("Donald J. Trump image post from January 1, 2025") == ""
    assert P.headline_of("", 'Donald J. Trump: "fallback text"') == "fallback text"


def test_a_post_uses_its_own_text_over_the_archives_truncated_headline(tmp_path):
    articles, _ = P.load_posts([write_jsonl(tmp_path / "s.jsonl", STATUS_ROWS)])
    first = next(a for a in articles if a.source_id == "1")
    # The archive's headline ends "...Too Late..." mid-sentence; the body does not.
    assert first.headline == "Our Interest Rates are TOO HIGH, and Too Late Powell knows it!"


# --------------------------------------------------------------- the cache


def test_the_corpus_goes_into_its_own_cache_namespace_and_comes_back(tmp_path):
    store = Store(tmp_path)
    articles, _ = P.load_posts([write_jsonl(tmp_path / "p.jsonl", POSTS_ROWS),
                                write_jsonl(tmp_path / "s.jsonl", STATUS_ROWS)])
    assert P.store_posts(store, articles) == 2
    assert store.get("wire:posts:1")[0] is True
    # Never under the wire's own namespace: a cache filled from a file must not
    # look like a cache filled from the network.
    assert store.get(f"wire:article:{articles[0].url}")[0] is False
    back = P.cached_posts(store, ts("2025-01-01 00:00"), ts("2025-01-31 00:00"))
    assert [(a.source_id, a.body, a.published_ts) for a in back] == \
           [(a.source_id, a.body, a.published_ts) for a in articles]
    # ... and the window is applied.
    assert P.cached_posts(store, ts("2025-01-03 00:00"), ts("2025-01-31 00:00")) == back[1:]


def test_importing_twice_keeps_one_index_entry_per_post(tmp_path):
    store = Store(tmp_path)
    path = write_jsonl(tmp_path / "p.jsonl", POSTS_ROWS)
    P.import_posts(store, [path], 0.0, 4e9, progress=False)
    P.import_posts(store, [path], 0.0, 4e9, progress=False)
    assert store.get(P.INDEX_KEY)[1] == ["1"]


# --------------------------------------------------------------- the weekend


def test_the_weekday_label_is_the_posts_own_day():
    assert P.weekday_label(ts("2025-01-03 12:00")) == "weekday"   # Friday
    assert P.weekday_label(ts("2025-01-04 12:00")) == "weekend"   # Saturday
    assert P.weekday_label(ts("2025-01-05 12:00")) == "weekend"   # Sunday
    assert P.weekday_label(ts("2025-01-06 12:00")) == "weekday"   # Monday
    assert P.is_weekend(ts("2025-01-04 12:00")) is True


def test_what_the_tape_could_not_price_is_counted_apart_from_what_it_could():
    """"The tape was shut" and "the reader stayed out" are different findings."""
    from jevtrade.fx.study import Outcome

    signals = [
        Signal("a", ts("2025-01-03 12:00"), "", "EURUSD", +1),   # Friday, priced
        Signal("b", ts("2025-01-04 12:00"), "", "EURUSD", +1),   # Saturday, shut
        Signal("c", ts("2025-01-05 12:00"), "", "EURUSD", -1),   # Sunday, shut
        Signal("d", ts("2025-01-06 12:00"), "", "EURUSD", +1),   # Monday, no bars
    ]
    outcomes = [Outcome(signal=signals[0], symbol="EURUSD", pre_bps=0.0,
                        release_bar_bps=0.0, fwd_bps={15: 1.0})]
    shut = WS.shut_out(signals, outcomes)
    assert (shut.signals, shut.measured, shut.unmeasured) == (4, 1, 3)
    assert (shut.posted_weekday, shut.posted_weekend) == (2, 2)
    assert (shut.unmeasured_weekend, shut.unmeasured_weekday) == (2, 1)
    assert "2 posted at a weekend" in shut.summary()


def test_the_weekday_breakdown_splits_the_traded_outcomes():
    from jevtrade.fx.study import Outcome

    def out(when: str, fwd: float):
        return Outcome(signal=Signal(when, ts(when), "", "EURUSD", +1), symbol="EURUSD",
                       pre_bps=0.0, release_bar_bps=0.0, fwd_bps={15: fwd})

    rows = {c.key: c for c in WS.breakdown(
        [out("2025-01-03 12:00", 4.0), out("2025-01-06 12:00", 2.0),
         out("2025-01-04 12:00", -1.0)], WS.by_weekday, horizons=(15,))}
    assert rows["weekday"].n == 2 and rows["weekday"].fwd[15].mean == pytest.approx(3.0)
    assert rows["weekend"].n == 1


# ------------------------------------------------------------------- mock


def test_the_mock_reads_this_corpus_vocabulary():
    """The offline rule has to see these words or the smoke measures nothing."""
    from jevtrade.fx import reader as R
    from jevtrade.fx.mock import MockWireClient

    reader = R.WireReader(MockWireClient())

    def read(text: str):
        return reader.read(W.Article(url="https://x/1/", published_ts=ts("2025-06-02 14:00"),
                                     headline=text, section=P.SECTION, body=text))

    fed = read("Our Interest Rates are TOO HIGH. Too Late Powell is costing us a fortune!")
    assert fed.currency == "USD" and fed.direction == R.WEAKER
    assert fed.verdict.pair == "EURUSD" and fed.verdict.sign == +1
    assert fed.rounds == 2

    assert read("Tariffs on China are making us RICH!").currency == "CNY"
    assert read("The European Union must lower their rates").currency == "EUR"
    assert read("Japan is investing Trillions in America").currency == "JPY"
    assert read("Canada should become the 51st state").currency == "CAD"
    # No MXN pair in the judge, so a Mexico post is classified and never traded.
    assert R.wire_signed_pair("MXN", R.STRONGER) is None
    assert read("Mexico is sending us their worst").verdict is None
    assert read("The dollar is the strongest it has ever been").currency == "USD"


def test_the_keyword_bot_and_the_mock_read_the_same_words():
    """One lexicon, so "the model beat the words" stays a comparable claim."""
    text = "Our Interest Rates are TOO HIGH, and Too Late Powell knows it!"
    assert W.currency_of(text) == "USD"
    assert W.direction_of(text) == "weaker"
    signals = WS.keyword_signals([W.Article(url="https://x/1/", published_ts=1.0,
                                            headline=text, body=text)])
    assert [(s.pair, s.sign) for s in signals] == [("EURUSD", +1)]


# --------------------------------------------------------------------- CLI


def day_bytes(opens: list[int]) -> bytes:
    return lzma.compress(b"".join(
        struct.pack(">IIIIIf", i * 60, v, v, v - 5, v + 5, 100.0)
        for i, v in enumerate(opens)))


def weekday_feed(start: date, days: int):
    """Every weekday priced, every weekend an empty file -- as the tape really is."""
    table: dict[str, list[int]] = {}
    for offset in range(days):
        day = start + timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        table[day.isoformat()] = [100000 + (offset * 1440 + i) for i in range(1440)]

    def fetch(url: str) -> bytes:
        parts = url.rstrip("/").split("/")
        key = f"{int(parts[-4]):04d}-{int(parts[-3]) + 1:02d}-{int(parts[-2]):02d}"
        if key not in table:
            return b""
        opens = table[key]
        return day_bytes([v + 20 for v in opens] if parts[-1].startswith("ASK") else opens)

    return fetch


CLI_ROWS = [
    dict(POSTS_ROWS[0], status_id=str(i), post_id=str(i),
         text=text, date_published=f"2025-01-{day:02d}T1{i % 8}:00:00+00:00")
    for i, (day, text) in enumerate([
        (2, "Our Interest Rates are TOO HIGH, Too Late Powell must cut NOW!"),
        (3, "Tariffs on China are making our Country RICH again!"),
        (6, "The European Union has agreed to lower their barriers, a weak deal"),
        (7, "Canada should become the 51st State, their economy is weak"),
        (8, "Japan is investing Trillions of Dollars, the strongest ever"),
        (4, "A GREAT weekend rally, the dollar is strongest in history!"),
    ], start=1)
]


def run_cli(tmp_path, monkeypatch, *extra: str, files=None):
    from jevtrade.cli import main

    def no_fetching(*_a, **_k):
        raise AssertionError("a --posts run touched the wire")

    monkeypatch.setattr(W, "get_text", no_fetching)
    monkeypatch.setattr(C, "fetch_bi5", weekday_feed(date(2024, 12, 28), 30))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    paths = files or [str(write_jsonl(tmp_path / "posts.jsonl", CLI_ROWS))]
    return main([
        "fx", "--wire", "--posts", *paths, "--provider", "mock",
        "--cache", str(tmp_path / "cache"), "--since", "2025-01-01",
        "--until", "2025-01-10", *extra,
    ])


def test_cli_posts_collect_only_imports_and_stops(tmp_path, monkeypatch, capsys):
    assert run_cli(tmp_path, monkeypatch, "--collect-only") == 0
    out = capsys.readouterr().out
    assert "robots.txt is not consulted" in out
    assert "6 posts with text" in out
    assert "per month: 2025-01 6" in out
    assert "on a weekday" in out and "at a weekend" in out
    assert "reader (" not in out  # it stopped before reading anything
    # The corpus is in the cache under its own namespace, ready for a later run.
    store = Store(tmp_path / "cache")
    assert len(store.get(P.INDEX_KEY)[1]) == 6
    assert store.get("wire:robots")[0] is False  # robots was never even fetched


def test_a_posts_run_never_consults_robots_txt(tmp_path, monkeypatch, capsys):
    """The assertion the whole flag exists for: a local file is not a request."""
    store = Store(tmp_path / "cache")
    store.put("wire:robots", (Path(__file__).parent / "fixtures" / "fx"
                              / "investinglive-robots.txt").read_text())
    assert run_cli(tmp_path, monkeypatch, "--collect-only") == 0
    out = capsys.readouterr().out
    # The wire's robots.txt disallows this client outright, and it makes no
    # difference at all, because nothing is being fetched from the wire.
    assert "DISALLOWED" not in out
    assert "robots.txt is not consulted" in out


def test_cli_runs_the_whole_study_on_an_imported_corpus(tmp_path, monkeypatch, capsys):
    code = run_cli(tmp_path, monkeypatch, "--horizons", "5,15,60", "--sample", "3",
                   "--null-per", "1", "--out", str(tmp_path / "run.json"))
    out = capsys.readouterr().out
    assert code == 0
    assert "6 posts, 2025-01-02" in out
    assert "reader (mock, w1)" in out
    assert "what the tape could and could not price" in out
    assert "reader, weekday or weekend" in out
    assert "reader, by category" in out
    assert "latency sweep" in out

    record = json.loads((tmp_path / "run.json").read_text())
    assert record["corpus"] == "posts-file"
    assert record["robots"] == {"consulted": False}
    assert record["tree"] == "w1"
    assert len(record["posts_files"]) == 1
    assert record["coverage"]["loaded"] == 6
    assert any(row["arm"].startswith("reader") for row in record["shut"])
    assert any(t["table"] == "reader, weekday or weekend" for t in record["breakdowns"])
    # A Saturday post cannot be graded, so it is in the corpus and not in the trades.
    shut = next(r for r in record["shut"] if r["arm"].startswith("reader"))
    assert shut["signals"] >= shut["measured"]


def test_cli_accepts_the_csv_as_an_alternative_input(tmp_path, monkeypatch, capsys):
    path = str(write_csv(tmp_path / "posts.csv", CLI_ROWS))
    assert run_cli(tmp_path, monkeypatch, "--collect-only", files=[path]) == 0
    assert "6 posts with text" in capsys.readouterr().out


def test_cli_takes_several_files_and_merges_them(tmp_path, monkeypatch, capsys):
    a = str(write_jsonl(tmp_path / "a.jsonl", CLI_ROWS[:3]))
    b = str(write_jsonl(tmp_path / "b.jsonl", CLI_ROWS[2:]))
    assert run_cli(tmp_path, monkeypatch, "--collect-only", files=[a, b]) == 0
    out = capsys.readouterr().out
    assert "6 posts with text" in out
    assert "in 2 file(s)" in out
    assert "1 rows merged into an id already seen" in out
