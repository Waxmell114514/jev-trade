"""The news study's measurement machinery.

These test the instrument, not the finding: parsing, alignment, and above all
that the before/after comparison is symmetric. An asymmetric measure would
manufacture a 'headlines predict the future' result out of nothing.
"""

import math

import pytest

from jevtrade.news.feeds import Headline, _normalise, fetch_feed
from jevtrade.news.label import Bars, label
from jevtrade.news.study import excursion, keyword_hit, matched_null, score_arm

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Channel title, not a story</title>
  <item><title>SEC approves the thing</title>
        <pubDate>Wed, 17 Sep 2026 12:00:00 +0000</pubDate></item>
  <item><title><![CDATA[Exchange denies halt rumours]]></title>
        <pubDate>Wed, 17 Sep 2026 12:30:00 +0000</pubDate></item>
</channel></rss>"""


def flat_bars(n=200, interval=5, sigma=0.001):
    """A perfectly flat tape: every excursion must be zero."""
    return Bars(interval_min=interval, ts=[i * interval * 60 for i in range(n)],
                close=[100.0] * n, sigma=sigma)


def ramp_bars(n=200, interval=5, step=0.001, sigma=0.001):
    close = [100.0 * math.exp(step * i) for i in range(n)]
    return Bars(interval_min=interval, ts=[i * interval * 60 for i in range(n)],
                close=close, sigma=sigma)


# -------------------------------------------------------------------- feeds


def test_feed_parsing_pairs_each_title_with_its_own_timestamp(tmp_path):
    path = tmp_path / "f.xml"
    path.write_text(RSS)
    rows = fetch_feed("test", path.as_uri())
    assert [r.title for r in rows] == [
        "SEC approves the thing", "Exchange denies halt rumours"
    ]
    assert rows[1].ts - rows[0].ts == 1800  # half an hour apart


def test_channel_title_is_not_treated_as_a_story(tmp_path):
    path = tmp_path / "f.xml"
    path.write_text(RSS)
    assert all("Channel title" not in r.title for r in fetch_feed("t", path.as_uri()))


def test_near_duplicates_across_sources_collapse():
    a = "SEC Approves Spot Bitcoin ETF For Major Issuer"
    b = "SEC approves spot bitcoin ETF for major issuer!"
    assert _normalise(a) == _normalise(b)


# ------------------------------------------------------------------- labels


def test_a_flat_tape_produces_no_movers():
    bars = flat_bars()
    rows = label([Headline(ts=t, title="x", source="s")
                  for t in range(3000, 40_000, 900)], bars, horizon_bars=3)
    assert rows
    assert all(r.post == 0.0 and r.pre == 0.0 for r in rows)
    assert not any(r.moved(2.0) for r in rows)


def test_headlines_outside_the_price_window_are_dropped():
    bars = flat_bars(n=50)
    far = Headline(ts=bars.ts[-1] + 999_999, title="x", source="s")
    assert label([far], bars, horizon_bars=3) == []


def test_a_move_before_the_headline_marks_it_reactive_not_clean():
    bars = ramp_bars(step=0.01)  # a strong steady trend both sides
    rows = label([Headline(ts=bars.ts[100], title="x", source="s")],
                 bars, horizon_bars=3)
    assert rows[0].pre > 2.0 and rows[0].post > 2.0
    assert rows[0].moved(2.0)
    assert rows[0].reactive(2.0)
    assert not rows[0].clean_mover(2.0)  # too late to act on


def test_index_at_finds_the_containing_bar():
    bars = flat_bars(n=10)
    assert bars.index_at(bars.ts[3] + 1) == 3
    assert bars.index_at(bars.ts[0] - 1) is None


# -------------------------------------------------------------- the measure


def test_before_and_after_excursions_are_symmetric():
    """The heart of it: on a symmetric tape the two must agree exactly.

    If 'after' were a maximum over horizons while 'before' was a single
    window, every group would look forward-looking for free.
    """
    bars = ramp_bars(step=0.002)
    for h in (1, 3, 6):
        before = excursion(bars, bars.ts[100], h, forward=False)
        after = excursion(bars, bars.ts[100], h, forward=True)
        assert before == pytest.approx(after, rel=1e-9)


def test_excursion_scales_out_the_horizon():
    """Sigma is per bar, so a random-walk-sized move reads the same at any h."""
    bars = ramp_bars(step=0.001, sigma=0.001)
    one = excursion(bars, bars.ts[100], 1, forward=True)
    assert one == pytest.approx(1.0, rel=1e-6)


def test_excursion_is_none_at_the_edges():
    bars = flat_bars(n=20)
    assert excursion(bars, bars.ts[0], 3, forward=False) is None
    assert excursion(bars, bars.ts[-1], 3, forward=True) is None


def test_matched_null_on_a_flat_tape_is_zero():
    bars = flat_bars()
    mean, sd = matched_null(bars, (bars.ts[10], bars.ts[-10]),
                            pre_low=0.0, pre_high=99.0, n=5, trials=20)
    assert mean == pytest.approx(0.0)


def test_score_arm_reports_the_firing_rate():
    bars = flat_bars()
    rows = label([Headline(ts=t, title="x", source="s")
                  for t in range(3000, 40_000, 900)], bars, horizon_bars=3)
    result = score_arm("half", rows[: len(rows) // 2], rows, bars,
                       threshold=2.0, horizon_bars=3)
    assert result.fire_rate == pytest.approx(0.5, abs=0.05)
    assert result.movers == 0


# ------------------------------------------------------------------ keyword


def test_keyword_rule_fires_on_market_moving_vocabulary():
    assert keyword_hit("SEC approves spot Bitcoin ETF")
    assert keyword_hit("Exchange halts withdrawals after breach")
    assert keyword_hit("Fed hikes rates")


def test_keyword_rule_ignores_ordinary_copy():
    assert not keyword_hit("Analysts discuss the long-term outlook for adoption")
    assert not keyword_hit("Diego Kochen included in squad for friendly")


def test_a_word_can_mean_two_things_and_the_rule_cannot_tell():
    """'Settlement' is a lawsuit outcome and a payments term. A desk list has
    to carry it for the first sense, and then fires on the second."""
    assert keyword_hit("Exchange reaches settlement with the SEC")
    assert keyword_hit("Conference panel debates on-chain settlement")  # false alarm


def test_keyword_rule_uses_word_boundaries():
    assert not keyword_hit("Analyst reiterates a long-term view")
