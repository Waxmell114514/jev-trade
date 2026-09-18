"""Features are pure arithmetic; the state Jev sees is pure vocabulary."""

import math

from jevtrade.discretize import (
    ALL_VOCABULARIES,
    BOOK,
    BOOK_CUTS,
    MOVE,
    MOVE_CUTS,
    bucket,
    build_state,
)
from jevtrade.features import FeatureWindow
from jevtrade.feed import SyntheticConfig, SyntheticFeed
from jevtrade.types import Tick


def _tick(seq, price, volume=1.0, bid_size=1.0, ask_size=1.0, spread=1.0):
    return Tick(
        seq=seq,
        ts=float(seq),
        symbol="X-USD",
        bid=price - spread / 2,
        ask=price + spread / 2,
        last=price,
        volume=volume,
        bid_size=bid_size,
        ask_size=ask_size,
    )


def test_window_withholds_features_until_warm():
    window = FeatureWindow(size=50, warmup=10)
    for seq in range(9):
        assert window.update(_tick(seq, 100.0)) is None
    assert window.update(_tick(9, 100.0)) is not None


def test_returns_and_imbalance_are_computed_correctly():
    window = FeatureWindow(size=50, warmup=3)
    for seq in range(3):
        window.update(_tick(seq, 100.0))
    features = window.update(_tick(3, 101.0, bid_size=3.0, ask_size=1.0))

    assert math.isclose(features.ret_1_bps, math.log(101 / 100) * 1e4, rel_tol=1e-9)
    assert math.isclose(features.imbalance, 0.5)  # (3-1)/4
    assert features.run_length == 1


def test_run_length_is_signed_and_counts_consecutive_moves():
    window = FeatureWindow(size=50, warmup=2)
    prices = [100, 101, 102, 103, 102]
    features = None
    for seq, price in enumerate(prices):
        features = window.update(_tick(seq, float(price))) or features
    assert features.run_length == -1  # the streak broke on the last tick


def test_spread_bps_matches_definition():
    tick = _tick(0, 100.0, spread=1.0)
    assert math.isclose(tick.spread_bps, 1.0 / 100.0 * 1e4)


def test_bucket_boundaries_are_half_open():
    assert bucket(-99, MOVE_CUTS, MOVE) == MOVE[0]
    assert bucket(99, MOVE_CUTS, MOVE) == MOVE[-1]
    # A value exactly on a cut belongs to the upper bucket.
    assert bucket(MOVE_CUTS[0], MOVE_CUTS, MOVE) == MOVE[1]


def test_bucket_rejects_mismatched_labels():
    try:
        bucket(0.0, (1.0, 2.0), ("a", "b"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_bucket_is_monotone_over_its_vocabulary():
    seen = [bucket(v / 10, BOOK_CUTS, BOOK) for v in range(-10, 11)]
    indices = [BOOK.index(label) for label in seen]
    assert indices == sorted(indices)


def test_state_contains_no_numbers_only_vocabulary():
    """The whole point: Jev is shown words, never floats."""
    window = FeatureWindow()
    features = None
    for tick in SyntheticFeed(SyntheticConfig(n_ticks=200)):
        features = window.update(tick) or features

    state = build_state(features, position_units=0.05, max_units=0.1)
    allowed = {label for vocab in ALL_VOCABULARIES for label in vocab}

    leaves = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        else:
            leaves.append(node)

    walk(state)
    assert leaves, "state should not be empty"
    for leaf in leaves:
        assert isinstance(leaf, str), f"{leaf!r} is not a string"

    # Every bucketed field must come from a declared vocabulary. The free-text
    # fields are descriptive facts computed in code, not model judgements.
    bucketed = [
        state["price_action"]["last_5_snapshots"],
        state["order_book"]["resting_size"],
        state["our_book"]["exposure"],
    ]
    for label in bucketed:
        assert label in allowed


def test_state_reports_flat_book_honestly():
    window = FeatureWindow()
    features = None
    for tick in SyntheticFeed(SyntheticConfig(n_ticks=200)):
        features = window.update(tick) or features
    state = build_state(features, position_units=0.0, max_units=0.1)
    assert state["our_book"]["exposure"] == "flat, no position"
    assert state["our_book"]["open_position_pnl"] == "no open position"


def test_synthetic_feed_is_reproducible_and_alpha_is_switchable():
    a = list(SyntheticFeed(SyntheticConfig(n_ticks=50, seed=3)))
    b = list(SyntheticFeed(SyntheticConfig(n_ticks=50, seed=3)))
    assert [t.mid for t in a] == [t.mid for t in b]

    c = list(SyntheticFeed(SyntheticConfig(n_ticks=50, seed=3, alpha=0.0)))
    assert [t.mid for t in a] != [t.mid for t in c]


def test_tick_volatility_scales_with_the_snapshot_interval():
    fast = SyntheticConfig(interval_ms=250).tick_vol
    slow = SyntheticConfig(interval_ms=1000).tick_vol
    assert math.isclose(slow / fast, 2.0, rel_tol=1e-6)  # sqrt(4)
