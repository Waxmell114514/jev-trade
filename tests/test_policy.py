"""Every gate in the policy, and the arithmetic around them."""

import pytest
from conftest import make_response

from jevtrade.features import FeatureWindow
from jevtrade.feed import SyntheticConfig, SyntheticFeed
from jevtrade.policy import PolicyConfig, decide


@pytest.fixture(scope="module")
def features():
    window = FeatureWindow()
    latest = None
    for tick in SyntheticFeed(SyntheticConfig(n_ticks=200)):
        latest = window.update(tick) or latest
    return latest


def target(features, position=0.0, config=None, **kwargs):
    config = config or PolicyConfig()
    return decide(features, make_response(**kwargs), position, config)


def test_a_clean_long_read_produces_a_long_target(features):
    decision = target(features, p_up=0.85, p_down=0.05)
    assert decision.target_units > 0
    assert decision.gate == ""


def test_a_clean_short_read_produces_a_short_target(features):
    decision = target(features, p_up=0.05, p_down=0.85)
    assert decision.target_units < 0


def test_target_is_capped_at_max_units(features):
    config = PolicyConfig(max_units=0.1)
    decision = target(features, config=config, p_up=1.0, p_down=0.0, setup=3.0)
    assert abs(decision.target_units) <= config.max_units + 1e-12


def test_disorderly_tape_flattens_everything(features):
    decision = target(features, position=0.1, p_up=0.95, p_down=0.01, disorderly=0.95)
    assert decision.target_units == 0.0
    assert decision.gate == "disorderly"


def test_cut_position_flattens_even_on_a_strong_read(features):
    decision = target(features, position=0.1, p_up=0.95, p_down=0.01, cut=0.9)
    assert decision.target_units == 0.0
    assert decision.gate == "cut_position"


def test_low_confidence_stands_aside(features):
    decision = target(features, p_up=0.6, p_down=0.35, confidence=0.05)
    assert decision.target_units == 0.0
    assert decision.gate == "low_confidence"


def test_weak_setup_stands_aside(features):
    decision = target(features, p_up=0.9, p_down=0.05, setup=0.2)
    assert decision.target_units == 0.0
    assert decision.gate == "weak_setup"


def test_no_edge_stands_aside(features):
    decision = target(features, p_up=0.45, p_down=0.44)
    assert decision.target_units == 0.0
    assert decision.gate == "no_edge"


def test_illiquid_blocks_increases_but_not_exits(features):
    """A thin book must never trap us in a position."""
    increase = target(features, position=0.02, p_up=0.95, p_down=0.01, liquidity=0.05)
    assert increase.gate == "illiquid"
    assert increase.target_units == pytest.approx(0.02)

    # The same thin book, but now the model wants out: the exit is allowed.
    exit_decision = target(
        features, position=0.1, p_up=0.95, p_down=0.01, liquidity=0.05, cut=0.9
    )
    assert exit_decision.target_units == 0.0


def test_reversal_reading_sizes_smaller_than_continuation(features):
    continuation = target(features, continuation=0.9, reversal=0.05)
    reversal = target(features, continuation=0.05, reversal=0.9)
    assert abs(reversal.target_units) < abs(continuation.target_units)


def test_higher_conviction_sizes_larger(features):
    weak = target(features, setup=1.2)
    strong = target(features, setup=3.0)
    assert abs(strong.target_units) > abs(weak.target_units)


def test_rebalance_deadband_suppresses_small_adjustments(features):
    config = PolicyConfig(max_units=0.1, rebalance_deadband=0.5)
    decision = decide(
        features, make_response(p_up=0.8, p_down=0.1), 0.06, config
    )
    # The ideal target moves only slightly, so the position is left alone.
    assert decision.target_units == pytest.approx(0.06)


def test_deadband_never_blocks_an_instruction_to_go_flat(features):
    config = PolicyConfig(max_units=0.1, rebalance_deadband=10.0)
    decision = decide(
        features, make_response(p_up=0.9, p_down=0.02, disorderly=0.99), 0.001, config
    )
    assert decision.target_units == 0.0


def test_decision_records_the_probabilities_for_the_audit_trail(features):
    decision = target(features, p_up=0.72, p_down=0.18)
    assert decision.p_up == pytest.approx(0.72)
    assert decision.p_down == pytest.approx(0.18)
    assert decision.edge == pytest.approx(0.54)
    assert decision.usage_tokens == 600
