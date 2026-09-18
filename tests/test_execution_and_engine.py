"""Fills, position accounting, and the latency behaviour of the loop."""

import pytest
from conftest import StubClient, make_response

from jevtrade.engine import EngineConfig, run
from jevtrade.execution import Broker, ExecutionConfig
from jevtrade.feed import SyntheticConfig, SyntheticFeed
from jevtrade.metrics import evaluate, max_drawdown
from jevtrade.policy import PolicyConfig
from jevtrade.types import Tick

FREE = ExecutionConfig(taker_fee_bps=0.0, impact_bps_at_full_book=0.0)


def tick(seq, price, spread=1.0, size=10.0):
    return Tick(
        seq=seq,
        ts=float(seq),
        symbol="X-USD",
        bid=price - spread / 2,
        ask=price + spread / 2,
        last=price,
        volume=1.0,
        bid_size=size,
        ask_size=size,
    )


# ------------------------------------------------------------------ execution


def test_buys_lift_the_ask_and_sells_hit_the_bid():
    broker = Broker(FREE)
    fill = broker.move_to(tick(0, 100.0), 1.0)
    assert fill.side == "buy"
    assert fill.price == pytest.approx(100.5)

    fill = broker.move_to(tick(1, 100.0), 0.0)
    assert fill.side == "sell"
    assert fill.price == pytest.approx(99.5)


def test_crossing_the_spread_costs_money_with_no_price_move():
    broker = Broker(FREE)
    broker.move_to(tick(0, 100.0), 1.0)
    broker.move_to(tick(1, 100.0), 0.0)
    assert broker.position.realized_pnl == pytest.approx(-1.0)  # the full spread


def test_round_trip_profit_is_exact():
    broker = Broker(FREE)
    broker.move_to(tick(0, 100.0), 1.0)  # buy at 100.5
    broker.move_to(tick(1, 110.0), 0.0)  # sell at 109.5
    assert broker.position.realized_pnl == pytest.approx(9.0)
    assert broker.position.units == 0.0
    assert broker.position.avg_price == 0.0


def test_fees_are_tracked_separately_from_trading_pnl():
    broker = Broker(ExecutionConfig(taker_fee_bps=10.0, impact_bps_at_full_book=0.0))
    broker.move_to(tick(0, 100.0), 1.0)
    expected = 100.5 * 1.0 * 10.0 / 1e4
    assert broker.position.fees_paid == pytest.approx(expected)
    assert broker.position.gross_pnl(100.0) != broker.position.net_pnl(100.0)
    assert broker.position.net_pnl(100.0) == pytest.approx(
        broker.position.gross_pnl(100.0) - expected
    )


def test_averaging_up_moves_the_average_price():
    broker = Broker(FREE)
    broker.move_to(tick(0, 100.0), 1.0)  # at 100.5
    broker.move_to(tick(1, 200.0), 2.0)  # +1 at 200.5
    assert broker.position.units == pytest.approx(2.0)
    assert broker.position.avg_price == pytest.approx(150.5)


def test_flipping_through_flat_realises_the_old_side():
    broker = Broker(FREE)
    broker.move_to(tick(0, 100.0), 1.0)  # long 1 @ 100.5
    broker.move_to(tick(1, 110.0), -1.0)  # sell 2 @ 109.5
    assert broker.position.units == pytest.approx(-1.0)
    assert broker.position.realized_pnl == pytest.approx(9.0)
    assert broker.position.avg_price == pytest.approx(109.5)


def test_larger_clips_pay_more_impact():
    config = ExecutionConfig(taker_fee_bps=0.0, impact_bps_at_full_book=10.0)
    small = Broker(config).fill_price(tick(0, 100.0, size=10.0), "buy", 1.0)
    large = Broker(config).fill_price(tick(0, 100.0, size=10.0), "buy", 5.0)
    assert large > small


def test_no_trade_when_already_at_target():
    broker = Broker(FREE)
    broker.move_to(tick(0, 100.0), 1.0)
    assert broker.move_to(tick(1, 100.0), 1.0) is None


# --------------------------------------------------------------------- engine


def ticks(n=400, **kwargs):
    return list(SyntheticFeed(SyntheticConfig(n_ticks=n, **kwargs)))


def test_decisions_are_filled_after_the_latency_not_at_the_decision_tick():
    """A decision made at t must never be filled at t. That would be look-ahead."""
    data = ticks()
    result = run(
        data,
        StubClient(latency_ms=100.0),
        execution=FREE,
        engine=EngineConfig(decide_every=8, deadline_ms=0),
        interval_ms=250,
    )
    assert result.decisions
    for decision, applied in zip(result.decisions, result.applied_seqs):
        assert applied > decision.seq


def test_bigger_latency_means_a_longer_delay():
    data = ticks()
    fast = run(data, StubClient(latency_ms=50.0), execution=FREE,
               engine=EngineConfig(deadline_ms=0), interval_ms=250)
    slow = run(data, StubClient(latency_ms=50.0), execution=FREE,
               engine=EngineConfig(deadline_ms=0, extra_latency_ms=2000.0),
               interval_ms=250)

    fast_delay = fast.applied_seqs[0] - fast.decisions[0].seq
    slow_delay = slow.applied_seqs[0] - slow.decisions[0].seq
    assert slow_delay > fast_delay


def test_decisions_past_the_deadline_are_dropped_not_acted_on():
    result = run(
        ticks(),
        StubClient(latency_ms=900.0),
        execution=FREE,
        engine=EngineConfig(deadline_ms=400.0),
        interval_ms=250,
    )
    assert result.expired > 0
    assert result.decisions == []
    assert result.fills == []


def test_a_deadline_of_zero_disables_the_check():
    result = run(
        ticks(),
        StubClient(latency_ms=900.0),
        execution=FREE,
        engine=EngineConfig(deadline_ms=0.0),
        interval_ms=250,
    )
    assert result.expired == 0
    assert result.decisions


def test_only_one_request_is_in_flight_so_slow_providers_get_fewer_looks():
    data = ticks(n=600)
    engine = EngineConfig(decide_every=2, deadline_ms=0, extra_latency_ms=5000.0)
    result = run(data, StubClient(latency_ms=0.0), execution=FREE,
                 engine=engine, interval_ms=250)
    assert result.skipped_busy > 0


def test_a_provider_failure_does_not_take_the_book_with_it():
    client = StubClient(latency_ms=10.0, fail_times=10_000)
    result = run(ticks(), client, execution=FREE,
                 engine=EngineConfig(deadline_ms=0), interval_ms=250)
    assert result.errors > 0
    assert result.decisions == []
    assert result.final_equity == 0.0


def test_the_stop_loss_runs_in_code_without_the_model():
    """A long into a collapsing tape must be stopped out locally."""
    data = [tick(seq, 100.0) for seq in range(80)]
    data += [tick(80 + seq, 100.0 - seq * 0.5) for seq in range(40)]

    response = make_response(p_up=0.99, p_down=0.005, setup=3.0, latency_ms=10.0)
    result = run(
        data,
        StubClient(response=response),
        policy=PolicyConfig(max_units=1.0),
        execution=FREE,
        engine=EngineConfig(decide_every=8, deadline_ms=0, stop_loss_bps=20.0,
                            warmup_ticks=60, window_size=120),
        interval_ms=250,
    )
    assert result.stops > 0


def test_the_kill_switch_halts_trading_for_good():
    data = [tick(seq, 100.0) for seq in range(80)]
    data += [tick(80 + seq, 100.0 - seq * 2.0) for seq in range(60)]

    response = make_response(p_up=0.99, p_down=0.005, setup=3.0, latency_ms=10.0)
    result = run(
        data,
        StubClient(response=response),
        policy=PolicyConfig(max_units=1.0),
        execution=FREE,
        engine=EngineConfig(decide_every=8, deadline_ms=0, stop_loss_bps=None,
                            max_drawdown=5.0, warmup_ticks=60, window_size=120),
        interval_ms=250,
    )
    assert result.halted
    assert result.equity_curve[-1] == result.equity_curve[-1]  # not NaN


def test_no_decisions_are_made_during_warmup():
    result = run(
        ticks(n=100),
        StubClient(latency_ms=10.0),
        execution=FREE,
        engine=EngineConfig(decide_every=1, deadline_ms=0, warmup_ticks=90),
        interval_ms=250,
    )
    assert all(d.seq >= 89 for d in result.decisions)


# -------------------------------------------------------------------- metrics


def test_max_drawdown_measures_peak_to_trough():
    assert max_drawdown([0.0, 10.0, 4.0, 12.0, 2.0]) == pytest.approx(10.0)
    assert max_drawdown([0.0, 1.0, 2.0]) == pytest.approx(0.0)


def test_break_even_fee_is_where_gross_equals_fees():
    result = run(
        ticks(n=1500),
        StubClient(latency_ms=50.0),
        execution=ExecutionConfig(taker_fee_bps=0.0),
        engine=EngineConfig(deadline_ms=0),
        interval_ms=250,
    )
    metrics = evaluate(result)
    if result.turnover > 0:
        implied = metrics.break_even_fee_bps * result.turnover / 1e4
        assert implied == pytest.approx(metrics.gross_pnl, rel=1e-6)


def test_calibration_buckets_stay_within_their_range():
    result = run(
        ticks(n=2000),
        StubClient(latency_ms=50.0),
        execution=FREE,
        engine=EngineConfig(deadline_ms=0),
        interval_ms=250,
    )
    for bucket in evaluate(result).calibration:
        assert bucket.low <= bucket.predicted <= bucket.high
        assert 0.0 <= bucket.realized <= 1.0
        assert bucket.n > 0


def test_metrics_survive_an_empty_run():
    result = run([], StubClient(), execution=FREE, interval_ms=250)
    metrics = evaluate(result)
    assert metrics.net_pnl == 0.0
    assert metrics.calibration == []
