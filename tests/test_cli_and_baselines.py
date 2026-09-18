"""End-to-end: the CLI runs, and the baselines are a fair comparison."""

import json

import pytest
from conftest import StubClient

from jevtrade.baselines import buy_and_hold, flat, imbalance, registry, run_rule
from jevtrade.cli import main
from jevtrade.engine import EngineConfig, run
from jevtrade.execution import ExecutionConfig
from jevtrade.feed import CsvReplayFeed, SyntheticConfig, SyntheticFeed, write_csv
from jevtrade.metrics import evaluate
from jevtrade.policy import PolicyConfig

FREE = ExecutionConfig(taker_fee_bps=0.0, impact_bps_at_full_book=0.0)


@pytest.fixture(scope="module")
def ticks():
    return list(SyntheticFeed(SyntheticConfig(n_ticks=1200)))


def test_flat_baseline_never_trades(ticks):
    result = run_rule(ticks, flat, name="flat", execution=FREE, interval_ms=250)
    assert result.fills == []
    assert result.final_equity == 0.0


def test_buy_and_hold_tracks_the_price(ticks):
    result = run_rule(
        ticks,
        buy_and_hold,
        name="buy_and_hold",
        policy=PolicyConfig(max_units=1.0),
        execution=FREE,
        interval_ms=250,
    )
    move = ticks[-1].mid - ticks[0].mid
    assert result.final_equity == pytest.approx(move, rel=0.05)


def test_baselines_run_and_are_all_registered(ticks):
    for name, signal in registry().items():
        result = run_rule(ticks, signal, name=name, execution=FREE, interval_ms=250)
        assert result.ticks == len(ticks)
        assert result.model == name


def test_baselines_also_execute_one_tick_late(ticks):
    """No baseline may peek at the price it trades on."""
    result = run_rule(ticks, imbalance, name="imbalance", execution=FREE,
                      interval_ms=250)
    for fill in result.fills:
        assert fill.seq >= 1


def test_alpha_zero_leaves_no_edge_for_anyone(ticks):
    """The honesty check on the simulator itself.

    With the predictable component switched off, a strategy reading the book
    should have no systematic edge. Averaged over several paths its gross P&L
    should sit near zero rather than reliably positive.
    """
    totals = []
    for seed in range(6):
        data = list(
            SyntheticFeed(SyntheticConfig(n_ticks=2500, seed=seed, alpha=0.0))
        )
        result = run_rule(
            data, imbalance, name="imbalance", execution=FREE, interval_ms=250
        )
        capital = PolicyConfig().max_units * data[0].mid
        totals.append(result.gross_pnl / capital)

    mean = sum(totals) / len(totals)
    assert abs(mean) < 0.02, f"expected no edge without alpha, got {mean:.4f}"


def test_alpha_creates_an_edge_the_book_can_see():
    """And the mirror image: with alpha on, the book-reading rule earns."""
    totals = []
    for seed in range(6):
        data = list(
            SyntheticFeed(SyntheticConfig(n_ticks=2500, seed=seed, alpha=0.6))
        )
        result = run_rule(
            data, imbalance, name="imbalance", execution=FREE, interval_ms=250
        )
        capital = PolicyConfig().max_units * data[0].mid
        totals.append(result.gross_pnl / capital)

    assert sum(totals) / len(totals) > 0


def test_csv_round_trip(tmp_path, ticks):
    path = tmp_path / "ticks.csv"
    written = write_csv(path, ticks)
    assert written == len(ticks)

    replayed = list(CsvReplayFeed(path))
    assert len(replayed) == len(ticks)
    assert replayed[0].symbol == ticks[0].symbol
    assert replayed[5].mid == pytest.approx(ticks[5].mid, rel=1e-9)


def test_csv_replay_respects_a_limit(tmp_path, ticks):
    path = tmp_path / "ticks.csv"
    write_csv(path, ticks)
    assert len(list(CsvReplayFeed(path, limit=17))) == 17


def test_cli_backtest_runs_offline(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["backtest", "--ticks", "800", "--provider", "mock"]) == 0
    out = capsys.readouterr().out
    assert "provider=mock" in out  # the warning must always be shown
    assert "break-even fee" in out


def test_cli_backtest_json_is_machine_readable(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["backtest", "--ticks", "800", "--provider", "mock", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "mock"
    assert "net_pnl" in payload and "break_even_fee_bps" in payload


def test_cli_decide_shows_the_whole_exchange(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["decide", "--ticks", "300", "--provider", "mock"]) == 0
    out = capsys.readouterr().out
    assert "state sent to Jev" in out
    assert "direction" in out and "cut_position" in out


def test_cli_sweep_reports_the_latency_curve(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(
        ["sweep", "--ticks", "700", "--provider", "mock",
         "--latencies", "0,2000", "--seeds", "2"]
    ) == 0
    out = capsys.readouterr().out
    assert "executed" in out and "2000 ms" in out


def test_cli_backtest_with_baselines(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(
        ["backtest", "--ticks", "700", "--provider", "mock", "--baselines"]
    ) == 0
    out = capsys.readouterr().out
    for name in registry():
        assert name in out


def test_engine_and_baselines_agree_on_a_flat_strategy(ticks):
    """Same loop, same costs: a never-trading model matches the flat rule."""
    from conftest import make_response

    never = StubClient(response=make_response(p_up=0.34, p_down=0.33, setup=0.0))
    engine_result = run(
        ticks, never, execution=FREE,
        engine=EngineConfig(deadline_ms=0), interval_ms=250,
    )
    rule_result = run_rule(ticks, flat, name="flat", execution=FREE, interval_ms=250)
    assert evaluate(engine_result).net_pnl == pytest.approx(
        evaluate(rule_result).net_pnl
    )


def test_cli_csv_source_reads_the_whole_file_by_default(tmp_path, capsys, monkeypatch, ticks):
    """--ticks defaults to 0, which means 'all of them', not 'none of them'."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    path = tmp_path / "ticks.csv"
    write_csv(path, ticks)
    assert main(["backtest", "--source", "csv", "--csv", str(path),
                 "--provider", "mock"]) == 0
    assert f"{len(ticks):,}" in capsys.readouterr().out
