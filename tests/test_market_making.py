"""The market-making simulator, its arms, and the markout accounting."""

import pytest

from jevtrade.mm import questions as Q
from jevtrade.mm.engine import run as mm_run
from jevtrade.mm.events import EventGenerator, keyword_alarm
from jevtrade.mm.market import MarketConfig, MarketSim, Quote
from jevtrade.mm.metrics import HORIZONS, evaluate
from jevtrade.mm.mock import MockHeadlineClient
from jevtrade.mm.quoter import Posture, Quoter, QuoterConfig
from jevtrade.mm.strategies import (
    JevConfig,
    JevStrategy,
    KeywordStrategy,
    NaiveStrategy,

)
from jevtrade.types import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer


def quote_flat(sim, half_bps=2.0, ticks=None):
    """Run a sim quoting a fixed spread; returns the fills."""
    fills = []
    while (obs := sim.observe()) is not None:
        if ticks is not None and obs.seq >= ticks:
            break
        half = obs.mid * half_bps / 1e4
        fills += sim.advance(Quote(obs.mid - half, obs.mid + half, 1.0, 1.0))
    return fills


# ------------------------------------------------------------------- events


def test_only_material_events_move_the_price():
    gen = EventGenerator(seed=1)
    seen = {}
    for seq in range(20_000):
        event = gen.maybe(seq)
        if event:
            seen.setdefault(event.kind, []).append(event.impact_bps)

    for kind in ("noise", "priced_in"):
        assert all(v == 0.0 for v in seen[kind]), f"{kind} must not move the tape"
    assert all(v > 0 for v in seen["material_up"])
    assert all(v < 0 for v in seen["material_down"])


def test_keyword_rule_is_a_fair_competitor():
    """It must catch every obvious event, or beating it proves nothing."""
    audit = EventGenerator().audit()
    assert audit["hits"] == 6, audit
    assert audit["misses"] == 4, audit  # all four are the deliberately subtle ones
    assert audit["false_alarms"] == 7  # denials and re-reports it cannot read


def test_keyword_rule_cannot_read_a_denial():
    assert keyword_alarm("Major exchange halts all BTC withdrawals citing wallet issue")
    assert keyword_alarm("Exchange denies reports that BTC withdrawals have been halted")


def test_word_boundaries_so_rates_does_not_match_reiterates():
    assert not keyword_alarm("Analyst reiterates long-term constructive view")


# ------------------------------------------------------------------- market


def test_price_path_is_independent_of_our_quoting():
    """Otherwise the arms are not comparable."""
    paths = []
    for half in (2.0, 12.0):
        sim = MarketSim(MarketConfig(n_ticks=900))
        quote_flat(sim, half)
        paths.append(list(sim.mids))
    assert paths[0] == paths[1]


def test_no_informed_flow_when_toxicity_is_zero():
    sim = MarketSim(MarketConfig(n_ticks=3000, toxicity=0.0))
    assert not any(f.informed for f in quote_flat(sim))


def test_informed_flow_exists_otherwise():
    sim = MarketSim(MarketConfig(n_ticks=3000, toxicity=1.0))
    assert any(f.informed for f in quote_flat(sim))


def test_informed_fills_only_happen_near_an_event():
    cfg = MarketConfig(n_ticks=4000)
    sim = MarketSim(cfg)
    fills = quote_flat(sim)
    events = [e.seq for e in [sim.events.maybe(-1)] if e]  # generator is spent
    # Reconstruct event ticks by re-running the same event stream.
    gen = EventGenerator(seed=cfg.seed + 101)
    events = [s for s in range(cfg.n_ticks) if (e := gen.maybe(s)) and e.impact_bps]
    for fill in fills:
        if fill.informed:
            assert any(0 <= fill.seq - e <= cfg.informed_window for e in events)


def test_quoting_wider_earns_fewer_fills():
    tight = len(quote_flat(MarketSim(MarketConfig(n_ticks=2500)), 1.0))
    wide = len(quote_flat(MarketSim(MarketConfig(n_ticks=2500)), 12.0))
    assert wide < tight


def test_fill_edge_is_positive_at_the_moment_of_the_fill():
    """We always cross our own spread in our favour when we are hit."""
    for fill in quote_flat(MarketSim(MarketConfig(n_ticks=1500))):
        assert fill.edge_bps > 0


# ------------------------------------------------------------------- quoter


def observation(mid=77_000.0, seq=0, imbalance=0.0, event=None):
    from jevtrade.mm.market import Observation

    return Observation(seq=seq, mid=mid, vol_bps=1.0, event=event,
                       recent_flow_imbalance=imbalance, ticks_since_event=None)


def test_long_inventory_pushes_quotes_down():
    q = Quoter()
    flat = q.quote(observation(), 0.0, Posture.normal())
    long = q.quote(observation(), 5.0, Posture.normal())
    assert long.bid < flat.bid and long.ask < flat.ask


def test_posture_applies_per_side():
    q = Quoter()
    flat = q.quote(observation(), 0.0, Posture.normal())
    skewed = q.quote(observation(), 0.0,
                     Posture(ask_spread_mult=5.0, ask_size_mult=0.0, ttl=5))
    assert skewed.ask > flat.ask
    assert skewed.ask_size == 0.0
    assert skewed.bid == pytest.approx(flat.bid)  # the other side is untouched
    assert skewed.bid_size == flat.bid_size


def test_inventory_cap_stops_us_adding_to_the_position():
    q = Quoter(QuoterConfig(max_inventory=3.0))
    assert q.quote(observation(), 3.0, Posture.normal()).bid_size == 0.0
    assert q.quote(observation(), -3.0, Posture.normal()).ask_size == 0.0


def test_posture_expires_after_its_ttl():
    p = Posture(bid_spread_mult=5.0, ttl=2)
    assert p.defensive
    assert p.tick().defensive
    assert not p.tick().tick().defensive


# --------------------------------------------------------------- strategies


def mm_response(*, moves=0.9, severity=2.7, fresh=0.9, denied=0.0,
                up=0.8, down=0.05, informed=0.1, confidence=0.8):
    levels = len(Q.SEVERITY_LEVELS)
    return JevResponse(
        model="test",
        answers={
            Q.MOVES_PRICE: NoulAnswer(noul=moves),
            Q.NEW_INFORMATION: NoulAnswer(noul=fresh),
            Q.INFORMED_FLOW: NoulAnswer(noul=informed),
            Q.SEVERITY: ScoreAnswer(
                score=severity,
                legend={str(i): s for i, s in enumerate(Q.SEVERITY_LEVELS)},
                probabilities={str(i): 1 / levels for i in range(levels)},
                confidence=0.6,
            ),
            Q.DIRECTION: ChoiceAnswer(
                choice=Q.UP if up > down else Q.DOWN,
                probabilities={Q.UP: up, Q.DOWN: down,
                               Q.UNCLEAR: max(0.0, 1 - up - down)},
                confidence=confidence,
            ),
            Q.REPORT_TYPE: ChoiceAnswer(
                choice=Q.DENIED if denied > 0.5 else Q.HAPPENED,
                probabilities={Q.HAPPENED: 1 - denied, Q.DENIED: denied,
                               Q.NEITHER: 0.0},
                confidence=0.7,
            ),
        },
        input_tokens=300, output_tokens=0, latency_ms=380.0, provider="test",
    )


class FixedClient:
    provider = "test"
    model = "fixed"

    def __init__(self, response):
        self.response = response
        self.questions = None

    def evaluate(self, state):
        return self.response


def event(headline="Major exchange halts all BTC withdrawals citing wallet issue",
          kind="material_down"):
    from jevtrade.mm.events import NewsEvent

    return NewsEvent(seq=10, headline=headline, kind=kind, subtle=False,
                     alarming=True, impact_bps=-20.0)


def posture_for(response, obs=None):
    strategy = JevStrategy(FixedClient(response), JevConfig(risk_floor=0.12))
    update = strategy.evaluate(obs or observation(seq=10, event=event()), 0.0)
    assert update is not None
    return update.posture, update.detail


def test_material_news_pulls_the_exposed_side_only():
    """Bullish news means our offer gets lifted, so the offer goes, not the bid."""
    posture, detail = posture_for(mm_response(up=0.85, down=0.03))
    assert detail["stance"] == "pull offer"
    assert posture.ask_spread_mult > posture.bid_spread_mult
    assert posture.bid_spread_mult == pytest.approx(1.0)  # still quoting the bid


def test_bearish_news_pulls_the_bid():
    posture, detail = posture_for(mm_response(up=0.03, down=0.85))
    assert detail["stance"] == "pull bid"
    assert posture.bid_spread_mult > posture.ask_spread_mult


def test_a_denial_is_not_an_event():
    """The single judgment a keyword rule cannot make."""
    _, alarming = posture_for(mm_response(denied=0.0))
    _, denial = posture_for(mm_response(denied=0.98))
    assert alarming["stance"] != "normal"
    assert denial["stance"] == "normal"
    assert denial["risk"] < alarming["risk"]


def test_a_re_report_is_discounted():
    _, fresh = posture_for(mm_response(fresh=0.95))
    _, stale = posture_for(mm_response(fresh=0.05))
    assert stale["risk"] < fresh["risk"]


def test_unclear_direction_widens_both_sides_rather_than_standing_aside():
    """Uncertainty has a safe direction here: quote wider, keep earning."""
    posture, detail = posture_for(mm_response(up=0.4, down=0.4, confidence=0.1))
    assert detail["stance"] == "widen both"
    assert posture.bid_spread_mult > 1.0 and posture.ask_spread_mult > 1.0
    assert posture.bid_size_mult > 0 and posture.ask_size_mult > 0


def test_noise_leaves_the_quotes_alone():
    _, detail = posture_for(mm_response(moves=0.05, severity=0.2))
    assert detail["stance"] == "normal"


def test_tape_anomaly_takes_its_direction_from_the_flow_not_the_headline():
    obs = observation(seq=500, imbalance=0.9, event=None)
    strategy = JevStrategy(
        FixedClient(mm_response(moves=0.1, severity=0.2, informed=0.9,
                                up=0.3, down=0.3, confidence=0.05)),
        JevConfig(risk_floor=0.12),
    )
    update = strategy.evaluate(obs, 0.0)
    assert update is not None
    # People lifting our offer means price is rising: the offer is exposed.
    assert update.detail["direction"] == "up"
    assert update.detail["direction_from"] == "tape"
    assert update.posture.ask_spread_mult > update.posture.bid_spread_mult


def test_a_provider_failure_fails_safe_not_open():
    class Broken:
        provider, model, questions = "broken", "x", None

        def evaluate(self, state):
            from jevtrade.jev.client import JevError

            raise JevError("down")

    strategy = JevStrategy(Broken())
    update = strategy.evaluate(observation(seq=10, event=event()), 0.0)
    assert update is not None and update.posture.defensive
    assert strategy.errors == 1


def test_keyword_arm_only_fires_on_alarming_headlines():
    arm = KeywordStrategy()
    assert arm.evaluate(observation(seq=1, event=event()), 0.0) is not None
    quiet = event("Conference panel debates the future of on-chain settlement", "noise")
    assert arm.evaluate(observation(seq=2, event=quiet), 0.0) is None


# ------------------------------------------------------------ engine/metrics


def test_a_slow_posture_lands_late_and_never_on_the_same_tick():
    cfg = MarketConfig(n_ticks=600)
    strategy = JevStrategy(FixedClient(mm_response()), JevConfig(risk_floor=0.12))
    result = mm_run(strategy, market=cfg)
    assert result.posture_log
    # 380 ms of latency against 250 ms ticks is two ticks of delay.
    assert all(entry["delay_ticks"] >= 2 for entry in result.posture_log)


def test_the_model_is_told_the_instrument_that_is_being_simulated():
    strategy = JevStrategy(FixedClient(mm_response()))
    mm_run(strategy, market=MarketConfig(n_ticks=300, symbol="SOL-USD spot"))
    assert strategy.config.instrument == "SOL-USD spot"


def test_markout_decomposes_into_capture_plus_drift():
    result = mm_run(NaiveStrategy(), market=MarketConfig(n_ticks=2500))
    m = evaluate(result)
    assert m.all.fills > 0
    assert m.all.capture > 0  # we always earn the half-spread at the fill
    assert m.informed.fills + m.benign.fills == m.all.fills
    for h in HORIZONS:
        assert m.all.adverse[h] == pytest.approx(
            m.informed.adverse[h] + m.benign.adverse[h]
        )


def test_informed_flow_is_what_costs_money():
    m = evaluate(mm_run(NaiveStrategy(), market=MarketConfig(n_ticks=4000)))
    assert m.informed.adverse[20] < 0
    assert m.informed.adverse[20] < m.benign.adverse[20]


def test_defending_cannot_help_when_nobody_is_informed():
    """The falsifiability check, run offline so it is always enforced."""
    cfg = dict(n_ticks=4000, toxicity=0.0, event_impact_bps=0.0)
    naive = evaluate(mm_run(NaiveStrategy(), market=MarketConfig(**cfg))).final_equity
    keyword = evaluate(mm_run(KeywordStrategy(), market=MarketConfig(**cfg))).final_equity
    jev = evaluate(
        mm_run(JevStrategy(MockHeadlineClient(), JevConfig(risk_floor=0.12)),
               market=MarketConfig(**cfg))
    ).final_equity
    assert keyword <= naive
    assert jev <= naive
