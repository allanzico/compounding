"""Tests. The ones that matter are the look-ahead and gate-honesty tests."""
import numpy as np
import pandas as pd
import pytest

from cbets.devig import devig, overround
from cbets.model import fit, score_matrix, probs_1x2, prob_over, ah_home
from cbets.backtest import walk_forward, evaluate, simulate_bets, bet_metrics, gate, log_loss
from cbets.picks import Pick, check_gates, kelly_stake, GateError, append_pick
from cbets.synth import make_league


# ------------------------------------------------------------------ devig

@pytest.mark.parametrize("method", ["multiplicative", "power", "shin"])
def test_devig_sums_to_one(method):
    p = devig([2.10, 3.50, 3.60], method)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p > 0).all()


def test_devig_recovers_proportional_margin_exactly():
    """If the book applied margin proportionally, multiplicative de-vig must invert it."""
    true = np.array([0.55, 0.25, 0.20])
    odds = 1.0 / (true * 1.06)
    assert abs(overround(odds) - 0.06) < 1e-9
    assert np.allclose(devig(odds, "multiplicative"), true, atol=1e-12)


def test_shin_shifts_probability_toward_favourite():
    """Shin assumes longshots carry more margin, so removing it favours the favourite."""
    odds = [1.50, 4.50, 7.00]
    mult = devig(odds, "multiplicative")
    shin = devig(odds, "shin")
    assert shin[0] > mult[0]
    assert shin[-1] < mult[-1]


def test_devig_rejects_impossible_odds():
    with pytest.raises(ValueError):
        devig([1.0, 3.0, 4.0])


# ------------------------------------------------------------------ model

def test_score_matrix_is_a_distribution():
    m = score_matrix(1.7, 1.2, -0.06, 12)
    assert abs(m.sum() - 1.0) < 1e-10
    assert (m >= 0).all()


def test_1x2_partitions_the_matrix():
    m = score_matrix(1.4, 1.1, -0.04)
    assert abs(probs_1x2(m).sum() - 1.0) < 1e-10


def test_asian_handicap_identities():
    """AH 0.0 is draw-no-bet; AH -0.5 is the straight win; AH +0.5 is win-or-draw."""
    m = score_matrix(1.6, 1.1, -0.05)
    ph, pd_, pa = probs_1x2(m)
    w0, push0, _ = ah_home(m, 0.0)
    assert abs(w0 - ph) < 1e-9 and abs(push0 - pd_) < 1e-9
    assert abs(ah_home(m, -0.5)[0] - ph) < 1e-9
    assert abs(ah_home(m, 0.5)[0] - (ph + pd_)) < 1e-9


def test_quarter_line_is_the_average_of_its_neighbours():
    m = score_matrix(1.5, 1.3, -0.05)
    a = np.array(ah_home(m, -0.5))
    b = np.array(ah_home(m, 0.0))
    assert np.allclose(np.array(ah_home(m, -0.25)), (a + b) / 2, atol=1e-12)


def test_ah_outcomes_sum_to_one():
    m = score_matrix(1.9, 0.9, -0.05)
    for line in (-2.0, -1.25, -0.75, 0.0, 0.25, 1.5):
        assert abs(sum(ah_home(m, line)) - 1.0) < 1e-9


def test_fit_recovers_true_ratings():
    L = make_league(n_teams=18, seasons=5, seed=3)
    m = fit(L.matches, response="blend", half_life_days=500)
    j = L.truth_table().set_index("team").join(m.team_table().set_index("team"))
    assert np.corrcoef(j.true_attack, j.attack)[0, 1] > 0.90
    assert np.corrcoef(j.true_defence, j.defence)[0, 1] > 0.85
    assert abs(m.home_adv - L.true_home_adv) < 0.08


def test_unknown_team_falls_back_to_league_average():
    L = make_league(n_teams=14, seasons=2, seed=5)
    m = fit(L.matches)
    lam_known, _ = m.rates(L.teams[0], L.teams[1])
    lam_new, _ = m.rates("Newly Promoted FC", L.teams[1])
    assert np.isfinite(lam_new) and lam_new > 0
    assert lam_new != lam_known


# -------------------------------------------------------------- no look-ahead

def test_fit_never_sees_the_match_it_predicts():
    L = make_league(n_teams=14, seasons=3, seed=9)
    df = L.matches
    cutoff = df["date"].quantile(0.6)
    m = fit(df, as_of=cutoff)
    assert m.trained_through < cutoff


def test_walk_forward_predictions_are_all_out_of_sample():
    """The single most important test here: a leaky backtest is worse than none."""
    L = make_league(n_teams=14, seasons=4, seed=11)
    df = L.matches
    start = df["date"].min() + pd.Timedelta(days=380)
    pred = walk_forward(df, start_date=start, refit_days=21, min_train_matches=150,
                        verbose=False)
    assert not pred.empty
    assert pred["date"].min() >= start
    # every prediction must have been made from strictly fewer matches than exist before it
    for r in pred.sample(min(40, len(pred)), random_state=0).itertuples():
        available = (df["date"] < r.date).sum()
        assert r.n_train <= available


# ------------------------------------------------------------------- gate

def test_gate_reports_no_edge_against_a_sharp_book():
    """If this ever passes, the harness is lying and nothing downstream can be trusted."""
    L = make_league(n_teams=20, seasons=7, seed=7)
    df = L.matches
    pred = walk_forward(df, start_date=df["date"].min() + pd.Timedelta(days=380),
                        refit_days=21, min_train_matches=300, verbose=False,
                        half_life_days=400)
    met = evaluate(pred)
    passed, reason = gate(met, min_matches=800)
    assert not passed, f"found an edge in a sharp market: {reason}"


def test_gate_finds_a_planted_edge():
    L = make_league(n_teams=20, seasons=7, seed=7, bias_draw=-0.05)
    df = L.matches
    pred = walk_forward(df, start_date=df["date"].min() + pd.Timedelta(days=380),
                        refit_days=21, min_train_matches=300, verbose=False,
                        half_life_days=400)
    met = evaluate(pred)
    passed, reason = gate(met, min_matches=800)
    assert passed, f"missed a planted edge: {reason}"
    bets = simulate_bets(pred, edge_threshold=0.04, price="o")
    stats = bet_metrics(bets)
    assert stats["roi"] > 0
    # the planted bias was on draws, so that is where the bets should land
    assert (bets["selection"] == "D").mean() > 0.5


def test_gate_refuses_to_conclude_on_a_small_sample():
    met = {"n": 120, "logloss_model": 0.9, "logloss_closing": 1.0,
           "logloss_delta": 0.1, "logloss_delta_ci": (0.05, 0.15)}
    passed, reason = gate(met, min_matches=1000)
    assert not passed and "INCONCLUSIVE" in reason


def test_gate_refuses_when_the_interval_includes_zero():
    met = {"n": 2000, "logloss_model": 0.99, "logloss_closing": 1.00,
           "logloss_delta": 0.01, "logloss_delta_ci": (-0.002, 0.022)}
    passed, reason = gate(met, min_matches=1000)
    assert not passed and "NOT PROVEN" in reason


def test_log_loss_is_minimised_by_honest_probabilities():
    """Log loss is a proper scoring rule — shading your numbers must cost you."""
    rng = np.random.default_rng(0)
    truth = np.array([0.5, 0.25, 0.25])
    y = rng.choice(3, size=40000, p=truth)
    honest = np.tile(truth, (len(y), 1))
    shaded = np.tile([0.7, 0.15, 0.15], (len(y), 1))
    assert log_loss(honest, y) < log_loss(shaded, y)


# ------------------------------------------------------------------ picks

def _ok(**kw):
    base = dict(kickoff="2026-09-19 20:00", league="E0", fixture="Arsenal v Brighton",
                market="1x2", selection="H", p_model=0.62, odds_available=1.80,
                market_odds_all=[1.80, 3.90, 4.60],
                source_note="football-data.co.uk 2627/E0.csv pulled 2026-09-18")
    base.update(kw)
    return Pick(**base)


def test_a_complete_pick_passes():
    assert check_gates(_ok()) == []


def test_thin_edge_is_rejected():
    fails = check_gates(_ok(p_model=0.55))
    assert any("Gate 3" in f for f in fails)


def test_estimated_price_is_rejected():
    fails = check_gates(_ok(odds_available=float("nan")))
    assert any("Gate 2" in f for f in fails)


def test_missing_source_is_rejected():
    assert any("Gate 0" in f for f in check_gates(_ok(source_note="looks good")))


def test_player_props_are_banned():
    assert any("banned" in f for f in check_gates(_ok(market="player_prop")))


def test_two_selections_on_one_fixture_are_rejected():
    first = _ok()
    second = _ok(market="over_under", selection="over", p_model=0.60,
                 odds_available=1.85, market_odds_all=[1.85, 1.95])
    assert any("Gate 5" in f for f in check_gates(second, existing_slate=[first]))


def test_append_refuses_a_failing_pick(tmp_path):
    p = tmp_path / "picks.csv"
    with pytest.raises(GateError):
        append_pick(_ok(market="corners"), p)
    assert not p.exists()


def test_append_writes_a_good_pick(tmp_path):
    p = tmp_path / "picks.csv"
    append_pick(_ok(), p)
    df = pd.read_csv(p)
    assert len(df) == 1
    assert abs(df.loc[0, "edge_pct"] - 11.6) < 0.01


def test_kelly_is_capped():
    assert kelly_stake(0.95, 5.0, 1000, fraction=0.25, cap=0.02) == pytest.approx(20.0)


def test_kelly_is_zero_without_an_edge():
    assert kelly_stake(0.40, 2.0, 1000) == 0.0
