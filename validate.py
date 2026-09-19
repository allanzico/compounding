"""End-to-end validation against synthetic worlds where the truth is known."""
import numpy as np, pandas as pd
from cbets.synth import make_league
from cbets.model import fit
from cbets.backtest import walk_forward, evaluate, simulate_bets, bet_metrics, gate, BacktestResult, report
from cbets.devig import devig

pd.set_option("display.width", 200)

print("#" * 72)
print("TEST 1 — does the fit recover the true team ratings?")
print("#" * 72)
L = make_league(n_teams=20, seasons=6, seed=42)
m = fit(L.matches, response="blend", half_life_days=400, ridge=0.02)
truth = L.truth_table().set_index("team")
est = m.team_table().set_index("team")
j = truth.join(est)
print(f"  attack  corr(true, fitted) = {np.corrcoef(j.true_attack, j.attack)[0,1]:.4f}")
print(f"  defence corr(true, fitted) = {np.corrcoef(j.true_defence, j.defence)[0,1]:.4f}")
print(f"  home advantage  true {L.true_home_adv:.4f}  fitted {m.home_adv:.4f}")
print(f"  intercept       true {L.true_intercept:.4f}  fitted {m.intercept:.4f}")
print(f"  rho fitted {m.rho:+.4f} (generator used -0.05)")

print()
print("#" * 72)
print("TEST 2 — SHARP book: the gate must report NO edge")
print("#" * 72)
L2 = make_league(n_teams=20, seasons=8, seed=7)
d2 = L2.matches
start = d2.date.min() + pd.Timedelta(days=380)
pred = walk_forward(d2, start_date=start, refit_days=14, min_train_matches=300,
                    response="blend", half_life_days=400, ridge=0.02)
met = evaluate(pred)
bets = simulate_bets(pred, edge_threshold=0.04, price="o")
res = BacktestResult(pred, met, bets, bet_metrics(bets))
print(report(res))

print()
print("#" * 72)
print("TEST 3 — book with a PLANTED 5-point draw underestimate: must FIND it")
print("#" * 72)
L3 = make_league(n_teams=20, seasons=8, seed=7, bias_draw=-0.05)
d3 = L3.matches
pred3 = walk_forward(d3, start_date=d3.date.min() + pd.Timedelta(days=380), refit_days=14,
                     min_train_matches=300, response="blend", half_life_days=400, ridge=0.02)
met3 = evaluate(pred3)
bets3 = simulate_bets(pred3, edge_threshold=0.04, price="o")
res3 = BacktestResult(pred3, met3, bets3, bet_metrics(bets3))
print(report(res3))
if not bets3.empty:
    print("\n  bets by selection:")
    print(bets3.groupby("selection").agg(n=("pnl","size"), roi=("pnl","mean")).to_string())
