# cbets — Compounding Bets, Method v2

Reference implementation of the Method v2 spec. It exists to answer exactly one
question before any money moves:

> **Does the model beat the de-vigged closing line?**

If the answer is no, the honest output is "stake nothing", and this code is
built to give you that answer quickly rather than slowly and expensively.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Prove the harness is honest (no network needed — synthetic ground truth)
python -m cbets.cli validate

# 2. Pull real data and see what you actually got
python -m cbets.cli fetch --divisions E0,E1,D1,SP1,I1,F1,N1 --from 2019 --to 2026

# 3. Run the gate
python -m cbets.cli backtest --divisions E0,D1,SP1,I1,F1,N1 \
    --from 2019 --to 2026 --eval-from 2021-08-01 --out predictions.csv

# 4. Score your pick log
python -m cbets.cli log-report --path picks.csv
```

`backtest` exits 0 if the gate passes, 2 if it does not. Wire that into anything
you like; a non-zero exit means do not stake.

**Run step 1 first, every time you change the model.** It takes about 30 seconds
and it is the only thing standing between you and a backtest that quietly lies.

---

## Why step 1 exists

Backtests lie, and they lie in ways you cannot detect from real-data results,
because with real data you don't know what the right answer was supposed to be.

`cbets/synth.py` builds a world where you do: teams with known true ratings,
matches simulated from them, and a bookmaker whose sharpness, bias and margin
you set. That supports three tests that actually mean something:

| Test | What it proves |
|---|---|
| Fit recovers true ratings | attack r=0.99, defence r=0.98, home advantage within 0.002 |
| **Sharp book → gate must FAIL** | if it ever passes, the harness is lying |
| **Planted bias → gate must PASS** | if it doesn't, the harness is blind |

Both failure directions are covered. A harness that only checks it can find
edges will happily find them where none exist.

### Two real bugs this caught

Neither was hypothetical. Both were in the first working version.

**1. Independent opening and closing lines.** The generator originally drew
opening and closing odds as independent perturbations around the truth. The
validation then reported **+15% CLV and +9% ROI against a book that was
supposed to be unbeatable** — because an accurate model can systematically pick
whichever independent draw happened to run long. Real closing lines are the
opening line *plus information*, not a fresh sample. They are now drawn as a
correlated bivariate normal.

**2. Raw CLV is inflated by the margin.** Comparing the price you took against
the bookmaker's *own* closing price flatters you by roughly the vig, because
that closing price still carries it. The sharp-book test showed **+3.18% raw
CLV alongside −2% ROI**. The honest measure is CLV against the **de-vigged**
close, which showed −3.49% — correctly reporting no edge. Both numbers are now
reported, and the fair one is the one to trust.

That second finding is a correction to the Method v2 spec, which called CLV the
metric without distinguishing the two.

### And one thing the spec got slightly wrong

The planted-bias test is profitable (+18.8% ROI, gate passes) yet shows
*negative* fair CLV. That is not a contradiction — it means the book's error
**persists to the close**. The line never moves to correct it, so there is no
line movement for CLV to capture.

So: **CLV is a proxy for edge only against an efficient closing line.** In a
market that stays wrong, you can be profitable with negative CLV; against a
sharp book, you can show positive raw CLV and still lose. Method v2 §2 presents
CLV as the metric; it is better described as the *fastest* signal, valid when
the close is efficient, and the log-loss gate is the primary test.

---

## The gate

```
model log loss   vs   de-vigged closing line log loss
```

Log loss is a **proper** scoring rule: it is minimised by reporting your honest
probabilities, so it cannot be gamed by shading picks toward flattering hit
rates. (`test_log_loss_is_minimised_by_honest_probabilities` checks this.)

The gate returns one of four verdicts, and three of them mean stop:

- `PASS` — beats the close, and the 95% CI excludes zero
- `NOT PROVEN` — beats the close, but the CI includes zero. That is noise.
- `FAIL` — worse than the close. Per Method v2 §8, stop; do **not** tune until
  it passes, because that is fitting the test.
- `INCONCLUSIVE` — fewer than 1,000 matches evaluated

---

## Modules

| File | What it does |
|---|---|
| `data.py` | Fetch, cache, and **normalise** football-data.co.uk across seasons. Column sets change over time — closing odds only exist from ~2019/20, xG only very recently, older files use `BbAv*` where newer use `Avg*`. Naive concatenation gives silent NaNs in exactly the columns the gate needs. `coverage()` tells you what you really have. |
| `devig.py` | Margin removal: multiplicative, power, and Shin (default). Getting this wrong biases every edge number in your favour. |
| `model.py` | Dixon–Coles. Ridge-penalised weighted Poisson with an analytic gradient for ratings, then a 1-D fit of the low-score correction. Time decay, xG as a quasi-Poisson response, full scoreline matrix → 1X2, totals, Asian handicap with correct quarter-line and push handling. |
| `backtest.py` | Strict walk-forward, log loss / Brier / calibration, bootstrap CIs, bet simulation, and the gate. |
| `picks.py` | The pick log, with the Method v2 gates enforced in code. It refuses to write a pick that fails one. |
| `synth.py` | Synthetic leagues with known ground truth. |

---

## The pick log

```python
from cbets.picks import Pick, append_pick, GateError

pick = Pick(
    kickoff="2026-09-19 20:00", league="E0", fixture="Arsenal v Brighton",
    market="1x2", selection="H",
    p_model=0.62,                       # written BEFORE the price is looked up
    odds_available=1.80,                # a real posted price, never an estimate
    market_odds_all=[1.80, 3.90, 4.60], # full market, so it can be de-vigged
    source_note="football-data.co.uk 2627/E0.csv pulled 2026-09-18",
)
append_pick(pick, "picks.csv")          # raises GateError if any gate fails
```

Enforced, with the reason given on rejection:

- **Gate 0** — `source_note` must name a source *and* carry a date. A length
  check is not a source check; "looks good" is ten characters.
- **Gate 1** — `p_model` must be a real probability
- **Gate 2** — `odds_available` must be a real posted price. `~1.70–1.85` is not a price.
- **Gate 3** — edge ≥ 4%
- **Gate 4** — the full market must be supplied so it can be de-vigged
- **Gate 5** — no two selections on one fixture
- **Market scope** — player props, corners and cards are refused with the reason

`report_log()` scores **calibration and CLV, not hit rate**, and prints the
Method v2 §8 kill criterion when it is met.

---

## Notes and limits

- **Closing odds only exist from roughly 2019/20.** Pulling 2010–2026 gives you
  far fewer usable matches than rows. Check `coverage()` — it reports this
  explicitly rather than letting you draw conclusions from NaNs.
- **xG is only in the most recent seasons.** `--response blend` (the default)
  uses xG where present and falls back to goals where not.
- **The default bet price is the opening number** (`--price o`). Betting the
  close is diagnostic only: if a strategy is only profitable at closing prices,
  it is not a strategy.
- **The honest prior is that the gate will fail on top-five-league 1X2.** Those
  are among the sharpest markets in football. A clean "no" is the most valuable
  output this project can produce, because it is the one result that stops you
  finding out slowly and expensively.
- **If the network refuses `football-data.co.uk`** (a locked-down proxy will),
  download the CSVs by hand into `~/.cache/cbets/<season>/<div>.csv` using the
  same layout. Everything else works unchanged.

---

## Tests

```bash
python -m pytest tests/ -q     # 30 tests
```

The one that matters most is
`test_walk_forward_predictions_are_all_out_of_sample`. A leaky backtest is
worse than no backtest, because it is confident.
