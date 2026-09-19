# Compounding Bets — Method v2

Replaces the v1 "Research Methodology" document entirely. v1 is retired, not amended.

---

## 1. Why v1 kept producing mistakes

**The analogy.** v1 is a meticulous cargo-loading checklist — weight distribution, lashing
points, manifest verification — for a ship with nobody checking the heading. Every section
governs *how to place bets safely*. No section asks *is this price wrong?* That question is
the only thing that produces profit. Everything else is damage control.

Three failures, in order of how much they cost you:

### 1.1 There is no edge test anywhere in v1

v1 never compares an estimated probability against the price being offered. "Milan vs Lecce,
Over 2.5" is a **prediction**. A **bet** is "I make it 62%, the price implies 57%, so I buy."
Without the second half there is nothing to be right or wrong about, and no amount of squad
verification or market diversity changes that.

Worse, §7 of v1 has it backwards: it derives odds *from* a win probability plus an assumed
margin. The arrow runs the other way. You take the real posted odds, strip the margin out of
them, and *that* is the market's probability — the number you have to beat.

### 1.2 The experiment cannot answer its own question

This is why it feels like it produces no value: structurally, it can't. Simulated at the v1
parameters (2 steps × 5 legs, odds 1.75):

| Truth about your method | P(the €10 shows a profit) |
|---|---|
| No edge at all (5% book margin) | 36.2% |
| Exactly break-even | 43.2% |
| A genuine +5% ROI edge | 50.2% |

A real, valuable edge shows a profit **half the time**. No edge shows a profit **more than a
third** of the time. The output is nearly uninformative about the input.

Sharper still — take two methods, one with a true +5% ROI edge and one with a true −5%, and
run both for 10 legs: **48.6% of the time the 10 legs fail to identify the better one** (31.4%
the worse one looks better, 17.3% a tie). You have been grading your research with a coin.

To confirm a +3% ROI edge from win/loss results alone takes roughly **4,000 bets**. Ten is not
a small sample; it is not a sample.

### 1.3 The data layer was a Claude session

Gyökeres and Lewandowski were not discipline lapses that a checklist fixes. They are the
predictable output of asking a language model to recall a current squad. v1 §3's answer —
verify harder — routes the verification through the same unreliable channel.

Two hard facts that make this concrete:

- **`fetch_sports_data`, ranked #1 in v1 §2, does not exist in this session.** It is not in the
  tool list here. A session without it silently degrades to web search plus model recall — which
  is exactly the failure mode that produced both errors. v1 never says "check the tool is there
  first," so the degradation is invisible.
- **This container cannot bulk-download data files.** The egress proxy refuses
  `football-data.co.uk` over curl (403 on CONNECT), though WebFetch can read pages. So a Claude
  session can never be the data layer, in this environment, by design.

### 1.4 Three factual claims in v1 that are simply wrong

- *"No reliable corners data source currently available."* Corners are in the standard
  football-data.co.uk match file as `HC` / `AC`, every match, every season back to the 1990s.
- *"Eredivisie / NL leagues have no feed."* The Netherlands is one of the eleven **main**
  covered countries on the same source, with full odds history.
- *"All odds are estimates, can't be verified."* Actual Bet365 **opening and closing** odds ship
  in the same CSV (`B365H/D/A` and `B365CH/CD/CA`), plus Betfair exchange closing prices.

All three were limitations of one session's tooling, mistaken for limitations of the world.

### 1.5 The staking section optimises the wrong objective

v1 §1 chose split-5 on a **survival** criterion. Nobody's goal is to survive ten steps and
finish with €3. If the objective is "turn €10 into a target within N steps", the mathematics
inverts — in a game with no edge, spreading stakes applies the house margin more times and
*reduces* your chance of getting there:

| | P(reach €30 within 10 steps) |
|---|---|
| **No edge**, bold staking | 29.9% |
| **No edge**, split-5 | 10.3% |
| **+5% edge**, bold staking | 36.9% |
| **+5% edge**, split-5 | 26.8% |

Split staking is correct for **maximising long-run growth when you have an edge** (that is
Kelly). It is wrong for **hitting a target**, and it is irrelevant when there is no edge. v1
conflated the two objectives and picked a rule that fits neither of its stated goals.

Two smaller corrections: v1's "~90% chance of €0 by step 5–10" understates it — at p=0.60 it is
**99.4%** by step 10. And "expected value is identical across structures" is only true per euro
staked; with a 5% margin and no edge, the expected end of the v1 experiment is **€9.03** from
€10, whatever the structure.

### 1.6 The mandated player prop is the worst rule in the document

v1 §4 requires one player prop per slate. Book margins run ~2–5% on main markets in big
matches and **7%+ on props, corners and specials**. So v1 mandates the leg where the bookmaker
takes the largest cut *and* where your data is weakest *and* which caused both logged errors.
It is a quota that forces you into your worst bet every single week.

Related: "market diversity" is not the same as independence. Five Over-2.5 bets in five
*different* matches are near-independent and diversify fine. "Team to win" + "Over 2.5" in the
*same* match are positively correlated and behave closer to one bet than two. **Diversify by
fixture, not by market label.**

### 1.7 "Compounding" is not a strategy

Compounding is what fractional staking does automatically. It is a side effect, not a method.
The project is named after the arithmetic rather than the edge.

---

## 2. The one change that fixes this: measure CLV

**Closing Line Value** = the odds you took versus the odds the same market closed at. If you
consistently take prices better than closing, you are beating the market and profit follows.
If you don't, no staking rule, market mix or verification protocol will save you.

Why it matters here specifically — it is a *continuous* measurement, not a binary win/loss, so
the noise collapses:

| Evidence you're using | Bets needed for a verdict |
|---|---|
| Win/loss results, +3% ROI edge | ~4,000 |
| CLV (SD 4%, true mean +2%) | **~16** |
| CLV (SD 5%, true mean +1.5%) | ~44 |

**Two hundred and fifty times faster.** One extra column in the log turns a diary into an
experiment. This is the surgical fix; everything below is scaffolding around it.

### Two corrections to this section, found while implementing it

**CLV must be measured against the DE-VIGGED close, not the raw close.** Comparing the price
you took against the bookmaker's own closing price flatters you by roughly the margin, because
that closing price still carries it. In the reference implementation's sharp-book test this
showed up as **+3.18% raw CLV alongside −2% ROI** — positive CLV on a book that is, by
construction, unbeatable. Measured against the de-vigged close the same bets scored −3.49%,
correctly reporting no edge. Log both; trust the fair one.

**CLV is a proxy for edge only when the closing line is efficient.** The planted-bias test is
genuinely profitable (+18.8% ROI, gate passed) and still shows negative fair CLV, because that
book's error *persists to the close* — the line never moves, so there is no movement for CLV to
capture. In a market that stays wrong you can profit with negative CLV; against a sharp book you
can show positive raw CLV and still lose.

So CLV is the **fastest** signal, not the **primary** one. The log-loss gate in §5 is primary;
CLV is what gives you a reading in tens of bets instead of thousands once the gate is passed.

Record, for every selection, paper or real:

```
timestamp_taken, fixture, market, selection,
p_model,              # your probability, written BEFORE looking at the price
odds_available,       # the real price at that moment
odds_closing,         # filled in after the market closes
clv = odds_available / odds_closing - 1,
result
```

`p_model` is written before `odds_available` is looked up. If you write the probability after
seeing the price, you are rationalising, not forecasting, and the whole log is worthless.

---

## 3. Pick protocol — hard gates

A selection that fails any gate is not "a weaker pick". It is **not a pick**.

**Gate 0 — Sources.** Name the data source and its date. If no structured current-season source
is available in this session, say so and stop. No pick is built on model recall.

**Gate 1 — Probability first.** State `p_model` and the reasoning for it, before the price.

**Gate 2 — Price.** Get the *real* posted odds. Not an estimate, not "~1.70–1.85". If a real
price cannot be obtained, the output is a *candidate*, explicitly labelled "unpriced", and it
is never counted as a pick.

**Gate 3 — Edge.**
```
fair_odds     = 1 / p_model
edge          = p_model * odds_available - 1
```
Bet only if `edge >= 0.04` (4%). Below that, the edge is inside your own model error.

**Gate 4 — Benchmark.** De-vig the market (normalise 1/odds across all outcomes so they sum to
1) and state the market's implied probability alongside yours. If you cannot articulate *why*
the market is wrong on this specific match, the disagreement is your error, not their mispricing.

**Gate 5 — Correlation.** No two selections in a slate may share a fixture or a single causal
driver (same key player, same weather event, same referee).

**Expected consequence: most weekends produce zero or one qualifying bet.** That is the correct
output. v1 structurally forbade this answer by requiring five legs; a method that must produce
five bets a week will manufacture five bets a week.

---

## 4. Market scope

**Kept** (liquid, low margin, modellable from data you can hold locally):
- Match result / double chance
- Asian handicap
- Over/Under total goals

**Dropped:**
- **Player props** — highest margin, weakest data, both logged errors. Out until there is a
  structured, current, machine-readable squad and minutes source.
- **Corners and cards as bets** — historical counts exist (`HC`/`AC`), historical *odds* don't,
  so there is no way to backtest. Fine as a model input, not as a bet.
- **The 1.5–2.0 odds band.** This is a price filter masquerading as a value filter. It selects
  bets by cost rather than by worth. Deleted. At 1.50 you need a 66.7% strike rate; at 2.00 you
  need 50%. The band tells you nothing about whether either is achievable.

---

## 5. The data layer — local, not in a chat session

Source: **football-data.co.uk**, `mmz4281/{season}/{div}.csv` (e.g. `2627/E0.csv` — confirmed
live, 30 matches through 14 Sep 2026).

That one file carries everything the method needs:

| Purpose | Columns |
|---|---|
| Result | `FTHG`, `FTAG`, `FTR` |
| **Expected goals** | `HxG`, `AxG` |
| Shots / on target | `HS`, `AS`, `HST`, `AST` |
| Corners | `HC`, `AC` |
| Bet365 opening 1X2 | `B365H`, `B365D`, `B365A` |
| **Bet365 closing 1X2** | `B365CH`, `B365CD`, `B365CA` |
| Closing totals | `B365C>2.5`, `B365C<2.5` |
| Closing Asian handicap | `AHCh`, `B365CAHH`, `B365CAHA` |
| Exchange closing (near-fair benchmark) | `BFECH`, `BFECD`, `BFECA` |
| Best / average closing across books | `MaxCH`, `AvgCH` … |

Coverage: 11 main countries (England, Scotland, Germany, Italy, Spain, France, **Netherlands**,
Belgium, Portugal, Turkey, Greece) plus 15 more, back 25 seasons.

Opening *and* closing odds in the same row is the important part: it means **CLV is backtestable
before you ever stake anything.**

### Baseline model to build (runs on your Mac, not in a session)

1. Pull seasons 2021/22 → current for E0, E1, D1, SP1, I1, F1, N1.
2. Fit Dixon–Coles: per-team attack and defence ratings, home advantage, exponential time decay
   (~180-day half-life). **Fit on xG, not goals** — less noise per match, better out-of-sample.
3. Walk-forward strictly: every prediction uses only matches dated before it. No exceptions; this
   is where backtests usually lie.
4. Output the full scoreline matrix → P(H/D/A), P(Over 2.5), any handicap line.
5. De-vig `B365C*` (and `BFEC*`) and score both.

**The gate before any money moves:**

> Compute your model's **log loss** against match outcomes. Compute the de-vigged closing line's
> log loss over the same matches. **If your model does not beat the closing line, you have no
> edge and you stake nothing.** No staking scheme, market mix or verification protocol rescues a
> model that is worse-calibrated than the price.

Only if it clears that: simulate betting where `edge >= 4%` against `B365H/D/A` (the earlier
price) and check both ROI and CLV proxy (`B365H` vs `B365CH`).

**The honest prior:** top-five-league 1X2 is among the sharpest markets in football, and a
public-xG Dixon–Coles beating Bet365's close there is unlikely. Where an edge has plausibly
survived: early-posted lines before the market forms, lower-liquidity divisions, and totals
rather than results. Expect the backtest to say no. **A clean no is the most valuable output
this project can produce**, because it is the one result that stops you spending money to find
out slowly.

---

## 6. Staking

**Now, and until the model beats the closing line: €0. Paper only.**

If and when it clears the gate:
- Fractional Kelly at **¼ Kelly**, hard-capped at **2% of bankroll per bet**.
- Stake follows the edge. No fixed leg count, no equal splits, no odds band.
- €10 cannot support this — Kelly on €10 is cents. So separate the two things, permanently:

| | Purpose | Rules |
|---|---|---|
| **The experiment** | Answer "do I beat the close?" | Paper, unlimited volume, CLV-logged, zero money |
| **The €10** | Fun | Stake it however you like. It is not evidence and is never analysed as though it were. |

Conflating these is what made the project feel like it was failing. It wasn't failing — it was
being measured with an instrument that cannot detect the thing it was looking for.

---

## 7. What a Claude session is for, and what it isn't

**Is for:** reasoning over model output you supply; de-vigging and edge arithmetic; reading
team news and manager quotes; sanity-checking a slate against these gates; writing and reviewing
the model code.

**Is not for:** being the data source. Not squads, not current form, not "estimated" odds. In
this environment there is no sports data tool and no bulk download. Any session that starts
producing fixture-level numbers without naming a live source is reconstructing them from recall,
which is where every logged error came from.

**Standing instruction for every session in this project:** open by stating which data sources
are actually available in that session. If the answer is "web search only", the session can
critique, compute and write code — it cannot originate picks.

---

## 8. Kill criteria

Written down now, while it costs nothing to be honest:

- Model fails to beat the de-vigged closing line on ≥1,000 historical matches → **stop**. Do not
  tune until it passes; that is fitting the test.
- After 100 logged paper picks, mean CLV ≤ 0 → **stop**.
- Any pick reaches the log without `p_model` recorded before the price → the whole slate is void,
  because the log is the only asset this project builds.

---

## 9. Log format

Kept in the project. One row per selection, paper and real in the same table, distinguished by
a `stake` column that is `0` for paper.

```
date_taken | kickoff | league | fixture | market | selection |
p_model | odds_available | fair_odds | edge_pct | devigged_market_p |
odds_closing | fair_closing | clv_raw_pct | clv_fair_pct |
result | stake | pnl | source_note
```

`fair_closing` is `1 / devigged_closing_probability`. `clv_fair_pct` is the honest number;
`clv_raw_pct` is kept only so the gap between them stays visible, because that gap is the
margin.

Corrections stay visible with a note on what went wrong — that part of v1 was right and carries
over unchanged.


---

## 10. Implementation status

The reference implementation of this spec exists: package `cbets`, 30 passing tests.

**Validated against synthetic worlds with known ground truth**, because a backtest's errors are
invisible when you don't know the right answer. Three checks, all passing:

- the fit recovers true team ratings (attack r=0.99, defence r=0.98, home advantage within 0.002)
- against a **sharp** book the gate correctly reports **no edge** — if it ever passes here, the
  harness is lying
- against a book with a **planted** 5-point draw bias the gate finds it, and the simulated bets
  land on draws, which is where the bias was put

Both directions matter. A harness that only checks it can find edges will find them everywhere.

**Gate 0 was implemented wrong first.** It checked that `source_note` was at least ten
characters — which "looks good" satisfies. A length check is not a source check. It now requires
a named source *and* a parseable date, because the failure this gate exists to prevent is a stale
archive page being taken for current information.

**The honest prior stands.** The gate is expected to fail on top-five-league 1X2. That is the
result that saves money.
