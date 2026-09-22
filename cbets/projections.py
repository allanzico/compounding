"""Per-fixture projections across every market the data can support.

One fitted model set per division per run, then every upcoming fixture priced
across goals, half-time, handicap and the count markets. Probabilities only —
where a real market price exists in the fixture file it is shown alongside, and
where it does not, the fair odds are given so the price can be checked by hand.

Every market carries the verdict from its own walk-forward validation, because a
number that does not beat "the average game has 9.8 corners" should not look the
same on the page as one that does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .counts import METRICS, fit_counts
from .leagues import country_of, label_of, name_of
from .devig import devig
from .model import ah_home, fit as fit_goals, prob_over, probs_1x2, score_matrix

__all__ = ["project_division", "MARKET_LABELS", "COUNT_MARKETS"]

COUNT_MARKETS = ["shots", "sot", "corners", "fouls", "yellows"]

# Filled from the validation run; anything missing is reported as untested.
MARKET_LABELS: dict[str, str] = {}

MIN_TEAM_EFFECTIVE_MATCHES = 10.0


def _fair(p: float) -> float:
    return float(1.0 / p) if p > 1e-9 else float("inf")


def _lines_around(mean: float, step: float = 1.0, n: int = 3) -> list[float]:
    """Half-lines bracketing the expected value, which is where books set them."""
    base = np.floor(mean) + 0.5
    offs = [-step, 0.0, step] if n == 3 else [-2 * step, -step, 0.0, step, 2 * step]
    return [float(base + o) for o in offs if base + o > 0]


def project_division(hist: pd.DataFrame, fixtures: pd.DataFrame, div: str,
                     today: pd.Timestamp, half_life: float = 180.0,
                     ridge: float = 1.0) -> tuple[list[dict], dict]:
    """Project every fixture in one division. Returns (rows, diagnostics)."""
    train = hist[(hist["div"] == div) & (hist["date"] < today)]
    diag = {"div": div, "train": len(train), "skipped": [], "models": []}
    if len(train) < 380:
        diag["error"] = f"only {len(train)} training matches"
        return [], diag

    try:
        gm = fit_goals(train, as_of=today, response="blend",
                       half_life_days=half_life, ridge=ridge)
    except Exception as exc:
        diag["error"] = f"goals fit failed: {exc}"
        return [], diag
    diag["models"].append("goals")

    cms: dict[str, object] = {}
    for metric in ["ht_goals"] + COUNT_MARKETS:
        try:
            cms[metric] = fit_counts(train, metric, as_of=today,
                                     half_life_days=half_life, ridge=ridge)
            diag["models"].append(metric)
        except Exception:
            pass

    rows = []
    for r in fixtures.itertuples():
        h, a = str(r.home), str(r.away)
        wh, wa = gm.effective_matches(h), gm.effective_matches(a)
        if min(wh, wa) < MIN_TEAM_EFFECTIVE_MATCHES:
            diag["skipped"].append(f"{h} v {a} ({min(wh, wa):.0f} eff. matches)")
            continue

        ko = getattr(r, "time", None)
        ko = str(ko) if ko is not None and str(ko) not in ("nan", "NaT", "") else ""
        base = dict(date=str(pd.Timestamp(r.date).date()), kickoff=ko, div=div,
                    country=country_of(div), league=name_of(div), league_label=label_of(div),
                    fixture=f"{h} v {a}", home=h, away=a,
                    confidence=round(float(min(wh, wa)), 1))

        lam, mu = gm.rates(h, a)
        m = score_matrix(lam, mu, gm.rho, gm.max_goals)
        p1x2 = probs_1x2(m)
        base["xg_home"], base["xg_away"] = round(lam, 2), round(mu, 2)

        def add(market, selection, p, price=np.nan, group="goals"):
            p = float(np.clip(p, 1e-6, 1 - 1e-6))
            row = dict(base, group=group, market=market, selection=selection,
                       prob=round(p, 4), fair_odds=round(_fair(p), 2),
                       validation=MARKET_LABELS.get(group, "untested"))
            if np.isfinite(price) and price > 1:
                row["price"] = float(price)
                row["edge_pct"] = round((p * price - 1) * 100, 2)
            rows.append(row)

        # ---- match result ----
        for k, (sel, price) in enumerate((("Home", r.o_h), ("Draw", r.o_d), ("Away", r.o_a))):
            add("1X2", sel, p1x2[k], price)
        add("Double chance", "Home or Draw", p1x2[0] + p1x2[1])
        add("Double chance", "Home or Away", p1x2[0] + p1x2[2])
        add("Double chance", "Draw or Away", p1x2[1] + p1x2[2])

        # ---- goals ----
        for line in (0.5, 1.5, 2.5, 3.5, 4.5):
            po = prob_over(m, line)
            price_o = r.o_over25 if line == 2.5 else np.nan
            price_u = r.o_under25 if line == 2.5 else np.nan
            add("Total goals", f"Over {line}", po, price_o, "goals")
            add("Total goals", f"Under {line}", 1 - po, price_u, "goals")
        idx = np.arange(m.shape[0])
        btts = float(m[np.ix_(idx >= 1, idx >= 1)].sum())
        add("Both teams to score", "Yes", btts, group="goals")
        add("Both teams to score", "No", 1 - btts, group="goals")
        for side, rate in (("Home", lam), ("Away", mu)):
            for line in (0.5, 1.5, 2.5):
                add(f"{side} team goals", f"Over {line}",
                    float(poisson(rate).sf(np.floor(line))), group="goals")

        # ---- asian handicap ----
        if np.isfinite(getattr(r, "ah_line", np.nan)):
            ln = float(r.ah_line)
            w, push, l = ah_home(m, ln)
            denom = max(1e-9, w + l)
            add("Asian handicap", f"Home {ln:+g}", w / denom, getattr(r, "o_ah_h", np.nan), "goals")
            add("Asian handicap", f"Away {-ln:+g}", l / denom, getattr(r, "o_ah_a", np.nan), "goals")

        # ---- half time ----
        if "ht_goals" in cms:
            c = cms["ht_goals"]
            if c.knows(h, a):
                hl, hm = c.rates(h, a)
                hmat = score_matrix(hl, hm, 0.0, 6)
                hp = probs_1x2(hmat)
                for k, sel in enumerate(("Home", "Draw", "Away")):
                    add("Half-time result", sel, hp[k], group="ht_goals")
                for line in (0.5, 1.5, 2.5):
                    add("Half-time goals", f"Over {line}", prob_over(hmat, line),
                        group="ht_goals")

        # ---- count markets ----
        for metric in COUNT_MARKETS:
            c = cms.get(metric)
            if c is None or not c.knows(h, a):
                continue
            ch, ca, tot = c.expected(h, a)
            nice = {"sot": "Shots on target", "shots": "Shots", "corners": "Corners",
                    "fouls": "Fouls", "yellows": "Yellow cards"}[metric]
            step = 1.0 if metric in ("corners", "sot", "yellows") else 2.0
            for line in _lines_around(tot, step=step):
                po = c.prob_total_over(h, a, line)
                add(f"Total {nice.lower()}", f"Over {line}", po, group=metric)
                add(f"Total {nice.lower()}", f"Under {line}", 1 - po, group=metric)
            for side, rate in (("Home", ch), ("Away", ca)):
                for line in _lines_around(rate, step=step, n=3)[:2]:
                    add(f"{side} {nice.lower()}", f"Over {line}",
                        c.prob_team_over(h, a, line, side.lower()), group=metric)
            base[f"exp_{metric}"] = round(tot, 1)

    diag["rows"] = len(rows)
    return rows, diag
