#!/usr/bin/env python3
"""The daily run. Deterministic: all numbers come from here, none from a model's recall.

    python daily.py --repo https://raw.githubusercontent.com/<user>/<repo>/main \
                    --log picks.csv --out report.json

What one run does, in order:

  1. Pull data/fixtures.csv (upcoming matches + live odds) and the season files.
  2. SETTLE pending picks: find each one in the now-completed season data, record
     the result and the closing odds, and compute both CLV numbers.
  3. Refit Dixon-Coles per division on everything dated before today.
  4. Price every upcoming fixture and compare against the de-vigged market.
  5. Apply the Method v2 gates. Most days this yields nothing, which is correct.
  6. Emit report.json with a `notify` flag so the session stays quiet on a
     nothing day.

The probability is computed before the price is consulted, structurally: step 3
never sees the odds columns at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cbets import data as D
from cbets.devig import devig
from cbets.model import fit as fit_model
from cbets.model import probs_1x2, score_matrix
from cbets.picks import FIELDS

EDGE_THRESHOLD = 0.04
MIN_TRAIN = 380
START_YEAR, END_YEAR = 2019, 2026
KILL_AFTER_N_PICKS = 100

# 1X2 selections plus the totals market. Player props, corners and cards are not
# here and are not coming back -- Method v2 section 4.
SEL_1X2 = ("H", "D", "A")


def _read(repo: str, rel: str) -> pd.DataFrame | None:
    """Read a CSV from the repo, whether `repo` is a URL base or a local path."""
    src = f"{repo.rstrip('/')}/{rel}" if "://" in repo else str(Path(repo) / rel)
    try:
        if "://" in src:
            import requests
            r = requests.get(src, timeout=45)
            if r.status_code != 200 or len(r.content) < 200:
                return None
            import io
            return pd.read_csv(io.BytesIO(r.content), encoding="latin-1", on_bad_lines="skip")
        p = Path(src)
        return pd.read_csv(p, encoding="latin-1", on_bad_lines="skip") if p.exists() else None
    except Exception as exc:
        print(f"  ! {rel}: {exc}", file=sys.stderr)
        return None


def load_history(repo: str, divisions: list[str]) -> pd.DataFrame:
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        sc = D.season_code(year)
        for div in divisions:
            raw = _read(repo, f"data/{sc}/{div}.csv")
            if raw is None or raw.empty:
                continue
            norm = D.normalise(raw, sc, div)
            if not norm.empty:
                frames.append(norm)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return df


def load_fixtures(repo: str) -> pd.DataFrame:
    raw = _read(repo, "data/fixtures.csv")
    if raw is None or raw.empty:
        return pd.DataFrame()
    fx = D.normalise(raw, "fixtures", "")
    if fx.empty:
        # fixtures.csv has no results, so normalise's dropna removes everything.
        raw = raw.copy()
        raw["FTHG"] = 0
        raw["FTAG"] = 0
        fx = D.normalise(raw, "fixtures", "")
        fx["fthg"] = np.nan
        fx["ftag"] = np.nan
    fx["div"] = raw["Div"].astype(str).str.strip().values[: len(fx)] if "Div" in raw else ""
    return fx


# ----------------------------------------------------------------- settling

def settle(log: pd.DataFrame, hist: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill in result and closing odds for picks whose match has now been played."""
    if log.empty or hist.empty:
        return log, 0
    pending = log["result"].astype(str).str.lower().eq("pending")
    if not pending.any():
        return log, 0

    key = hist.assign(k=hist["home"].str.lower() + "|" + hist["away"].str.lower())
    settled = 0
    for i in log.index[pending]:
        fixture = str(log.at[i, "fixture"])
        if " v " not in fixture:
            continue
        h, a = [x.strip().lower() for x in fixture.split(" v ", 1)]
        m = key[(key["k"] == f"{h}|{a}") & (key["date"] >= pd.Timestamp(log.at[i, "kickoff"]).normalize()
                                            - pd.Timedelta(days=3))]
        if m.empty:
            continue
        row = m.iloc[0]
        if not np.isfinite(row["fthg"]):
            continue

        sel = str(log.at[i, "selection"]).strip().upper()
        won = {"H": row["fthg"] > row["ftag"], "D": row["fthg"] == row["ftag"],
               "A": row["fthg"] < row["ftag"],
               "OVER": (row["fthg"] + row["ftag"]) > 2.5,
               "UNDER": (row["fthg"] + row["ftag"]) < 2.5}.get(sel)
        if won is None:
            continue

        log.at[i, "result"] = "win" if won else "loss"
        odds = float(log.at[i, "odds_available"])
        log.at[i, "pnl"] = (odds - 1.0) if won else -1.0

        if sel in SEL_1X2:
            trio = [row["c_h"], row["c_d"], row["c_a"]]
            if all(np.isfinite(x) and x > 1 for x in trio):
                k = SEL_1X2.index(sel)
                log.at[i, "odds_closing"] = trio[k]
                q = devig(trio, "shin")
                fair = 1.0 / q[k]
                log.at[i, "fair_closing"] = round(fair, 4)
                log.at[i, "clv_raw_pct"] = round((odds / trio[k] - 1) * 100, 3)
                log.at[i, "clv_fair_pct"] = round((odds / fair - 1) * 100, 3)
        else:
            pair = [row["c_over25"], row["c_under25"]]
            if all(np.isfinite(x) and x > 1 for x in pair):
                k = 0 if sel == "OVER" else 1
                log.at[i, "odds_closing"] = pair[k]
                q = devig(pair, "multiplicative")
                fair = 1.0 / q[k]
                log.at[i, "fair_closing"] = round(fair, 4)
                log.at[i, "clv_raw_pct"] = round((odds / pair[k] - 1) * 100, 3)
                log.at[i, "clv_fair_pct"] = round((odds / fair - 1) * 100, 3)
        settled += 1
    return log, settled


# ------------------------------------------------------------------ scanning

def scan(hist: pd.DataFrame, fixtures: pd.DataFrame, today: pd.Timestamp,
         horizon_days: int = 3) -> tuple[list[dict], dict]:
    """Price every upcoming fixture and return the selections that clear the gates."""
    upcoming = fixtures[(fixtures["date"] >= today)
                        & (fixtures["date"] <= today + pd.Timedelta(days=horizon_days))]
    candidates, stats = [], {"fixtures_scanned": 0, "divisions_priced": [], "divisions_skipped": {}}

    for div, grp in upcoming.groupby("div"):
        train = hist[(hist["div"] == div) & (hist["date"] < today)]
        if len(train) < MIN_TRAIN:
            stats["divisions_skipped"][str(div)] = f"only {len(train)} training matches"
            continue
        try:
            model = fit_model(train, as_of=today, response="blend", half_life_days=180, ridge=0.02)
        except Exception as exc:
            stats["divisions_skipped"][str(div)] = f"fit failed: {exc}"
            continue
        stats["divisions_priced"].append(str(div))

        for r in grp.itertuples():
            stats["fixtures_scanned"] += 1
            # --- probability first: the odds columns are not consulted here ---
            lam, mu = model.rates(r.home, r.away)
            m = score_matrix(lam, mu, model.rho, model.max_goals)
            p1x2 = probs_1x2(m)
            idx = np.arange(m.shape[0])
            p_over = float(m[(idx[:, None] + idx[None, :]) > 2.5].sum())

            # --- now, and only now, the price ---
            trio = [r.o_h, r.o_d, r.o_a]
            if all(np.isfinite(x) and x > 1 for x in trio):
                q = devig(trio, "shin")
                for k, sel in enumerate(SEL_1X2):
                    edge = p1x2[k] * trio[k] - 1.0
                    if edge >= EDGE_THRESHOLD:
                        candidates.append(dict(
                            kickoff=str(r.date.date()), league=str(div),
                            fixture=f"{r.home} v {r.away}", market="1x2", selection=sel,
                            p_model=float(p1x2[k]), odds_available=float(trio[k]),
                            devigged_market_p=float(q[k]), edge_pct=float(edge * 100),
                            lam=lam, mu=mu))
            pair = [r.o_over25, r.o_under25]
            if all(np.isfinite(x) and x > 1 for x in pair):
                q = devig(pair, "multiplicative")
                for k, (sel, p) in enumerate((("OVER", p_over), ("UNDER", 1 - p_over))):
                    edge = p * pair[k] - 1.0
                    if edge >= EDGE_THRESHOLD:
                        candidates.append(dict(
                            kickoff=str(r.date.date()), league=str(div),
                            fixture=f"{r.home} v {r.away}", market="over_under", selection=sel,
                            p_model=float(p), odds_available=float(pair[k]),
                            devigged_market_p=float(q[k]), edge_pct=float(edge * 100),
                            lam=lam, mu=mu))

    # Gate 5: one selection per fixture, keeping the largest edge.
    best: dict[str, dict] = {}
    for c in candidates:
        if c["fixture"] not in best or c["edge_pct"] > best[c["fixture"]]["edge_pct"]:
            best[c["fixture"]] = c
    return sorted(best.values(), key=lambda c: -c["edge_pct"]), stats


# ------------------------------------------------------------------ reporting

def running_stats(log: pd.DataFrame) -> dict:
    out = {"picks_total": int(len(log))}
    if log.empty:
        return out
    fair = pd.to_numeric(log.get("clv_fair_pct"), errors="coerce").dropna()
    out["picks_with_clv"] = int(len(fair))
    if len(fair):
        m, sd = float(fair.mean()), float(fair.std(ddof=1)) if len(fair) > 1 else 0.0
        se = sd / np.sqrt(len(fair)) if len(fair) > 1 else 0.0
        out["mean_clv_fair_pct"] = round(m, 3)
        out["clv_ci95"] = [round(m - 1.96 * se, 3), round(m + 1.96 * se, 3)]
        out["clv_verdict"] = (
            "positive and significant" if m - 1.96 * se > 0 else
            "negative and significant" if m + 1.96 * se < 0 else "not distinguishable from zero")
        if len(fair) >= KILL_AFTER_N_PICKS and m <= 0:
            out["kill_criterion"] = (
                f"MET — {len(fair)} picks with mean fair CLV {m:+.2f}%. Method v2 section 8 says stop.")
    settled = log[log["result"].astype(str).str.lower().isin(["win", "loss"])]
    out["picks_settled"] = int(len(settled))
    if len(settled):
        pnl = pd.to_numeric(settled["pnl"], errors="coerce").dropna()
        if len(pnl):
            out["paper_roi_pct"] = round(float(pnl.sum() / len(pnl) * 100), 3)
        out["hit_rate"] = round(float(settled["result"].str.lower().eq("win").mean()), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="raw.githubusercontent base URL, or a local path")
    ap.add_argument("--log", default="picks.csv")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--today", default=None)
    ap.add_argument("--horizon-days", type=int, default=3)
    a = ap.parse_args()

    today = pd.Timestamp(a.today) if a.today else pd.Timestamp(date.today())
    report = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "today": str(today.date()), "notify": False, "errors": []}

    fixtures = load_fixtures(a.repo)
    if fixtures.empty:
        report["errors"].append(
            "data/fixtures.csv missing or unreadable — the GitHub Action may not have run yet")
        report["notify"] = True
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1

    divisions = sorted(set(fixtures["div"].dropna().astype(str)) - {""})
    hist = load_history(a.repo, divisions)
    if hist.empty:
        report["errors"].append("no historical season files found in the repo")
        report["notify"] = True
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 1
    report["history_matches"] = int(len(hist))
    report["history_with_closing"] = int(hist[["c_h", "c_d", "c_a"]].notna().all(axis=1).sum())

    log_path = Path(a.log)
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=FIELDS)
    for c in FIELDS:
        if c not in log.columns:
            log[c] = np.nan
    before = running_stats(log)
    log, n_settled = settle(log, hist)
    report["picks_settled_today"] = n_settled

    picks, stats = scan(hist, fixtures, today, a.horizon_days)
    report.update(stats)
    report["qualifying_picks"] = picks

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for p in picks:
        log.loc[len(log)] = {
            "date_taken": now, "kickoff": p["kickoff"], "league": p["league"],
            "fixture": p["fixture"], "market": p["market"], "selection": p["selection"],
            "p_model": round(p["p_model"], 6), "odds_available": p["odds_available"],
            "fair_odds": round(1 / p["p_model"], 4), "edge_pct": round(p["edge_pct"], 3),
            "devigged_market_p": round(p["devigged_market_p"], 6),
            "odds_closing": np.nan, "fair_closing": np.nan,
            "clv_raw_pct": np.nan, "clv_fair_pct": np.nan,
            "result": "pending", "stake": 0.0, "pnl": np.nan,
            "source_note": f"cbets daily run, football-data.co.uk via repo, {today.date()}",
        }
    log.to_csv(log_path, index=False)

    after = running_stats(log)
    report["running"] = after

    # Quiet unless it matters.
    reasons = []
    if picks:
        reasons.append(f"{len(picks)} selection(s) cleared the {EDGE_THRESHOLD*100:.0f}% edge gate")
    if "kill_criterion" in after and "kill_criterion" not in before:
        reasons.append("kill criterion met")
    if after.get("clv_verdict") != before.get("clv_verdict") and after.get("picks_with_clv", 0) >= 20:
        reasons.append(f"CLV verdict changed to: {after.get('clv_verdict')}")
    if report["errors"]:
        reasons.append("errors during the run")
    report["notify"] = bool(reasons)
    report["notify_reasons"] = reasons

    Path(a.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
