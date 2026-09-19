"""End-to-end test of the daily agent against a fake repo built from synthetic data.

This is the test that matters for the autonomous setup: it exercises the real
daily.py against real CSV files laid out exactly as the GitHub repo will be,
including the settle-then-scan cycle across two consecutive days.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cbets.synth import make_league

ROOT = Path(__file__).resolve().parent.parent


def write_repo(tmp: Path, bias_draw: float = 0.0, seed: int = 21):
    """Lay out synthetic data in the raw football-data.co.uk format the repo uses."""
    L = make_league(n_teams=20, seasons=5, seed=seed, bias_draw=bias_draw)
    df = L.matches.copy()

    ren = {
        "home": "HomeTeam", "away": "AwayTeam", "fthg": "FTHG", "ftag": "FTAG",
        "hxg": "HxG", "axg": "AxG",
        "o_h": "B365H", "o_d": "B365D", "o_a": "B365A",
        "c_h": "B365CH", "c_d": "B365CD", "c_a": "B365CA",
        "o_over25": "B365>2.5", "o_under25": "B365<2.5",
        "c_over25": "B365C>2.5", "c_under25": "B365C<2.5",
    }
    out = df.rename(columns=ren)
    out["Div"] = "E0"
    out["Date"] = out["date"].dt.strftime("%d/%m/%Y")
    cols = ["Div", "Date"] + list(ren.values())

    # Hold back the final week as "upcoming fixtures", everything before as history.
    cutoff = out["date"].max() - pd.Timedelta(days=7)
    hist, future = out[out["date"] <= cutoff], out[out["date"] > cutoff]
    assert len(future) > 0

    (tmp / "data" / "2425").mkdir(parents=True, exist_ok=True)
    hist[cols].to_csv(tmp / "data" / "2425" / "E0.csv", index=False)

    # fixtures.csv carries odds but no results, exactly like the real file.
    fx = future[["Div", "Date", "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
                 "HomeTeam", "AwayTeam"]].copy()
    fx.to_csv(tmp / "data" / "fixtures.csv", index=False)

    # The full season, used to simulate "the next day" when results have landed.
    out[cols].to_csv(tmp / "full_season.csv", index=False)
    return future["date"].min(), cutoff


def run_daily(tmp: Path, today, log="picks.csv", out="report.json", extra=()):
    cmd = [sys.executable, str(ROOT / "daily.py"), "--repo", str(tmp),
           "--log", str(tmp / log), "--out", str(tmp / out),
           "--today", str(pd.Timestamp(today).date()), "--horizon-days", "7", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    import json
    rep = json.loads((tmp / out).read_text()) if (tmp / out).exists() else {}
    return r, rep


def test_daily_runs_and_finds_planted_edges(tmp_path):
    first_future, _ = write_repo(tmp_path, bias_draw=-0.06)
    r, rep = run_daily(tmp_path, first_future)
    assert r.returncode == 0, r.stderr[-2000:]
    assert rep["errors"] == []
    assert rep["fixtures_scanned"] > 0
    assert len(rep["qualifying_picks"]) > 0, "planted edge produced no picks"
    assert rep["notify"] is True

    log = pd.read_csv(tmp_path / "picks.csv")
    assert len(log) == len(rep["qualifying_picks"])
    assert (log["result"] == "pending").all()
    assert (log["stake"] == 0).all(), "the daily run must never stake money"
    # every logged pick must clear the edge gate
    assert (log["edge_pct"] >= 4.0 - 1e-9).all()
    # Gate 5: one selection per fixture
    assert log["fixture"].duplicated().sum() == 0


def test_sharp_book_produces_far_fewer_picks(tmp_path):
    """Against a sharp book the honest output is close to nothing."""
    first_future, _ = write_repo(tmp_path, bias_draw=0.0)
    _, rep = run_daily(tmp_path, first_future)
    sharp = len(rep["qualifying_picks"])

    tmp2 = tmp_path / "biased"
    tmp2.mkdir()
    ff2, _ = write_repo(tmp2, bias_draw=-0.06)
    _, rep2 = run_daily(tmp2, ff2)
    assert len(rep2["qualifying_picks"]) > sharp


def test_settling_fills_in_results_and_clv(tmp_path):
    """Day 2: results have landed, so pending picks settle and CLV appears."""
    first_future, _ = write_repo(tmp_path, bias_draw=-0.06)
    _, rep1 = run_daily(tmp_path, first_future)
    assert len(rep1["qualifying_picks"]) > 0

    # Simulate the next day's fetch: the season file now contains the played matches.
    full = pd.read_csv(tmp_path / "full_season.csv")
    full.to_csv(tmp_path / "data" / "2425" / "E0.csv", index=False)

    _, rep2 = run_daily(tmp_path, pd.Timestamp(first_future) + pd.Timedelta(days=10))
    assert rep2["picks_settled_today"] > 0, "nothing settled after results landed"

    log = pd.read_csv(tmp_path / "picks.csv")
    settled = log[log["result"].isin(["win", "loss"])]
    assert len(settled) > 0
    assert settled["clv_fair_pct"].notna().any(), "CLV never computed"
    assert settled["clv_raw_pct"].notna().any()
    # raw CLV must read higher than fair CLV -- that gap is the margin
    both = settled.dropna(subset=["clv_raw_pct", "clv_fair_pct"])
    assert (both["clv_raw_pct"] > both["clv_fair_pct"]).all()
    assert "mean_clv_fair_pct" in rep2["running"]


def test_missing_data_reports_an_error_and_notifies(tmp_path):
    (tmp_path / "data").mkdir()
    r, rep = run_daily(tmp_path, "2026-09-18")
    assert r.returncode == 1
    assert rep["notify"] is True
    assert any("fixtures.csv" in e for e in rep["errors"])
