#!/usr/bin/env python3
"""Build the self-contained dashboard page from the projection sheet.

Runs inside the GitHub Action, not in a Claude session. The session that
publishes the page must not have to transform anything: it downloads the
finished HTML and republishes it. That is deliberate — the cloud sandbox
refuses to execute code cloned from a repo ("Code from External"), so any
compute in the daily run is a blocker nobody is awake to approve at 06:00 UTC.
All of it happens here instead, on GitHub's runner, and gets committed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "dashboard" / "template.html"
OUT = ROOT / "dashboard" / "index.html"


def n(v, nd=2):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, nd)


def main() -> int:
    proj = ROOT / "projections.csv"
    summ = ROOT / "summary.json"
    if not proj.exists() or not summ.exists():
        print("projections.csv or summary.json missing", file=sys.stderr)
        return 1

    d = pd.read_csv(proj)
    s = json.load(open(summ))

    groups = sorted(d["group"].unique())
    markets = sorted(d["market"].unique())
    sels = sorted(d["selection"].unique())
    gi = {g: i for i, g in enumerate(groups)}
    mi = {m: i for i, m in enumerate(markets)}
    si = {x: i for i, x in enumerate(sels)}

    fx = (d.groupby(["fixture"], sort=False)
            .agg(date=("date", "first"), division=("div", "first"),
                 home=("home", "first"), away=("away", "first"),
                 conf=("confidence", "first"), xh=("xg_home", "first"),
                 xa=("xg_away", "first"), es=("exp_shots", "first"),
                 eo=("exp_sot", "first"), ec=("exp_corners", "first"),
                 ef=("exp_fouls", "first"))
            .reset_index())
    fi = {f: i for i, f in enumerate(fx["fixture"])}

    lines = [[fi[r.fixture], gi[r.group], mi[r.market], si[r.selection],
              n(r.prob, 4), n(r.fair_odds), n(getattr(r, "price", None)),
              n(getattr(r, "edge_pct", None), 1)] for r in d.itertuples()]

    payload = {
        "generated": s["generated"], "today": s["today"],
        "counts": {"fixtures": int(s["fixtures"]), "divisions": int(s["divisions"]),
                   "lines": int(s["market_lines"]), "history": int(s["history_matches"])},
        "validation": s.get("validation", {}),
        "groups": groups, "markets": markets, "selections": sels,
        "fixtures": [{"f": r.fixture, "d": r.date, "v": r.division, "h": r.home, "a": r.away,
                      "c": n(r.conf, 0), "xh": n(r.xh), "xa": n(r.xa), "es": n(r.es, 1),
                      "eo": n(r.eo, 1), "ec": n(r.ec, 1), "ef": n(r.ef, 1)}
                     for r in fx.itertuples()],
        "lines": lines,
    }

    html = TPL.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(payload, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"dashboard/index.html  {OUT.stat().st_size:,} bytes  "
          f"{payload['counts']['fixtures']} fixtures  {payload['counts']['lines']:,} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
