#!/usr/bin/env python3
"""Produce the full daily projection sheet across every fixture and market.

    python generate_sheet.py --repo . --out projections.csv --horizon-days 4
"""
from __future__ import annotations

import argparse, glob, json, sys, warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cbets import projections as PJ
from daily import load_fixtures, load_history


def load_labels(d: str) -> dict:
    out = {}
    for f in glob.glob(str(Path(d) / "val_*.json")):
        try:
            j = json.load(open(f))
            if j:
                out[j["metric"]] = j["label"]
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="projections.csv")
    ap.add_argument("--summary", default="projections_summary.json")
    ap.add_argument("--labels-dir", default="validation")
    ap.add_argument("--today", default=None)
    ap.add_argument("--horizon-days", type=int, default=4)
    a = ap.parse_args()

    today = pd.Timestamp(a.today) if a.today else pd.Timestamp(date.today())
    PJ.MARKET_LABELS.update(load_labels(a.labels_dir))

    fx = load_fixtures(a.repo)
    if fx.empty:
        print("no fixtures", file=sys.stderr); return 1
    fx = fx[(fx["date"] >= today) & (fx["date"] <= today + pd.Timedelta(days=a.horizon_days))]
    divs = sorted(set(fx["div"].dropna().astype(str)) - {""})
    hist = load_history(a.repo, divs)
    if hist.empty:
        print("no history", file=sys.stderr); return 1

    rows, diags = [], []
    for div in divs:
        grp = fx[fx["div"] == div]
        if grp.empty:
            continue
        r, d = PJ.project_division(hist, grp, div, today)
        rows.extend(r); diags.append(d)
        print(f"  {div}: {len(grp)} fixtures -> {len(r)} market lines", file=sys.stderr)

    if not rows:
        print("nothing projected", file=sys.stderr); return 1
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    summary = {
        "generated": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "today": str(today.date()),
        "fixtures": int(df["fixture"].nunique()),
        "divisions": int(df["div"].nunique()),
        "market_lines": int(len(df)),
        "history_matches": int(len(hist)),
        "validation": PJ.MARKET_LABELS,
        "diagnostics": diags,
    }
    json.dump(summary, open(a.summary, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "diagnostics"},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
