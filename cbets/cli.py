"""Command line entry points.

    python -m cbets.cli fetch     --divisions E0,D1,SP1,I1,F1,N1 --from 2019 --to 2026
    python -m cbets.cli backtest  --divisions E0 --from 2019 --to 2026 --eval-from 2021-08-01
    python -m cbets.cli validate
    python -m cbets.cli log-report --path picks.csv
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import data as D
from .backtest import (BacktestResult, bet_metrics, evaluate, report, simulate_bets,
                       walk_forward)


def _divs(s: str) -> list[str]:
    return [d.strip().upper() for d in s.split(",") if d.strip()]


def cmd_fetch(a) -> int:
    df = D.load(_divs(a.divisions), a.start_year, a.end_year, refresh=a.refresh)
    if df.empty:
        print("\nNo data loaded. If the network is blocked, download the CSVs by hand into")
        print(f"  {D.DEFAULT_CACHE}/<season>/<div>.csv   e.g. .../2627/E0.csv")
        print("using the same layout as https://www.football-data.co.uk/mmz4281/2627/E0.csv")
        return 1
    print()
    print(D.coverage(df))
    return 0


def cmd_backtest(a) -> int:
    df = D.load(_divs(a.divisions), a.start_year, a.end_year, refresh=False)
    if df.empty:
        print("no data — run `fetch` first", file=sys.stderr)
        return 1
    cov = D.coverage(df)
    print(cov)
    if cov.with_closing < a.min_matches:
        print(f"\nWARNING: only {cov.with_closing} matches have closing odds. The gate needs "
              f"{a.min_matches}. Closing odds only exist from roughly the 2019/20 season, so "
              "add seasons or divisions.\n")

    eval_from = pd.Timestamp(a.eval_from) if a.eval_from else df["date"].min() + pd.Timedelta(days=400)
    pred = walk_forward(
        df, start_date=eval_from, refit_days=a.refit_days,
        min_train_matches=a.min_train, devig_method=a.devig,
        response=a.response, half_life_days=a.half_life, ridge=a.ridge,
    )
    if pred.empty:
        print("no predictions produced", file=sys.stderr)
        return 1

    met = evaluate(pred)
    bets = simulate_bets(pred, edge_threshold=a.edge, price=a.price,
                         kelly_fraction=a.kelly)
    res = BacktestResult(pred, met, bets, bet_metrics(bets))
    print()
    print(report(res, min_matches=a.min_matches))
    if a.out:
        pred.to_csv(a.out, index=False)
        print(f"\npredictions written to {a.out}")
    return 0 if res.passed else 2


def cmd_validate(a) -> int:
    import subprocess
    from pathlib import Path
    script = Path(__file__).resolve().parent.parent / "validate.py"
    return subprocess.call([sys.executable, str(script)])


def cmd_log_report(a) -> int:
    from .picks import report_log
    print(report_log(a.path))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cbets", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download and cache season files, then report coverage")
    f.add_argument("--divisions", default="E0")
    f.add_argument("--from", dest="start_year", type=int, default=2019)
    f.add_argument("--to", dest="end_year", type=int, default=2026)
    f.add_argument("--refresh", action="store_true")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="walk-forward backtest and the go/no-go gate")
    b.add_argument("--divisions", default="E0")
    b.add_argument("--from", dest="start_year", type=int, default=2019)
    b.add_argument("--to", dest="end_year", type=int, default=2026)
    b.add_argument("--eval-from", default=None, help="date to start evaluating, e.g. 2021-08-01")
    b.add_argument("--refit-days", type=int, default=7)
    b.add_argument("--min-train", type=int, default=380)
    b.add_argument("--min-matches", type=int, default=1000)
    b.add_argument("--response", default="blend", choices=["goals", "xg", "blend"])
    b.add_argument("--half-life", type=float, default=180.0)
    b.add_argument("--ridge", type=float, default=0.02)
    b.add_argument("--devig", default="shin", choices=["shin", "power", "multiplicative"])
    b.add_argument("--edge", type=float, default=0.04)
    b.add_argument("--price", default="o", choices=["o", "c"])
    b.add_argument("--kelly", type=float, default=0.0)
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_backtest)

    v = sub.add_parser("validate", help="run the synthetic ground-truth validation")
    v.set_defaults(func=cmd_validate)

    l = sub.add_parser("log-report", help="score the pick log: calibration and CLV")
    l.add_argument("--path", default="picks.csv")
    l.set_defaults(func=cmd_log_report)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
