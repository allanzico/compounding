#!/usr/bin/env python3
"""Fetch football-data.co.uk files into data/ — runs on GitHub Actions.

This exists because the Claude sandbox that runs the daily analysis cannot reach
football-data.co.uk (its egress proxy refuses the host). GitHub's runners have
unrestricted network, so they do the fetching and commit the result; the daily
task then reads the CSVs from raw.githubusercontent.com, which IS reachable.

Two kinds of file:
  fixtures.csv        upcoming matches with live-ish odds. This is what the
                      daily scan prices. Overwritten every run.
  {season}/{div}.csv  completed matches with results AND closing odds. This is
                      what the model trains on and what settles CLV.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.football-data.co.uk"
OUT = Path(__file__).resolve().parent.parent / "data"

DIVISIONS = os.environ.get(
    "CBETS_DIVISIONS",
    "E0,E1,E2,D1,D2,SP1,SP2,I1,I2,F1,F2,N1,B1,P1,G1,T1,SC0",
).split(",")

# Closing odds only exist from roughly 2019/20, and they are what the gate needs.
START_YEAR = int(os.environ.get("CBETS_START_YEAR", "2019"))
END_YEAR = int(os.environ.get("CBETS_END_YEAR", "2026"))

UA = {"User-Agent": "Mozilla/5.0 (compatible; cbets-fetch/2.0)"}


def season_code(y: int) -> str:
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def get(url: str, tries: int = 3) -> bytes | None:
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60, headers=UA)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as exc:
            if i == tries - 1:
                print(f"  ! {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(2 * (i + 1))
    return None


def write(path: Path, blob: bytes) -> bool:
    """Write only when the content actually changed, to keep the diff honest."""
    if len(blob) < 200:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == blob:
        return False
    path.write_bytes(blob)
    return True


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    changed = 0

    blob = get(f"{BASE}/fixtures.csv")
    if blob and write(OUT / "fixtures.csv", blob):
        changed += 1
        print(f"  fixtures.csv  {len(blob):,} bytes")
    elif not blob:
        print("  ! fixtures.csv could not be fetched", file=sys.stderr)

    # Only the current and previous season change; older ones are frozen, so
    # re-fetch them once and then leave them alone.
    for year in range(START_YEAR, END_YEAR + 1):
        sc = season_code(year)
        for div in DIVISIONS:
            div = div.strip().upper()
            if not div:
                continue
            path = OUT / sc / f"{div}.csv"
            if path.exists() and year < END_YEAR - 1:
                continue
            blob = get(f"{BASE}/mmz4281/{sc}/{div}.csv")
            if blob and write(path, blob):
                changed += 1
                print(f"  {sc}/{div}.csv  {len(blob):,} bytes")
            time.sleep(0.3)

    print(f"\n{changed} file(s) changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
