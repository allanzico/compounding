"""Data layer: fetch, cache and normalise football-data.co.uk match files.

Why this module is more than a download:
football-data.co.uk changes its column set over time. Closing odds (B365CH...)
only appear from roughly 2019/20; expected goals (HxG/AxG) only in the most
recent seasons; older files use BbAv*/BbMx* where newer ones use Avg*/Max*.
Loading seasons naively and concatenating gives you silent NaNs in exactly the
columns the method depends on. Everything here is normalised to one schema with
explicit fallback chains, and `coverage()` tells you what you actually got.

Offline use: if the machine can't reach the site (a locked-down proxy will
refuse it), download the CSVs by hand into the cache directory using the same
{season}/{div}.csv layout and everything else works unchanged.
"""
from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281"
DEFAULT_CACHE = Path(os.environ.get("CBETS_CACHE", "~/.cache/cbets")).expanduser()

# Divisions worth pulling. Main-league coverage; add more freely.
DIVISIONS = {
    "E0": "England Premier League",
    "E1": "England Championship",
    "E2": "England League One",
    "D1": "Germany Bundesliga",
    "D2": "Germany 2. Bundesliga",
    "SP1": "Spain La Liga",
    "SP2": "Spain Segunda",
    "I1": "Italy Serie A",
    "I2": "Italy Serie B",
    "F1": "France Ligue 1",
    "F2": "France Ligue 2",
    "N1": "Netherlands Eredivisie",
    "B1": "Belgium Pro League",
    "P1": "Portugal Primeira Liga",
    "T1": "Turkey Super Lig",
    "G1": "Greece Super League",
    "SC0": "Scotland Premiership",
}

# canonical -> ordered fallback list of source column names
_COLMAP: dict[str, list[str]] = {
    "home":       ["HomeTeam", "Home"],
    "away":       ["AwayTeam", "Away"],
    "fthg":       ["FTHG", "HG"],
    "ftag":       ["FTAG", "AG"],
    "ftr":        ["FTR", "Res"],
    "hxg":        ["HxG"],
    "axg":        ["AxG"],
    # Half-time goals and discipline. These were missing from the first version of
    # this map, which silently dropped the half-time, fouls and cards markets: the
    # columns simply never reached the normalised frame, so every model for them
    # was skipped without an error.
    "hthg":       ["HTHG"],
    "htag":       ["HTAG"],
    "htr":        ["HTR"],
    "hf":         ["HF"],
    "af":         ["AF"],
    "hy":         ["HY"],
    "ay":         ["AY"],
    "hr":         ["HR"],
    "ar":         ["AR"],
    "hs":         ["HS"],
    "ashots":     ["AS"],
    "hst":        ["HST"],
    "ast":        ["AST"],
    "hc":         ["HC"],
    "ac":         ["AC"],
    # opening / early 1X2
    "o_h":        ["B365H", "AvgH", "BbAvH", "MaxH", "BbMxH", "PSH", "WHH"],
    "o_d":        ["B365D", "AvgD", "BbAvD", "MaxD", "BbMxD", "PSD", "WHD"],
    "o_a":        ["B365A", "AvgA", "BbAvA", "MaxA", "BbMxA", "PSA", "WHA"],
    # closing 1X2
    "c_h":        ["B365CH", "AvgCH", "MaxCH", "PSCH"],
    "c_d":        ["B365CD", "AvgCD", "MaxCD", "PSCD"],
    "c_a":        ["B365CA", "AvgCA", "MaxCA", "PSCA"],
    # betfair exchange closing -- the sharpest public benchmark available
    "x_h":        ["BFECH", "BFCH"],
    "x_d":        ["BFECD", "BFCD"],
    "x_a":        ["BFECA", "BFCA"],
    # totals
    "o_over25":   ["B365>2.5", "Avg>2.5", "BbAv>2.5", "Max>2.5", "P>2.5"],
    "o_under25":  ["B365<2.5", "Avg<2.5", "BbAv<2.5", "Max<2.5", "P<2.5"],
    "c_over25":   ["B365C>2.5", "AvgC>2.5", "MaxC>2.5", "PC>2.5"],
    "c_under25":  ["B365C<2.5", "AvgC<2.5", "MaxC<2.5", "PC<2.5"],
    # asian handicap
    "ah_line":    ["AHh", "AHh0", "BbAHh"],
    "o_ah_h":     ["B365AHH", "AvgAHH", "MaxAHH", "GBAHH", "LBAHH"],
    "o_ah_a":     ["B365AHA", "AvgAHA", "MaxAHA", "GBAHA", "LBAHA"],
    "c_ah_line":  ["AHCh"],
    "c_ah_h":     ["B365CAHH", "AvgCAHH", "MaxCAHH"],
    "c_ah_a":     ["B365CAHA", "AvgCAHA", "MaxCAHA"],
}

_NUMERIC = [c for c in _COLMAP if c not in ("home", "away", "ftr", "htr")]

# Columns whose absence changes what you are allowed to conclude.
CRITICAL = ["c_h", "c_d", "c_a"]


def season_code(start_year: int) -> str:
    """2026 -> '2627' (the 2026/27 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _url(season: str, div: str) -> str:
    return f"{BASE_URL}/{season}/{div}.csv"


def fetch_raw(season: str, div: str, cache_dir: Path = DEFAULT_CACHE,
              refresh: bool = False, timeout: int = 30) -> pd.DataFrame | None:
    """Fetch one season/division CSV, using the on-disk cache when present."""
    cache_dir = Path(cache_dir).expanduser()
    path = cache_dir / season / f"{div}.csv"
    if path.exists() and not refresh:
        return _read_csv(path.read_bytes())

    try:
        import requests
    except ImportError:
        print("requests not installed and no cached copy; pip install requests", file=sys.stderr)
        return None

    try:
        r = requests.get(_url(season, div), timeout=timeout)
        r.raise_for_status()
    except Exception as exc:  # network blocked, 404 for a division that doesn't exist, etc.
        print(f"  ! {season}/{div}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    if len(r.content) < 200:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return _read_csv(r.content)


def _read_csv(blob: bytes) -> pd.DataFrame | None:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(blob), encoding=enc, on_bad_lines="skip")
            if len(df.columns) > 3:
                return df
        except Exception:
            continue
    return None


def _parse_dates(s: pd.Series) -> pd.Series:
    """football-data mixes dd/mm/yy and dd/mm/yyyy, sometimes within one file."""
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    missing = out.isna()
    if missing.any():
        alt = pd.to_datetime(s[missing], format="%d/%m/%y", errors="coerce")
        out.loc[missing] = alt
    missing = out.isna()
    if missing.any():
        out.loc[missing] = pd.to_datetime(s[missing], errors="coerce", dayfirst=True)
    return out


def normalise(raw: pd.DataFrame, season: str, div: str) -> pd.DataFrame:
    """Map a raw season file onto the canonical schema, filling absent columns with NaN."""
    out = pd.DataFrame(index=raw.index)
    out["season"] = season
    out["div"] = div

    date_col = next((c for c in ("Date", "date") if c in raw.columns), None)
    if date_col is None:
        return pd.DataFrame()
    out["date"] = _parse_dates(raw[date_col].astype(str))

    for canon, candidates in _COLMAP.items():
        src = next((c for c in candidates if c in raw.columns), None)
        out[canon] = raw[src] if src is not None else np.nan

    for c in _NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in ("home", "away"):
        out[c] = out[c].astype("string").str.strip()

    out = out.dropna(subset=["date", "home", "away", "fthg", "ftag"])
    out = out[out["home"].astype(str).str.len() > 0]
    return out.sort_values("date").reset_index(drop=True)


def load(divisions: list[str], start_year: int, end_year: int,
         cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> pd.DataFrame:
    """Load and normalise several divisions across several seasons.

    start_year/end_year are the *starting* year of each season, inclusive.
    load(["E0"], 2021, 2026) gives 2021/22 through 2026/27.
    """
    frames = []
    for year in range(start_year, end_year + 1):
        sc = season_code(year)
        for div in divisions:
            raw = fetch_raw(sc, div, cache_dir=cache_dir, refresh=refresh)
            if raw is None or raw.empty:
                continue
            norm = normalise(raw, sc, div)
            if not norm.empty:
                frames.append(norm)
                print(f"  {sc}/{div}: {len(norm)} matches", file=sys.stderr)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    df["match_id"] = (
        df["div"] + "|" + df["season"] + "|"
        + df["date"].dt.strftime("%Y%m%d") + "|"
        + df["home"].astype(str) + "|" + df["away"].astype(str)
    )
    return df


@dataclass
class Coverage:
    rows: int
    with_closing: int
    with_xg: int
    with_opening: int
    with_totals_closing: int
    date_min: object
    date_max: object

    def __str__(self) -> str:
        def pct(n):
            return f"{n:6d} ({100*n/self.rows:5.1f}%)" if self.rows else "     0"
        return (
            f"matches            {self.rows}\n"
            f"date range         {self.date_min} .. {self.date_max}\n"
            f"opening 1X2 odds   {pct(self.with_opening)}\n"
            f"CLOSING 1X2 odds   {pct(self.with_closing)}   <- required for the gate\n"
            f"closing O/U 2.5    {pct(self.with_totals_closing)}\n"
            f"expected goals     {pct(self.with_xg)}"
        )


def coverage(df: pd.DataFrame) -> Coverage:
    """What you actually got. Check this before trusting any backtest number."""
    if df.empty:
        return Coverage(0, 0, 0, 0, 0, None, None)
    return Coverage(
        rows=len(df),
        with_closing=int(df[["c_h", "c_d", "c_a"]].notna().all(axis=1).sum()),
        with_xg=int(df[["hxg", "axg"]].notna().all(axis=1).sum()),
        with_opening=int(df[["o_h", "o_d", "o_a"]].notna().all(axis=1).sum()),
        with_totals_closing=int(df[["c_over25", "c_under25"]].notna().all(axis=1).sum()),
        date_min=df["date"].min().date(),
        date_max=df["date"].max().date(),
    )


def outcome(df: pd.DataFrame) -> pd.Series:
    """0 = home win, 1 = draw, 2 = away win."""
    return np.select(
        [df["fthg"] > df["ftag"], df["fthg"] == df["ftag"]], [0, 1], default=2
    )
