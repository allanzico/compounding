"""The pick log, with the Method v2 gates enforced in code.

A pick that fails a gate is not a weaker pick. It is not a pick. The whole
value of this file is that it refuses to write one.

The log is the only asset this project builds. Results take thousands of bets
to mean anything; a well-kept log of probabilities, prices and closing prices
starts paying evidence back in dozens.
"""
from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .devig import devig

__all__ = ["Pick", "GateError", "check_gates", "append_pick", "load_log", "report_log",
           "kelly_stake", "FIELDS"]

EDGE_THRESHOLD = 0.04
BANNED_MARKETS = {
    "player_prop": "highest book margin, weakest verifiable data, source of every logged error",
    "corners": "no historical odds exist, so it can never be backtested",
    "cards": "no historical odds exist, so it can never be backtested",
    "goalscorer": "player prop by another name",
}
ALLOWED_MARKETS = {"1x2", "double_chance", "asian_handicap", "over_under"}

_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_DATE_RE = re.compile(
    r"(\d{4}-\d{1,2}-\d{1,2})|(\d{1,2}/\d{1,2}/\d{2,4})|"
    r"\b(" + "|".join(_MONTHS) + r")[a-z]*\.?\s+\d{1,2}|\d{1,2}\s+(" + "|".join(_MONTHS) + r")",
    re.IGNORECASE,
)

FIELDS = [
    "date_taken", "kickoff", "league", "fixture", "market", "selection",
    "p_model", "odds_available", "fair_odds", "edge_pct", "devigged_market_p",
    "odds_closing", "fair_closing", "clv_raw_pct", "clv_fair_pct",
    "result", "stake", "pnl", "source_note",
]


class GateError(ValueError):
    """Raised when a proposed selection fails a Method v2 gate."""


@dataclass
class Pick:
    kickoff: str
    league: str
    fixture: str
    market: str
    selection: str
    p_model: float
    odds_available: float
    source_note: str
    market_odds_all: list = field(default_factory=list)  # full market for de-vigging
    date_taken: str = ""
    odds_closing: float = float("nan")
    result: str = "pending"
    stake: float = 0.0
    pnl: float = float("nan")

    def __post_init__(self):
        if not self.date_taken:
            self.date_taken = datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def fair_odds(self) -> float:
        return 1.0 / self.p_model if self.p_model > 0 else float("nan")

    @property
    def edge(self) -> float:
        return self.p_model * self.odds_available - 1.0

    def devigged_market_p(self) -> float:
        if len(self.market_odds_all) < 2:
            return float("nan")
        try:
            p = devig(self.market_odds_all, method="shin")
        except Exception:
            return float("nan")
        idx = self._selection_index()
        return float(p[idx]) if idx is not None and idx < len(p) else float("nan")

    def _selection_index(self):
        key = self.selection.strip().lower()
        for names, i in ((("h", "home", "1"), 0), (("d", "draw", "x"), 1),
                         (("a", "away", "2"), 2), (("over", "o"), 0), (("under", "u"), 1)):
            if key in names:
                return i
        return None


def check_gates(pick: Pick, existing_slate: list[Pick] | None = None) -> list[str]:
    """Return a list of gate failures. Empty list means the pick is admissible."""
    fails = []

    # Gate 0 — sources.
    # A length check is not a source check: "looks good" is ten characters. The note
    # has to actually carry a date, because the failure mode this gate exists to stop
    # is a stale archive page being mistaken for current information.
    note = (pick.source_note or "").strip()
    if len(note) < 15 or not _DATE_RE.search(note):
        fails.append(
            "Gate 0 (sources): source_note must name the data source AND carry a date, "
            "e.g. 'football-data.co.uk 2627/E0.csv pulled 2026-09-18'. "
            f"Got {note!r}. No pick is built on recall."
        )

    # Gate 1 — a real probability, stated first
    if not (0.0 < pick.p_model < 1.0):
        fails.append(f"Gate 1 (probability): p_model must be in (0,1), got {pick.p_model!r}.")

    # Gate 2 — a real price
    if not (np.isfinite(pick.odds_available) and pick.odds_available > 1.0):
        fails.append(
            f"Gate 2 (price): odds_available must be a real posted decimal price > 1.0, "
            f"got {pick.odds_available!r}. An estimate is not a price — log it as an "
            "unpriced candidate instead."
        )

    # Gate 3 — edge
    if not fails:
        if pick.edge < EDGE_THRESHOLD:
            fails.append(
                f"Gate 3 (edge): edge {pick.edge*100:+.2f}% is below the "
                f"{EDGE_THRESHOLD*100:.0f}% threshold. Below that it is inside model error."
            )

    # Gate 4 — benchmark
    if len(pick.market_odds_all) < 2:
        fails.append(
            "Gate 4 (benchmark): market_odds_all must carry the full market so the "
            "de-vigged market probability can be computed and compared."
        )

    # Market scope
    m = pick.market.strip().lower()
    if m in BANNED_MARKETS:
        fails.append(f"Market scope: '{m}' is banned — {BANNED_MARKETS[m]}.")
    elif m not in ALLOWED_MARKETS:
        fails.append(
            f"Market scope: '{m}' is not in the allowed set {sorted(ALLOWED_MARKETS)}."
        )

    # Gate 5 — correlation
    for other in (existing_slate or []):
        if other.fixture.strip().lower() == pick.fixture.strip().lower():
            fails.append(
                f"Gate 5 (correlation): '{pick.fixture}' already has a selection in this "
                "slate. Diversify by fixture, not by market label."
            )
            break
    return fails


def kelly_stake(p: float, odds: float, bankroll: float,
                fraction: float = 0.25, cap: float = 0.02) -> float:
    """Quarter-Kelly by default, hard-capped at 2% of bankroll (Method v2 section 6)."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * odds - 1.0) / b
    return max(0.0, min(f * fraction, cap)) * bankroll


def append_pick(pick: Pick, path: str | Path, existing_slate: list[Pick] | None = None,
                force: bool = False) -> None:
    """Write a pick to the CSV log, refusing anything that fails a gate."""
    fails = check_gates(pick, existing_slate)
    if fails and not force:
        raise GateError("pick rejected:\n  - " + "\n  - ".join(fails))

    path = Path(path)
    new = not path.exists()
    mp = pick.devigged_market_p()
    row = {
        "date_taken": pick.date_taken, "kickoff": pick.kickoff, "league": pick.league,
        "fixture": pick.fixture, "market": pick.market, "selection": pick.selection,
        "p_model": round(pick.p_model, 6),
        "odds_available": pick.odds_available,
        "fair_odds": round(pick.fair_odds, 4),
        "edge_pct": round(pick.edge * 100, 3),
        "devigged_market_p": round(mp, 6) if np.isfinite(mp) else "",
        "odds_closing": pick.odds_closing if np.isfinite(pick.odds_closing) else "",
        "fair_closing": "", "clv_raw_pct": "", "clv_fair_pct": "",
        "result": pick.result, "stake": pick.stake,
        "pnl": pick.pnl if np.isfinite(pick.pnl) else "",
        "source_note": pick.source_note,
    }
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def load_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("p_model", "odds_available", "odds_closing", "devigged_market_p",
              "stake", "pnl", "edge_pct"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def report_log(path: str | Path) -> str:
    """Score the log the way Method v2 says to: calibration and CLV, not hit rate."""
    df = load_log(path)
    if df.empty:
        return "log is empty"

    settled = df[df["result"].isin(["win", "loss", "push"])]
    out = [f"picks logged     {len(df)}", f"settled          {len(settled)}"]

    has_close = df["odds_closing"].notna() & (df["odds_closing"] > 1)
    if has_close.any():
        d = df[has_close].copy()
        d["clv_raw"] = d["odds_available"] / d["odds_closing"] - 1.0
        out.append(f"mean CLV (raw)   {d['clv_raw'].mean()*100:+.2f}%  over {len(d)} picks")
        if d["devigged_market_p"].notna().any():
            e = d[d["devigged_market_p"].notna()].copy()
            e["clv_fair"] = e["odds_available"] * e["devigged_market_p"] - 1.0
            out.append(
                f"mean CLV (fair)  {e['clv_fair'].mean()*100:+.2f}%  over {len(e)} picks"
                "   <- the honest number"
            )
            n = len(e)
            if n >= 100 and e["clv_fair"].mean() <= 0:
                out.append("\nKILL CRITERION MET (Method v2 section 8): "
                           f"{n} picks logged with mean fair CLV <= 0. Stop.")
    else:
        out.append("no closing odds recorded yet — CLV cannot be computed, so the log "
                   "is not yet evidence of anything")

    if len(settled) >= 20:
        s = settled.copy()
        s["won"] = (s["result"] == "win").astype(float)
        s["bucket"] = pd.cut(s["p_model"], np.arange(0, 1.05, 0.1))
        cal = s.groupby("bucket", observed=True).agg(
            n=("won", "size"), predicted=("p_model", "mean"), actual=("won", "mean"))
        out += ["", "CALIBRATION (predicted vs actual — this is the score that matters)",
                cal.to_string(float_format=lambda v: f"{v:.3f}")]
    return "\n".join(out)
