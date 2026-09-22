"""Division code -> country and league name.

football-data.co.uk identifies competitions by short codes (E0, SP1, T1). Those
are fine for filenames and useless for finding a match in a betting app, which
lists competitions by country and full name. Everything shown to a person goes
through this map.
"""
from __future__ import annotations

__all__ = ["LEAGUES", "country_of", "name_of", "label_of", "sort_key"]

# code: (country, competition name, tier)
LEAGUES: dict[str, tuple[str, str, int]] = {
    "E0":  ("England",     "Premier League",      1),
    "E1":  ("England",     "Championship",        2),
    "E2":  ("England",     "League One",          3),
    "E3":  ("England",     "League Two",          4),
    "EC":  ("England",     "National League",     5),
    "SC0": ("Scotland",    "Premiership",         1),
    "SC1": ("Scotland",    "Championship",        2),
    "SC2": ("Scotland",    "League One",          3),
    "SC3": ("Scotland",    "League Two",          4),
    "D1":  ("Germany",     "Bundesliga",          1),
    "D2":  ("Germany",     "2. Bundesliga",       2),
    "I1":  ("Italy",       "Serie A",             1),
    "I2":  ("Italy",       "Serie B",             2),
    "SP1": ("Spain",       "La Liga",             1),
    "SP2": ("Spain",       "La Liga 2",           2),
    "F1":  ("France",      "Ligue 1",             1),
    "F2":  ("France",      "Ligue 2",             2),
    "N1":  ("Netherlands", "Eredivisie",          1),
    "B1":  ("Belgium",     "Pro League",          1),
    "P1":  ("Portugal",    "Primeira Liga",       1),
    "T1":  ("Turkey",      "Super Lig",           1),
    "G1":  ("Greece",      "Super League",        1),
}


def country_of(code: str) -> str:
    return LEAGUES.get(str(code).upper(), ("", "", 9))[0]


def name_of(code: str) -> str:
    return LEAGUES.get(str(code).upper(), ("", str(code), 9))[1]


def label_of(code: str) -> str:
    """'England · Premier League' — what a person reads."""
    c, n, _ = LEAGUES.get(str(code).upper(), ("", str(code), 9))
    return f"{c} · {n}" if c else n


def sort_key(code: str) -> tuple:
    """Country first, then tier, so a country's divisions stay together in order."""
    c, n, t = LEAGUES.get(str(code).upper(), ("zz", str(code), 9))
    return (c, t, n)
