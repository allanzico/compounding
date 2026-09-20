"""Generic count models for the markets beyond goals.

Shots, shots on target, corners, fouls and cards are all per-team counts, so they
take the same structure as the goals model: each team gets a "produces" rating and
a "concedes" rating, plus a home effect, fitted by weighted Poisson regression
with time decay.

The one thing that must NOT be copied from the goals model is the Poisson
assumption for the spread. Measured over 43,246 matches:

    metric            mean/match   var/mean
    goals                  2.68       1.09   <- Poisson is fine
    halftime goals         1.20       1.03   <- Poisson is fine
    yellow cards           4.14       0.97   <- Poisson is fine
    shots                 24.38       2.15   <- NOT Poisson
    corners                9.76       1.61   <- NOT Poisson
    fouls                 25.12       1.36   <- NOT Poisson
    shots on target        8.51       1.40   <- NOT Poisson

A Poisson has variance equal to its mean. Shots vary more than twice that, so a
Poisson would put far too little probability in the tails and produce confident,
wrong numbers on exactly the over/under lines people bet. Those metrics get a
negative binomial whose dispersion is estimated from the data.

Tackles are not in this data source at all. Fouls are the nearest available
proxy and are a different thing; this module does not pretend otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

__all__ = ["CountModel", "fit_counts", "METRICS"]

# canonical name -> (home column, away column, is it overdispersed)
METRICS = {
    "goals":      ("fthg", "ftag"),
    "ht_goals":   ("hthg", "htag"),
    "shots":      ("hs", "ashots"),
    "sot":        ("hst", "ast"),
    "corners":    ("hc", "ac"),
    "fouls":      ("hf", "af"),
    "yellows":    ("hy", "ay"),
    "reds":       ("hr", "ar"),
}


@dataclass
class CountModel:
    metric: str
    teams: list[str]
    intercept: float
    home_adv: float
    produce: np.ndarray
    concede: np.ndarray
    phi: float                      # variance / mean; 1.0 means Poisson
    n_train: int = 0
    index: dict = field(default_factory=dict, repr=False)
    team_weight: dict = field(default_factory=dict, repr=False)

    def effective_matches(self, team: str) -> float:
        return float(self.team_weight.get(team, 0.0))

    def knows(self, *teams: str) -> bool:
        return all(t in self.index for t in teams)

    def rates(self, home: str, away: str) -> tuple[float, float]:
        ih, ia = self.index.get(home), self.index.get(away)
        ph = self.produce[ih] if ih is not None else 0.0
        ch = self.concede[ih] if ih is not None else 0.0
        pa = self.produce[ia] if ia is not None else 0.0
        ca = self.concede[ia] if ia is not None else 0.0
        return (float(np.exp(self.intercept + ph + ca + self.home_adv)),
                float(np.exp(self.intercept + pa + ch)))

    # ---------- distributions ----------

    def _dist(self, mean: float):
        """Poisson when the data is not overdispersed, negative binomial when it is."""
        if self.phi <= 1.02:
            return poisson(mean)
        r = mean / (self.phi - 1.0)          # NB: var = mean * phi
        p = r / (r + mean)
        return nbinom(r, p)

    def prob_team_over(self, home: str, away: str, line: float, side: str = "home") -> float:
        lam, mu = self.rates(home, away)
        m = lam if side == "home" else mu
        return float(self._dist(m).sf(np.floor(line)))

    def prob_total_over(self, home: str, away: str, line: float) -> float:
        lam, mu = self.rates(home, away)
        return float(self._dist(lam + mu).sf(np.floor(line)))

    def expected(self, home: str, away: str) -> tuple[float, float, float]:
        lam, mu = self.rates(home, away)
        return lam, mu, lam + mu


def _dispersion(y: np.ndarray, fitted: np.ndarray) -> float:
    """Pearson dispersion: mean squared standardised residual."""
    f = np.clip(fitted, 1e-6, None)
    return float(np.mean((y - f) ** 2 / f))


def fit_counts(df: pd.DataFrame, metric: str, as_of=None, half_life_days: float = 180.0,
               ridge: float = 1.0) -> CountModel:
    """Fit one count metric on every match dated strictly before `as_of`."""
    try:
        hcol, acol = METRICS[metric]
    except KeyError:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    if hcol not in df.columns or acol not in df.columns:
        raise ValueError(f"{metric}: columns {hcol}/{acol} are not in this data")

    if as_of is not None:
        df = df[df["date"] < pd.Timestamp(as_of)]
    df = df.dropna(subset=["home", "away", hcol, acol])
    if len(df) < 100:
        raise ValueError(f"{metric}: only {len(df)} usable training matches")

    teams = sorted(set(df["home"].astype(str)) | set(df["away"].astype(str)))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df["home"].astype(str).map(index).to_numpy()
    ai = df["away"].astype(str).map(index).to_numpy()
    yh = df[hcol].to_numpy(float)
    ya = df[acol].to_numpy(float)

    ref = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    age = (ref - df["date"]).dt.total_seconds().to_numpy() / 86400.0
    wm = 0.5 ** (np.clip(age, 0, None) / float(half_life_days))

    prod = np.concatenate([hi, ai])
    conc = np.concatenate([ai, hi])
    y = np.concatenate([yh, ya])
    w = np.concatenate([wm, wm])
    is_home = np.concatenate([np.ones(len(df)), np.zeros(len(df))])

    def nll_grad(p):
        inter, hadv, a, d = p[0], p[1], p[2:2 + n], p[2 + n:]
        eta = np.clip(inter + a[prod] + d[conc] + hadv * is_home, -10, 6)
        lam = np.exp(eta)
        nll = -np.sum(w * (y * eta - lam)) + ridge * (a @ a + d @ d)
        r = w * (y - lam)
        g = np.empty_like(p)
        g[0] = -r.sum()
        g[1] = -np.sum(r * is_home)
        g[2:2 + n] = -np.bincount(prod, weights=r, minlength=n) + 2 * ridge * a
        g[2 + n:] = -np.bincount(conc, weights=r, minlength=n) + 2 * ridge * d
        return nll, g

    x0 = np.zeros(2 + 2 * n)
    x0[0] = np.log(max(y.mean(), 0.05))
    res = minimize(nll_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 400, "ftol": 1e-10})
    inter, hadv, a, d = res.x[0], res.x[1], res.x[2:2 + n], res.x[2 + n:]

    fitted = np.exp(np.clip(inter + a[prod] + d[conc] + hadv * is_home, -10, 6))
    phi = max(1.0, _dispersion(y, fitted))

    tw = {t: float(wm[(hi == i) | (ai == i)].sum()) for i, t in enumerate(teams)}
    return CountModel(metric=metric, teams=teams, intercept=float(inter),
                      home_adv=float(hadv), produce=a, concede=d, phi=float(phi),
                      n_train=len(df), index=index, team_weight=tw)
