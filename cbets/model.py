"""Dixon-Coles goals model.

The idea in one line: every team gets an attack rating and a defence rating,
there is a global home advantage, and goals are Poisson around the rates those
imply. Dixon & Coles (1997) add two things that matter -- a correction for
low-scoring scorelines (0-0, 1-0, 0-1, 1-1 happen more often than independent
Poisson predicts) and exponential time decay so last month counts more than
last year.

    log lambda_home = intercept + attack[home] + defence[away] + home_adv
    log lambda_away = intercept + attack[away] + defence[home]

Note the sign convention: `defence[x]` is added to the rate of goals scored
AGAINST x, so a HIGHER defence value means a leakier defence. `team_table()`
prints it the intuitive way round.

Fitting is two-stage, which is both faster and numerically better behaved than
optimising everything jointly:

  1. Ridge-penalised weighted Poisson regression for intercept / attack /
     defence / home advantage, with an analytic gradient.
  2. A one-dimensional fit of the low-score correlation term `rho`, holding the
     ratings fixed. Dixon & Coles note rho is near-orthogonal to the ratings,
     so this costs almost nothing in fit quality and a lot less in time.

The ridge penalty does double duty: it identifies the model (attack and defence
are otherwise only determined up to a constant) and it shrinks teams with few
matches toward the league average, which is exactly what you want in August.

Fitting on xG rather than goals: the Poisson log-likelihood y*log(lambda) -
lambda is perfectly well defined for continuous y, so xG can be used directly
as a quasi-Poisson response. xG carries less per-match noise than goals, so the
ratings converge faster. rho is always fitted on actual goals, because it
describes the scoreline distribution, not the rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

__all__ = ["DixonColes", "fit", "RHO_BOUNDS"]

RHO_BOUNDS = (-0.25, 0.25)
_EPS = 1e-9


@dataclass
class DixonColes:
    teams: list[str]
    intercept: float
    home_adv: float
    attack: np.ndarray
    defence: np.ndarray
    rho: float
    max_goals: int = 10
    n_train: int = 0
    trained_through: object = None
    index: dict = field(default_factory=dict, repr=False)

    # ---------- rates ----------

    def rates(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals for (home, away). Unknown teams fall back to league average."""
        ih, ia = self.index.get(home), self.index.get(away)
        ah = self.attack[ih] if ih is not None else 0.0
        dh = self.defence[ih] if ih is not None else 0.0
        aa = self.attack[ia] if ia is not None else 0.0
        da = self.defence[ia] if ia is not None else 0.0
        lam = np.exp(self.intercept + ah + da + self.home_adv)
        mu = np.exp(self.intercept + aa + dh)
        return float(lam), float(mu)

    # ---------- scoreline distribution ----------

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.rates(home, away)
        return score_matrix(lam, mu, self.rho, self.max_goals)

    def probs_1x2(self, home: str, away: str) -> np.ndarray:
        return probs_1x2(self.score_matrix(home, away))

    def prob_over(self, home: str, away: str, line: float = 2.5) -> float:
        return prob_over(self.score_matrix(home, away), line)

    def ah_home(self, home: str, away: str, line: float) -> tuple[float, float, float]:
        return ah_home(self.score_matrix(home, away), line)

    # ---------- inspection ----------

    def team_table(self) -> pd.DataFrame:
        """Ratings in human terms. Higher attack = scores more; higher defence = concedes fewer."""
        return (
            pd.DataFrame({
                "team": self.teams,
                "attack": self.attack,
                "defence": -self.defence,   # flipped so higher = better defence
            })
            .assign(strength=lambda d: d["attack"] + d["defence"])
            .sort_values("strength", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------- scorelines

def _tau(matrix_shape: int, lam: float, mu: float, rho: float) -> np.ndarray:
    t = np.ones((matrix_shape, matrix_shape))
    t[0, 0] = 1.0 - lam * mu * rho
    t[0, 1] = 1.0 + lam * rho
    t[1, 0] = 1.0 + mu * rho
    t[1, 1] = 1.0 - rho
    return np.clip(t, _EPS, None)


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """P(home scores i, away scores j) for i, j in 0..max_goals."""
    n = max_goals + 1
    ph = poisson.pmf(np.arange(n), lam)
    pa = poisson.pmf(np.arange(n), mu)
    m = np.outer(ph, pa) * _tau(n, lam, mu, rho)
    s = m.sum()
    return m / s if s > 0 else m


def probs_1x2(m: np.ndarray) -> np.ndarray:
    idx = np.arange(m.shape[0])
    diff = idx[:, None] - idx[None, :]
    return np.array([m[diff > 0].sum(), m[diff == 0].sum(), m[diff < 0].sum()])


def prob_over(m: np.ndarray, line: float = 2.5) -> float:
    idx = np.arange(m.shape[0])
    total = idx[:, None] + idx[None, :]
    return float(m[total > line].sum())


def ah_home(m: np.ndarray, line: float) -> tuple[float, float, float]:
    """(win, push, loss) for the HOME side on Asian handicap `line`.

    line = -0.5 means home gives half a goal; +1.0 means home receives one goal.
    Quarter lines (-0.25, -0.75, ...) split the stake over the two neighbouring
    lines, which is what the bookmaker actually does.
    """
    q = round(line * 4) / 4
    if abs(q * 2 - round(q * 2)) > 1e-9:  # quarter line
        lo, hi = q - 0.25, q + 0.25
        a = ah_home(m, lo)
        b = ah_home(m, hi)
        return tuple((np.array(a) + np.array(b)) / 2.0)

    idx = np.arange(m.shape[0])
    margin = idx[:, None] - idx[None, :] + q
    win = float(m[margin > 1e-9].sum())
    push = float(m[np.abs(margin) <= 1e-9].sum())
    return win, push, 1.0 - win - push


# ---------------------------------------------------------------- fitting

def _response(df: pd.DataFrame, kind: str, xg_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (home_response, away_response). Falls back to goals where xG is absent."""
    gh = df["fthg"].to_numpy(float)
    ga = df["ftag"].to_numpy(float)
    if kind == "goals":
        return gh, ga

    xh = df["hxg"].to_numpy(float) if "hxg" in df else np.full(len(df), np.nan)
    xa = df["axg"].to_numpy(float) if "axg" in df else np.full(len(df), np.nan)
    have = np.isfinite(xh) & np.isfinite(xa)

    if kind == "xg":
        return np.where(have, xh, gh), np.where(have, xa, ga)
    if kind == "blend":
        w = float(xg_weight)
        bh = np.where(have, w * xh + (1 - w) * gh, gh)
        ba = np.where(have, w * xa + (1 - w) * ga, ga)
        return bh, ba
    raise ValueError(f"response must be goals|xg|blend, got {kind!r}")


def fit(df: pd.DataFrame, as_of=None, half_life_days: float = 180.0,
        ridge: float = 0.02, response: str = "blend", xg_weight: float = 0.6,
        max_goals: int = 10, fit_rho: bool = True) -> DixonColes:
    """Fit on every match in `df` dated strictly before `as_of`.

    The strict inequality is the whole point of walk-forward: a model must never
    see the match it is predicting. Pass `as_of` as the kickoff date.
    """
    if as_of is not None:
        df = df[df["date"] < pd.Timestamp(as_of)]
    df = df.dropna(subset=["home", "away", "fthg", "ftag"])
    if df.empty:
        raise ValueError("no training matches")

    teams = sorted(set(df["home"].astype(str)) | set(df["away"].astype(str)))
    index = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = df["home"].astype(str).map(index).to_numpy()
    ai = df["away"].astype(str).map(index).to_numpy()
    yh, ya = _response(df, response, xg_weight)

    ref = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    age = (ref - df["date"]).dt.total_seconds().to_numpy() / 86400.0
    w_match = 0.5 ** (np.clip(age, 0, None) / float(half_life_days))

    # stack: one row per (attacking team, defending team, is_home)
    att = np.concatenate([hi, ai])
    dfn = np.concatenate([ai, hi])
    y = np.concatenate([yh, ya])
    w = np.concatenate([w_match, w_match])
    is_home = np.concatenate([np.ones(len(df)), np.zeros(len(df))])

    def unpack(p):
        return p[0], p[1], p[2:2 + n], p[2 + n:2 + 2 * n]

    def nll_and_grad(p):
        inter, hadv, a, d = unpack(p)
        eta = inter + a[att] + d[dfn] + hadv * is_home
        eta = np.clip(eta, -10, 5)
        lam = np.exp(eta)
        nll = -np.sum(w * (y * eta - lam)) + ridge * (a @ a + d @ d)

        r = w * (y - lam)
        g = np.empty_like(p)
        g[0] = -r.sum()
        g[1] = -np.sum(r * is_home)
        g[2:2 + n] = -np.bincount(att, weights=r, minlength=n) + 2 * ridge * a
        g[2 + n:] = -np.bincount(dfn, weights=r, minlength=n) + 2 * ridge * d
        return nll, g

    x0 = np.zeros(2 + 2 * n)
    x0[0] = np.log(max(y.mean(), 0.05))
    x0[1] = 0.25
    res = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10})
    inter, hadv, a, d = unpack(res.x)

    rho = 0.0
    if fit_rho:
        rho = _fit_rho(df, index, inter, hadv, a, d, w_match)

    return DixonColes(
        teams=teams, intercept=float(inter), home_adv=float(hadv),
        attack=a, defence=d, rho=float(rho), max_goals=max_goals,
        n_train=len(df), trained_through=df["date"].max(), index=index,
    )


def _fit_rho(df, index, inter, hadv, a, d, w) -> float:
    """One-dimensional fit of the low-score correction, on ACTUAL goals."""
    hi = df["home"].astype(str).map(index).to_numpy()
    ai = df["away"].astype(str).map(index).to_numpy()
    lam = np.exp(inter + a[hi] + d[ai] + hadv)
    mu = np.exp(inter + a[ai] + d[hi])
    gh = df["fthg"].to_numpy(int)
    ga = df["ftag"].to_numpy(int)

    m00 = (gh == 0) & (ga == 0)
    m01 = (gh == 0) & (ga == 1)
    m10 = (gh == 1) & (ga == 0)
    m11 = (gh == 1) & (ga == 1)
    if not (m00 | m01 | m10 | m11).any():
        return 0.0

    def neg_ll(rho: float) -> float:
        t = np.ones(len(df))
        t[m00] = 1.0 - lam[m00] * mu[m00] * rho
        t[m01] = 1.0 + lam[m01] * rho
        t[m10] = 1.0 + mu[m10] * rho
        t[m11] = 1.0 - rho
        return -float(np.sum(w * np.log(np.clip(t, _EPS, None))))

    out = minimize_scalar(neg_ll, bounds=RHO_BOUNDS, method="bounded")
    return float(np.clip(out.x, *RHO_BOUNDS))
