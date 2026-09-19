"""Synthetic leagues with known ground truth.

Backtests lie. The usual way they lie is subtle look-ahead: a rating fitted on
data that includes the match being predicted, a column that is only knowable
after kickoff, an index misalignment of one row. You cannot detect any of that
by looking at real-data results, because you don't know what the right answer
was supposed to be.

So before pointing this pipeline at real money, point it at a world where the
answer IS known. This module builds one: teams with true attack and defence
ratings, matches simulated from those ratings, and a simulated bookmaker whose
sharpness, bias and margin you control.

That gives three tests worth having:
  1. Does the fitted model recover the true ratings?
  2. Against a SHARP book, does the backtest correctly report no edge?
     (If it finds one, the harness is lying.)
  3. Against a KNOWN-BIASED book, does it find the edge that was planted?
     (If it doesn't, the harness is blind.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import score_matrix, probs_1x2

__all__ = ["SyntheticLeague", "make_league"]

TARGET_TOTAL_GOALS = 2.75


@dataclass
class SyntheticLeague:
    matches: pd.DataFrame
    teams: list[str]
    true_attack: np.ndarray
    true_defence: np.ndarray
    true_intercept: float
    true_home_adv: float

    def truth_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "team": self.teams,
            "true_attack": self.true_attack,
            "true_defence": -self.true_defence,
        })


def _apply_margin(p: np.ndarray, margin: float, style: str, rng) -> np.ndarray:
    """Turn true probabilities into bookmaker implied probabilities."""
    if style == "proportional":
        return p * (1.0 + margin)
    if style == "shin":
        # Load more margin onto longshots: the observed favourite-longshot bias.
        z = margin / (1.0 + margin)
        q = p + z * np.sqrt(p * (1 - p))
        return q / q.sum() * (1.0 + margin)
    raise ValueError(f"margin_style must be proportional|shin, got {style!r}")


def make_league(
    n_teams: int = 20,
    seasons: int = 4,
    start: str = "2021-08-07",
    seed: int = 0,
    intercept: float = None,
    home_adv: float = 0.26,
    rho: float = -0.05,
    attack_sd: float = 0.33,
    defence_sd: float = 0.27,
    xg_shape: float = 6.0,
    margin: float = 0.05,
    margin_style: str = "proportional",
    book_noise_open: float = 0.045,
    book_noise_close: float = 0.020,
    open_close_corr: float = 0.90,
    bias_draw: float = 0.0,
    bias_home: float = 0.0,
) -> SyntheticLeague:
    """Simulate a league and a bookmaker.

    book_noise_open / book_noise_close / open_close_corr
        How badly the book estimates the true scoring rates, as a log-scale SD.
        Closing is sharper than opening, which is what makes CLV meaningful.

        The correlation matters more than it looks. A first version of this
        generator drew the opening and closing errors INDEPENDENTLY, and the
        validation then reported +15% CLV and +9% ROI against a book that was
        supposed to be unbeatable -- because an accurate model could
        systematically pick whichever independent perturbation happened to run
        long. Real closing lines are the opening line plus information, not a
        fresh draw. Errors here are drawn as a correlated bivariate normal so
        the close is a shrunk, better-informed version of the open.

    bias_draw / bias_home
        Plant a KNOWN mispricing, in probability points, applied to the book's
        view before margin. Leave at 0 for a sharp book that nobody can beat.
    """
    rng = np.random.default_rng(seed)
    if intercept is None:
        # Calibrate so mean total goals lands near a realistic 2.75. The variance
        # terms are the Jensen correction: E[exp(X)] = exp(mu + sigma^2/2).
        intercept = (
            np.log(TARGET_TOTAL_GOALS)
            - np.log(np.exp(home_adv) + 1.0)
            - (attack_sd ** 2 + defence_sd ** 2) / 2.0
        )

    teams = [f"Team{i:02d}" for i in range(n_teams)]
    attack = rng.normal(0, attack_sd, n_teams)
    defence = rng.normal(0, defence_sd, n_teams)
    attack -= attack.mean()
    defence -= defence.mean()

    rows = []
    day = pd.Timestamp(start)
    for season in range(seasons):
        pairs = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
        rng.shuffle(pairs)
        # ~2 rounds of fixtures spread over ~38 weeks
        per_week = max(1, len(pairs) // 38)
        for k, (h, a) in enumerate(pairs):
            date = day + pd.Timedelta(days=7 * (k // per_week))
            lam = np.exp(intercept + attack[h] + defence[a] + home_adv)
            mu = np.exp(intercept + attack[a] + defence[h])

            m = score_matrix(lam, mu, rho, 10)
            flat = m.ravel()
            pick = rng.choice(flat.size, p=flat / flat.sum())
            gh, ga = divmod(pick, m.shape[1])

            # xG: an unbiased but much less noisy signal of the true rate
            hxg = rng.gamma(xg_shape, lam / xg_shape)
            axg = rng.gamma(xg_shape, mu / xg_shape)

            # Correlated open/close errors: close = shrunk open + a little new info.
            so, sc, rc = book_noise_open, book_noise_close, open_close_corr
            beta = (rc * sc / so) if so > 0 else 0.0
            resid = sc * np.sqrt(max(1.0 - rc ** 2, 0.0))
            e_open_l, e_open_m = rng.normal(0, so, 2)
            e_close_l = beta * e_open_l + rng.normal(0, resid)
            e_close_m = beta * e_open_m + rng.normal(0, resid)

            odds = {}
            for tag, (el, em) in (("o", (e_open_l, e_open_m)), ("c", (e_close_l, e_close_m))):
                bl = lam * np.exp(el)
                bm = mu * np.exp(em)
                p = probs_1x2(score_matrix(bl, bm, rho, 10))
                if bias_draw or bias_home:
                    p = p + np.array([bias_home, bias_draw, -bias_home - bias_draw])
                    p = np.clip(p, 0.01, 0.98)
                    p = p / p.sum()
                q = _apply_margin(p, margin, margin_style, rng)
                odds[f"{tag}_h"], odds[f"{tag}_d"], odds[f"{tag}_a"] = 1.0 / q

                mm = score_matrix(bl, bm, rho, 10)
                idx = np.arange(mm.shape[0])
                tot = idx[:, None] + idx[None, :]
                po = float(mm[tot > 2.5].sum())
                qo = _apply_margin(np.array([po, 1 - po]), margin, margin_style, rng)
                odds[f"{tag}_over25"], odds[f"{tag}_under25"] = 1.0 / qo

            rows.append(dict(
                date=date, div="SYN", season=f"S{season}",
                home=teams[h], away=teams[a],
                fthg=int(gh), ftag=int(ga),
                ftr="H" if gh > ga else ("D" if gh == ga else "A"),
                hxg=hxg, axg=axg, **odds,
            ))
        day = day + pd.Timedelta(days=365)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["match_id"] = (
        df["div"] + "|" + df["season"] + "|" + df["date"].dt.strftime("%Y%m%d")
        + "|" + df["home"] + "|" + df["away"]
    )
    for c in ("x_h", "x_d", "x_a", "ah_line", "c_ah_line", "c_ah_h", "c_ah_a",
              "o_ah_h", "o_ah_a", "hs", "ashots", "hst", "ast", "hc", "ac"):
        df[c] = np.nan

    return SyntheticLeague(df, teams, attack, defence, float(intercept), float(home_adv))
