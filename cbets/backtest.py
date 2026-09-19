"""Walk-forward backtest, and the gate that decides whether money moves.

The gate, restated from Method v2 section 5:

    Compute the model's log loss against actual outcomes. Compute the de-vigged
    CLOSING line's log loss over the same matches. If the model does not beat
    the closing line, there is no edge and nothing is staked.

Log loss is the right scorer here because it is *proper*: it is minimised by
reporting your honest probabilities, so it cannot be gamed by shading picks
toward whatever produces flattering hit rates. Beating the closing line on log
loss is a genuinely hard test, and that is the point -- it is the test that
stops you funding a hunch.

Everything is strictly walk-forward. Each refit sees only matches dated before
the window it predicts. If you change one thing in this file, do not change
that.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .devig import devig
from .model import fit as fit_model
from .model import probs_1x2, score_matrix

__all__ = ["walk_forward", "evaluate", "simulate_bets", "gate", "report", "BacktestResult"]

_EPS = 1e-12
OUTCOMES = ("H", "D", "A")


def _outcome_idx(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["fthg"] > df["ftag"], 0, np.where(df["fthg"] == df["ftag"], 1, 2))


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], _EPS, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


# ------------------------------------------------------------- walk-forward

def walk_forward(df: pd.DataFrame, start_date, refit_days: int = 7,
                 min_train_matches: int = 300, devig_method: str = "shin",
                 verbose: bool = True, **fit_kwargs) -> pd.DataFrame:
    """Refit every `refit_days` and predict the matches in the following window.

    Returns one row per predicted match: model probabilities, de-vigged market
    probabilities (opening and closing), and the realised outcome.
    """
    df = df.sort_values("date").reset_index(drop=True)
    start = pd.Timestamp(start_date)
    end = df["date"].max()

    rows = []
    cursor = start
    n_fits = 0
    while cursor <= end:
        nxt = cursor + pd.Timedelta(days=refit_days)
        window = df[(df["date"] >= cursor) & (df["date"] < nxt)]
        if window.empty:
            cursor = nxt
            continue

        train = df[df["date"] < cursor]
        if len(train) < min_train_matches:
            cursor = nxt
            continue

        try:
            model = fit_model(train, as_of=cursor, **fit_kwargs)
        except Exception as exc:
            if verbose:
                print(f"  ! fit failed at {cursor.date()}: {exc}")
            cursor = nxt
            continue
        n_fits += 1

        for r in window.itertuples():
            lam, mu = model.rates(r.home, r.away)
            m = score_matrix(lam, mu, model.rho, model.max_goals)
            p = probs_1x2(m)
            idx = np.arange(m.shape[0])
            p_over = float(m[(idx[:, None] + idx[None, :]) > 2.5].sum())

            row = {
                "date": r.date, "div": r.div, "home": r.home, "away": r.away,
                "fthg": r.fthg, "ftag": r.ftag,
                "lam": lam, "mu": mu,
                "p_h": p[0], "p_d": p[1], "p_a": p[2], "p_over25": p_over,
                "n_train": model.n_train,
            }
            for tag in ("o", "c"):
                trio = [getattr(r, f"{tag}_h", np.nan), getattr(r, f"{tag}_d", np.nan),
                        getattr(r, f"{tag}_a", np.nan)]
                row[f"{tag}_h"], row[f"{tag}_d"], row[f"{tag}_a"] = trio
                if all(np.isfinite(x) and x > 1 for x in trio):
                    q = devig(trio, method=devig_method)
                    row[f"m{tag}_h"], row[f"m{tag}_d"], row[f"m{tag}_a"] = q
                else:
                    row[f"m{tag}_h"] = row[f"m{tag}_d"] = row[f"m{tag}_a"] = np.nan
            rows.append(row)

        cursor = nxt

    out = pd.DataFrame(rows)
    if verbose:
        print(f"  walk-forward: {n_fits} refits, {len(out)} matches predicted")
    if not out.empty:
        out["y"] = _outcome_idx(out)
    return out


# ----------------------------------------------------------------- scoring

def calibration(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability table: does the 60% bucket actually win ~60% of the time?"""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    p, hit = probs.ravel(), onehot.ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        rows.append({
            "bucket": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": int(sel.sum()),
            "predicted": float(p[sel].mean()),
            "actual": float(hit[sel].mean()),
            "gap": float(hit[sel].mean() - p[sel].mean()),
        })
    return pd.DataFrame(rows)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0):
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def evaluate(pred: pd.DataFrame) -> dict:
    """Score the model against the de-vigged closing line on the same matches."""
    have = pred[["mc_h", "mc_d", "mc_a"]].notna().all(axis=1)
    d = pred[have]
    if d.empty:
        return {"n": 0, "error": "no matches with closing odds — the gate cannot be run"}

    y = d["y"].to_numpy()
    pm = d[["p_h", "p_d", "p_a"]].to_numpy()
    pc = d[["mc_h", "mc_d", "mc_a"]].to_numpy()

    ll_m, ll_c = log_loss(pm, y), log_loss(pc, y)
    per_match = -np.log(np.clip(pc[np.arange(len(y)), y], _EPS, 1)) + np.log(
        np.clip(pm[np.arange(len(y)), y], _EPS, 1))
    lo, hi = _bootstrap_ci(per_match)

    out = {
        "n": int(len(d)),
        "logloss_model": ll_m,
        "logloss_closing": ll_c,
        "logloss_delta": ll_c - ll_m,          # positive = model better
        "logloss_delta_ci": (lo, hi),
        "brier_model": brier(pm, y),
        "brier_closing": brier(pc, y),
        "calibration": calibration(pm, y),
    }
    have_o = d[["mo_h", "mo_d", "mo_a"]].notna().all(axis=1)
    if have_o.any():
        po = d.loc[have_o, ["mo_h", "mo_d", "mo_a"]].to_numpy()
        out["logloss_opening"] = log_loss(po, d.loc[have_o, "y"].to_numpy())
    return out


# ------------------------------------------------------------------ betting

def simulate_bets(pred: pd.DataFrame, edge_threshold: float = 0.04,
                  price: str = "o", kelly_fraction: float = 0.0,
                  max_stake: float = 0.02) -> pd.DataFrame:
    """Bet every selection whose edge clears the threshold at the chosen price.

    price='o' bets the opening/early number, which is the only one you can
    realistically get. price='c' bets the close and is diagnostic only -- if a
    strategy is only profitable at closing prices, it is not a strategy.

    kelly_fraction=0 means flat 1-unit stakes, which is the honest way to read
    ROI. Set 0.25 for quarter-Kelly sizing capped at `max_stake` of bankroll.
    """
    rows = []
    for r in pred.itertuples():
        for k, name in enumerate(OUTCOMES):
            o = getattr(r, f"{price}_{name.lower()}", np.nan)
            o_close = getattr(r, f"c_{name.lower()}", np.nan)
            p = (r.p_h, r.p_d, r.p_a)[k]
            if not (np.isfinite(o) and o > 1 and np.isfinite(p)):
                continue
            edge = p * o - 1.0
            if edge < edge_threshold:
                continue
            if kelly_fraction > 0:
                f = max(0.0, (p * o - 1.0) / (o - 1.0)) * kelly_fraction
                stake = min(f, max_stake)
            else:
                stake = 1.0
            won = (r.y == k)

            # Two CLV numbers, and the difference between them matters.
            #
            # clv_raw compares the price you took against the bookmaker's own
            # closing price. That closing price still carries the margin, so
            # clv_raw is biased UPWARDS by roughly the vig: you can show
            # positive clv_raw and still lose money, which is exactly what this
            # harness produced against a deliberately unbeatable book.
            #
            # clv_fair compares against the DE-VIGGED close -- the market's
            # actual probability expressed as fair odds. That is the honest
            # measure, and the one to trust.
            p_close = (getattr(r, "mc_h", np.nan), getattr(r, "mc_d", np.nan),
                       getattr(r, "mc_a", np.nan))[k]
            fair_close = 1.0 / p_close if (np.isfinite(p_close) and p_close > 0) else np.nan
            rows.append({
                "date": r.date, "home": r.home, "away": r.away, "selection": name,
                "p_model": p, "odds_taken": o, "odds_closing": o_close,
                "fair_closing": fair_close,
                "edge": edge, "stake": stake,
                "won": bool(won),
                "pnl": stake * (o - 1.0) if won else -stake,
                "clv_raw": (o / o_close - 1.0) if (np.isfinite(o_close) and o_close > 1) else np.nan,
                "clv_fair": (o / fair_close - 1.0) if np.isfinite(fair_close) else np.nan,
            })
    return pd.DataFrame(rows)


def bet_metrics(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"n_bets": 0}
    staked = bets["stake"].sum()
    roi = bets["pnl"].sum() / staked if staked else np.nan
    per_unit = (bets["pnl"] / bets["stake"]).to_numpy()
    lo, hi = _bootstrap_ci(per_unit)
    raw = bets["clv_raw"].dropna().to_numpy()
    fair = bets["clv_fair"].dropna().to_numpy()
    flo, fhi = _bootstrap_ci(fair)
    rlo, rhi = _bootstrap_ci(raw)
    return {
        "n_bets": int(len(bets)),
        "staked": float(staked),
        "pnl": float(bets["pnl"].sum()),
        "roi": float(roi),
        "roi_ci": (lo, hi),
        "hit_rate": float(bets["won"].mean()),
        "mean_clv_raw": float(raw.mean()) if len(raw) else np.nan,
        "clv_raw_ci": (rlo, rhi),
        "mean_clv": float(fair.mean()) if len(fair) else np.nan,
        "clv_ci": (flo, fhi),
        "pct_positive_clv": float((fair > 0).mean()) if len(fair) else np.nan,
        "n_clv": int(len(fair)),
    }


# -------------------------------------------------------------------- gate

def gate(metrics: dict, min_matches: int = 1000) -> tuple[bool, str]:
    """The go/no-go decision. Returns (passed, human-readable reason)."""
    if metrics.get("n", 0) == 0:
        return False, "FAIL — no matches with closing odds; the gate could not be evaluated."
    n = metrics["n"]
    delta = metrics["logloss_delta"]
    lo, hi = metrics["logloss_delta_ci"]

    if n < min_matches:
        return False, (
            f"INCONCLUSIVE — only {n} matches evaluated, {min_matches} required. "
            f"Current log-loss delta {delta:+.5f} (95% CI {lo:+.5f}..{hi:+.5f}). "
            "Add seasons or divisions before drawing any conclusion."
        )
    if delta <= 0:
        return False, (
            f"FAIL — model log loss {metrics['logloss_model']:.5f} is WORSE than the "
            f"de-vigged closing line {metrics['logloss_closing']:.5f} over {n} matches "
            f"(delta {delta:+.5f}). Per Method v2 section 8: stop. Do not tune until it "
            "passes — that is fitting the test."
        )
    if lo <= 0:
        return False, (
            f"NOT PROVEN — model beats the close by {delta:+.5f} over {n} matches, but the "
            f"95% CI ({lo:+.5f}..{hi:+.5f}) includes zero. That is noise, not an edge. "
            "Stake nothing."
        )
    return True, (
        f"PASS — model log loss {metrics['logloss_model']:.5f} beats the de-vigged closing "
        f"line {metrics['logloss_closing']:.5f} over {n} matches (delta {delta:+.5f}, "
        f"95% CI {lo:+.5f}..{hi:+.5f}). Proceed to paper picks and CLV tracking — "
        "still not to money."
    )


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict
    bets: pd.DataFrame = field(default_factory=pd.DataFrame)
    bet_stats: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return gate(self.metrics)[0]


def report(res: BacktestResult, min_matches: int = 1000) -> str:
    m = res.metrics
    if m.get("n", 0) == 0:
        return m.get("error", "no data")
    passed, reason = gate(m, min_matches=min_matches)
    lines = [
        "=" * 72,
        "THE GATE — does the model beat the de-vigged closing line?",
        "=" * 72,
        f"matches evaluated      {m['n']}",
        f"model    log loss      {m['logloss_model']:.5f}",
        f"closing  log loss      {m['logloss_closing']:.5f}   <- the benchmark",
    ]
    if "logloss_opening" in m:
        lines.append(f"opening  log loss      {m['logloss_opening']:.5f}")
    lo, hi = m["logloss_delta_ci"]
    lines += [
        f"delta (closing-model)  {m['logloss_delta']:+.5f}  95% CI {lo:+.5f} .. {hi:+.5f}",
        f"brier model / closing  {m['brier_model']:.5f} / {m['brier_closing']:.5f}",
        "",
        reason,
        "",
        "CALIBRATION (does the 60% bucket win 60% of the time?)",
        m["calibration"].to_string(index=False, float_format=lambda v: f"{v:.4f}"),
    ]
    if res.bet_stats.get("n_bets"):
        b = res.bet_stats
        lines += ["", "=" * 72, "SIMULATED BETTING", "=" * 72,
                  f"bets            {b['n_bets']}",
                  f"hit rate        {b['hit_rate']:.4f}",
                  f"ROI             {b['roi']*100:+.2f}%   95% CI "
                  f"{b['roi_ci'][0]*100:+.2f}% .. {b['roi_ci'][1]*100:+.2f}%"]
        if b["n_clv"]:
            lines += [
                f"CLV vs raw close  {b['mean_clv_raw']*100:+.2f}%   (inflated by the margin — "
                "do not trust this one)",
                f"CLV vs FAIR close {b['mean_clv']*100:+.2f}%   95% CI "
                f"{b['clv_ci'][0]*100:+.2f}% .. {b['clv_ci'][1]*100:+.2f}%   <- the honest number",
                f"positive CLV      {b['pct_positive_clv']*100:.1f}% of {b['n_clv']} bets",
                "",
                "CLV is the faster signal: it settles in tens of bets where ROI needs thousands.",
            ]
            if b["mean_clv_raw"] > 0 and b["mean_clv"] <= 0:
                lines += [
                    "",
                    "NOTE: CLV is positive against the raw close but not against the fair close.",
                    "That gap IS the bookmaker's margin. You are beating the closing PRICE without",
                    "beating the closing PROBABILITY, which does not pay.",
                ]
        if passed and b["roi"] > 0 and b["mean_clv"] <= 0:
            lines += [
                "",
                "NOTE: the model beats the close AND makes money, yet fair CLV is negative. That",
                "combination means the market's error PERSISTS to the close — the line never moves",
                "to correct it. CLV is a proxy for edge only against an efficient closing line; in a",
                "market that stays wrong, you can be profitable with negative CLV. Here the gate and",
                "the ROI are the binding evidence, not the CLV.",
            ]
        if not passed and b["roi"] > 0:
            lines += [
                "",
                "RECONCILIATION: the gate failed but simulated betting shows a profit. These answer",
                "different questions. The gate asks whether the model beats the market's best",
                "estimate. The betting sim bets the EARLY price, so a profit there is early-line",
                "value, not model edge — it depends on getting the opening number before it moves,",
                "and it disappears the moment you are slow. Treat a failed gate as the binding",
                "result.",
            ]
    if not passed:
        lines += ["", "VERDICT: stake nothing."]
    return "\n".join(lines)
