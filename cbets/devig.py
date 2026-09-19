"""Margin removal.

Decimal odds carry the bookmaker's margin, so 1/odds is NOT a probability --
the implied probabilities sum to more than 1 (the "overround"). To compare a
model against the market you must first strip the margin out. Getting this
wrong biases every edge number you compute, usually in your favour, which is
the most dangerous direction.

Three methods, increasingly realistic:

  multiplicative  divide out proportionally. Simple, but assumes the margin is
                  spread evenly across outcomes. It isn't.
  power           p_i proportional to (1/o_i)**k. Handles some of the
                  favourite-longshot bias.
  shin            Shin (1993). Models the margin as protection against
                  insider betting, which loads more margin onto longshots.
                  This matches observed bookmaker behaviour best and is the
                  default for 1X2.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = ["overround", "devig", "devig_multiplicative", "devig_power", "devig_shin"]


def _raw(odds) -> np.ndarray:
    o = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(o)) or np.any(o <= 1.0):
        raise ValueError(f"decimal odds must all be finite and > 1.0, got {odds}")
    return 1.0 / o


def overround(odds) -> float:
    """Bookmaker margin as a fraction. 0.05 == a 5% book."""
    return float(_raw(odds).sum() - 1.0)


def devig_multiplicative(odds) -> np.ndarray:
    r = _raw(odds)
    return r / r.sum()


def devig_power(odds) -> np.ndarray:
    r = _raw(odds)
    if abs(r.sum() - 1.0) < 1e-12:
        return r.copy()

    def f(k: float) -> float:
        return float((r ** k).sum() - 1.0)

    try:
        k = brentq(f, 0.2, 5.0, xtol=1e-12)
    except ValueError:
        return devig_multiplicative(odds)
    p = r ** k
    return p / p.sum()


def devig_shin(odds) -> np.ndarray:
    r = _raw(odds)
    pi = float(r.sum())
    if pi <= 1.0 + 1e-12:
        return r / pi

    def p_of_z(z: float) -> np.ndarray:
        inner = z * z + 4.0 * (1.0 - z) * (r * r) / pi
        return (np.sqrt(inner) - z) / (2.0 * (1.0 - z))

    def f(z: float) -> float:
        return float(p_of_z(z).sum() - 1.0)

    lo, hi = 1e-12, 0.5
    try:
        while f(hi) > 0 and hi < 0.999:
            hi = min(0.999, hi * 1.5)
        z = brentq(f, lo, hi, xtol=1e-12)
    except ValueError:
        return devig_multiplicative(odds)
    p = p_of_z(z)
    s = p.sum()
    return p / s if s > 0 else devig_multiplicative(odds)


_METHODS = {
    "multiplicative": devig_multiplicative,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(odds, method: str = "shin") -> np.ndarray:
    """Return true probabilities implied by `odds`, margin removed.

    >>> p = devig([2.10, 3.50, 3.60], method="shin")
    >>> round(float(p.sum()), 12)
    1.0
    """
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(f"unknown de-vig method {method!r}; choose from {sorted(_METHODS)}")
    return fn(odds)
