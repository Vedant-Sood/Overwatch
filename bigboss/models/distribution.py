"""Forecast distributions from quantiles, conformal calibration, and probability calibration.

Every model emits (mean, q10, q50, q90) of the horizon excess return. These helpers turn that into
probabilities, downside measures, and calibrated intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

Z90 = norm.ppf(0.9)  # 1.2816


def sigma_from_quantiles(q10: np.ndarray, q90: np.ndarray) -> np.ndarray:
    return np.maximum((np.asarray(q90) - np.asarray(q10)) / (2 * Z90), 1e-6)


def prob_above(threshold, q10, q50, q90) -> np.ndarray:
    """P(R > threshold) from a split-normal fitted to (q10, q50, q90): separate scales below/above the median."""
    q10, q50, q90 = (np.asarray(a, float) for a in (q10, q50, q90))
    t = np.broadcast_to(np.asarray(threshold, float), q50.shape)
    s_lo = np.maximum((q50 - q10) / Z90, 1e-6)
    s_hi = np.maximum((q90 - q50) / Z90, 1e-6)
    z = np.where(t < q50, (t - q50) / s_lo, (t - q50) / s_hi)
    return 1 - norm.cdf(z)


def expected_shortfall(q10, q50, alpha: float = 0.10) -> np.ndarray:
    """ES at level alpha using the lower half of the split normal (a loss is a negative number)."""
    s_lo = np.maximum((np.asarray(q50) - np.asarray(q10)) / Z90, 1e-6)
    za = norm.ppf(alpha)
    return np.asarray(q50) - s_lo * norm.pdf(za) / alpha


def expected_downside(q10, q50, q90) -> np.ndarray:
    """E[min(R, 0)] under the split normal - the expected loss component of the forecast."""
    q50 = np.asarray(q50, float)
    s_lo = np.maximum((q50 - np.asarray(q10)) / Z90, 1e-6)
    s_hi = np.maximum((np.asarray(q90) - q50) / Z90, 1e-6)
    # Numerical integration on a fixed grid is fast and robust for vectors.
    grid = np.linspace(0.0005, 0.9995, 400)
    zq = norm.ppf(grid)
    vals = np.where(zq[None, :] < 0, q50[:, None] + s_lo[:, None] * zq[None, :], q50[:, None] + s_hi[:, None] * zq[None, :])
    return np.minimum(vals, 0).mean(axis=1)


class ConformalInterval:
    """Conformalized quantile regression (Romano et al. 2019) for the [q10, q90] band (target coverage 80%)."""

    def __init__(self, coverage: float = 0.8):
        self.coverage = coverage
        self.adjust = 0.0

    def fit(self, y, q_lo, q_hi) -> "ConformalInterval":
        scores = np.maximum(np.asarray(q_lo) - np.asarray(y), np.asarray(y) - np.asarray(q_hi))
        n = len(scores)
        if n < 30:
            self.adjust = 0.0
            return self
        level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        self.adjust = float(np.quantile(scores, level))
        return self

    def apply(self, q_lo, q_hi):
        lo, hi = np.asarray(q_lo) - self.adjust, np.asarray(q_hi) + self.adjust
        # never allow a negative-width band after a negative adjustment
        mid = (lo + hi) / 2
        return np.minimum(lo, mid - 1e-6), np.maximum(hi, mid + 1e-6)


class ProbabilityCalibrator:
    """Isotonic calibration of P(excess > 0) learned on held-out predictions."""

    def __init__(self):
        self.iso: IsotonicRegression | None = None

    def fit(self, p_raw, outcome) -> "ProbabilityCalibrator":
        p_raw, outcome = np.asarray(p_raw), np.asarray(outcome).astype(float)
        if len(p_raw) >= 200 and 0 < outcome.mean() < 1:
            self.iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p_raw, outcome)
        return self

    def apply(self, p_raw) -> np.ndarray:
        p_raw = np.asarray(p_raw)
        return self.iso.predict(p_raw) if self.iso is not None else p_raw


def calibration_table(p: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": np.asarray(outcome).astype(float)}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "mean_pred", "freq", "n"])
    df["bin"] = pd.qcut(df["p"], q=min(bins, df["p"].nunique()), duplicates="drop")
    t = df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), freq=("y", "mean"), n=("y", "size")).reset_index()
    t["bin"] = t["bin"].astype(str)
    return t


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    m = ~np.isnan(y)
    return float(((y[m] >= lo[m]) & (y[m] <= hi[m])).mean()) if m.any() else float("nan")


def pinball_loss(y, q, tau) -> float:
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))
