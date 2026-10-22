"""Model E: ensemble over validated base models, trained only on out-of-sample (out-of-fold) predictions.

Weights are non-negative least squares on prior folds' OOS predictions, shrunk toward equal weights.
The ensemble abstains when (a) base models disagree too much, (b) the calibrated interval is wide
relative to the expected value, or (c) there is not enough OOS history to estimate weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


class StackingEnsemble:
    def __init__(self, members: list[str], shrink: float = 0.3, min_oos_rows: int = 2000):
        self.members = members
        self.shrink = shrink
        self.min_oos_rows = min_oos_rows
        self.weights = {m: 1 / len(members) for m in members}
        self.fitted_on = 0

    def fit(self, oos: pd.DataFrame, y_col: str = "y") -> "StackingEnsemble":
        """oos: rows = (date, symbol) with one `mean_<model>` column per member and the realized label."""
        cols = [f"mean_{m}" for m in self.members]
        d = oos.dropna(subset=cols + [y_col])
        self.fitted_on = len(d)
        if len(d) < self.min_oos_rows:
            return self
        A = d[cols].to_numpy()
        w, _ = nnls(A, d[y_col].to_numpy())
        if w.sum() <= 0:
            w = np.ones(len(cols))
        w = w / w.sum()
        eq = np.ones(len(cols)) / len(cols)
        w = (1 - self.shrink) * w + self.shrink * eq
        self.weights = dict(zip(self.members, w.tolist()))
        return self

    def combine(self, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Weighted combination of means and quantiles plus disagreement measures."""
        w = np.array([self.weights[m] for m in self.members])
        stack = {k: np.vstack([preds[m][k].to_numpy() for m in self.members]) for k in ("mean", "q10", "q50", "q90")}
        out = pd.DataFrame({k: (w[:, None] * v).sum(axis=0) for k, v in stack.items()})
        out["disagreement"] = stack["mean"].std(axis=0)
        out["sign_agreement"] = (np.sign(stack["mean"]) == np.sign(out["mean"].to_numpy())[None, :]).mean(axis=0)
        out["n_models"] = len(self.members)
        return out


def abstain_mask(pred: pd.DataFrame, max_disagreement: float, min_sign_agreement: float = 0.6,
                 skill_ok: bool = True, feature_coverage: pd.Series | None = None, min_coverage: float = 0.5) -> pd.Series:
    """True where the system should answer INSUFFICIENT EVIDENCE rather than recommend.

    Aleatoric noise (wide return intervals) is NOT a reason to abstain - it is priced in by the decision engine.
    Abstention is for epistemic problems: models disagree, the ensemble has no demonstrated OOS skill,
    or the input features are mostly missing.
    """
    m = pred["mean"].isna()
    if "disagreement" in pred:
        m |= pred["disagreement"] > max_disagreement
    if "sign_agreement" in pred:
        m |= pred["sign_agreement"] < min_sign_agreement
    if not skill_ok:
        m |= True
    if feature_coverage is not None:
        m |= feature_coverage.to_numpy() < min_coverage
    return m
