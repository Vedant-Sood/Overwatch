"""Chronological splitting with purging, embargo, and a locked final holdout.

Purging: a training row whose label window ends on/after the first test date is removed, because its
label overlaps the test period. Embargo: an additional gap of `embargo` trading days.

The final holdout (last `holdout_fraction` of dates) is excluded from every split unless the caller
explicitly unlocks it; each unlock is appended to holdout_access.log so repeated peeking is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    calib_dates: pd.DatetimeIndex  # tail of the training window, used for conformal/probability calibration

    @property
    def test_start(self):
        return self.test_dates[0]


def holdout_boundary(dates: pd.DatetimeIndex, fraction: float) -> pd.Timestamp:
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    return dates[int(len(dates) * (1 - fraction))]


def record_holdout_access(data_dir: Path, reason: str) -> int:
    path = Path(data_dir) / "holdout_access.log"
    with open(path, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\t{reason}\n")
    return sum(1 for _ in open(path))


def walk_forward_folds(
    dates: pd.DatetimeIndex,
    label_end: pd.Series,
    min_train: int,
    step: int,
    horizon: int,
    embargo: int = 5,
    train_window: int | None = None,
    holdout_fraction: float = 0.15,
    include_holdout: bool = False,
    calib_fraction: float = 0.2,
) -> list[Fold]:
    """Expanding (or rolling, if train_window) walk-forward folds.

    `label_end` maps each date -> the date its horizon label is known (same for all symbols on a date).
    """
    udates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    end = len(udates) if include_holdout else int(len(udates) * (1 - holdout_fraction))
    folds = []
    pos = min_train
    while pos < end:
        test = udates[pos:min(pos + step, end)]
        first_test = test[0]
        # purge: label must be fully known before the first test decision, minus embargo
        cutoff_idx = max(0, pos - embargo)
        cutoff = udates[cutoff_idx]
        cand = udates[:pos]
        le = label_end.reindex(cand)
        train = cand[(le.notna()) & (le < cutoff) & (cand < first_test)]
        if train_window:
            train = train[-train_window:]
        if len(train) >= max(60, min_train // 4):
            n_cal = max(20, int(len(train) * calib_fraction))
            # calibration slice is itself purged from the fitting slice
            fit_part = train[:-n_cal]
            cal_part = train[-n_cal:]
            if len(fit_part) > horizon + embargo:
                fit_end = cal_part[0]
                fit_part = fit_part[label_end.reindex(fit_part).values < fit_end]
            folds.append(Fold(fit_part, test, cal_part))
        pos += step
    return folds


def purged_kfold(dates: pd.DatetimeIndex, label_end: pd.Series, k: int = 5, embargo: int = 5) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Purged K-fold (Lopez de Prado) for hyperparameter tuning inside a training window."""
    udates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    blocks = np.array_split(np.arange(len(udates)), k)
    out = []
    for b in blocks:
        test = udates[b]
        t0, t1 = test[0], test[-1]
        emb_end = udates[min(len(udates) - 1, b[-1] + embargo)]
        le = label_end.reindex(udates)
        keep = ((udates < t0) & (le < t0).values) | (udates > emb_end)
        out.append((udates[keep], test))
    return out
