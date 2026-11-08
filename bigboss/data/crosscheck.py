"""Cross-check an adjusted price series against raw exchange closes (e.g. Yahoo adjusted vs NSE raw).

Each day's pair of returns is classified as:
* corporate action  - the raw close jumps by >= `ca_threshold` (split, bonus, demerger): the adjusted series should be smooth;
* normal            - the two returns agree within `tol`;
* payout-like       - the raw close falls more than the adjusted one, by less than `ca_threshold` (dividends, small
                      demergers): what a correct adjustment looks like;
* suspicious        - anything else (the adjusted series moves without a matching raw move, or lags it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compare_returns(adjusted: pd.DataFrame, raw: pd.DataFrame, ca_threshold: float = 0.15, tol: float = 0.005) -> dict:
    a = adjusted.set_index(pd.to_datetime(adjusted["date"]))["adj_close"].astype(float)
    r = raw.set_index(pd.to_datetime(raw["date"]))["close"].astype(float)
    j = pd.concat({"adj": a, "raw": r}, axis=1, join="inner").sort_index()
    j = j[(j > 0).all(axis=1)]
    ret = j.pct_change().dropna()
    diff = ret["adj"] - ret["raw"]
    ca = ret["raw"].abs() >= ca_threshold
    normal = ~ca & (diff.abs() <= tol)
    payout = ~ca & ~normal & (ret["raw"] < ret["adj"]) & (diff <= ca_threshold)
    susp = ~ca & ~normal & ~payout
    sus = pd.DataFrame({"date": ret.index[susp], "adj_ret": ret["adj"][susp].to_numpy(), "raw_ret": ret["raw"][susp].to_numpy()})
    corr = float(ret["adj"][normal | payout].corr(ret["raw"][normal | payout])) if (normal | payout).sum() > 2 else np.nan
    return {"n_days": int(len(ret)), "corr_normal_days": corr, "n_corporate_action_days": int(ca.sum()),
            "n_payout_like_days": int(payout.sum()), "n_suspicious_days": int(susp.sum()),
            "corporate_action_dates": [str(d.date()) for d in ret.index[ca]], "suspicious": sus.reset_index(drop=True)}
