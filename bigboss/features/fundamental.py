"""Point-in-time fundamental features from SEC XBRL data.

Fundamentals change only when a filing is accepted. We compute metrics at each filing-availability
date (filed + 1 day) and as-of-join them to the daily calendar. Valuation ratios combine those values
with the RAW (unadjusted) close of the day, because share counts are as-reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bigboss.data.providers.sec import latest_instant, ttm_as_of
from bigboss.features.technical import _reg

FUNDAMENTAL_FEATURES = ["earnings_yield", "sales_yield", "fcf_yield", "revenue_growth_yoy", "earnings_growth_yoy",
                        "gross_margin", "operating_margin", "net_margin", "roe", "roic", "debt_to_equity",
                        "current_ratio", "accruals", "fundamentals_age_days"]
for _f in FUNDAMENTAL_FEATURES:
    _reg("fundamental", _f)


def _series(fund: pd.DataFrame, metric: str) -> pd.DataFrame:
    s = fund[fund["metric"] == metric]
    return s.rename(columns={"period_end": "end"})[["end", "value", "available"]].sort_values("end")


def fundamental_snapshots(fund: pd.DataFrame) -> pd.DataFrame:
    """Per filing-availability date, compute PIT fundamental metrics (excluding price-based ones)."""
    if fund.empty:
        return pd.DataFrame()
    m = {k: _series(fund, k) for k in fund["metric"].unique()}
    avail_dates = sorted(set(fund["available"].dropna()))
    rows = []
    for d in avail_dates:
        as_of = d + pd.Timedelta(days=1)  # usable the day after filing

        def ttm(name, lag_q=0):
            q = m.get(name)
            if q is None or q.empty:
                return np.nan
            if lag_q:
                q = q[q["available"] < as_of]
                q = q.iloc[:-lag_q] if len(q) > lag_q else q.iloc[0:0]
                return ttm_as_of(q, as_of, max_age_days=200 + 92 * lag_q)
            return ttm_as_of(q, as_of)

        def inst(name):
            s = m.get(name)
            return np.nan if s is None else latest_instant(s, as_of)

        rev, rev_prev = ttm("revenue"), ttm("revenue", 4)
        ni, ni_prev = ttm("net_income"), ttm("net_income", 4)
        cfo, capex = ttm("cfo"), ttm("capex")
        oi, gp = ttm("operating_income"), ttm("gross_profit")
        tax, pretax = ttm("income_tax"), ttm("pretax_income")
        equity, assets, cash = inst("equity"), inst("total_assets"), inst("cash")
        debt = np.nansum([inst("lt_debt"), inst("st_debt")])
        ca, cl = inst("current_assets"), inst("current_liabilities")
        shares = inst("shares")
        tax_rate = np.clip(tax / pretax, 0, 0.4) if pretax and pretax > 0 and not np.isnan(tax) else 0.21
        invested = debt + equity - (0 if np.isnan(cash) else cash)
        latest_end = max((s[s["available"] < as_of]["end"].max() for s in m.values() if not s[s["available"] < as_of].empty), default=pd.NaT)
        rows.append({
            "date": as_of, "ttm_revenue": rev, "ttm_net_income": ni, "ttm_fcf": cfo - capex if not np.isnan(capex) else cfo,
            "shares": shares,
            "revenue_growth_yoy": rev / rev_prev - 1 if rev_prev and rev_prev > 0 else np.nan,
            "earnings_growth_yoy": (ni - ni_prev) / abs(ni_prev) if ni_prev and ni_prev != 0 else np.nan,
            "gross_margin": gp / rev if rev else np.nan,
            "operating_margin": oi / rev if rev else np.nan,
            "net_margin": ni / rev if rev else np.nan,
            "roe": ni / equity if equity and equity > 0 else np.nan,
            "roic": oi * (1 - tax_rate) / invested if invested and invested > 0 else np.nan,
            "debt_to_equity": debt / equity if equity and equity > 0 else np.nan,
            "current_ratio": ca / cl if cl and cl > 0 else np.nan,
            "accruals": (ni - cfo) / assets if assets and assets > 0 else np.nan,
            "latest_period_end": latest_end,
        })
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")


def fundamental_features(fund_all: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Daily (date, symbol) fundamental features, as-of joined. Symbols without SEC data get no rows."""
    out = []
    if fund_all.empty:
        return pd.DataFrame()
    for sym, fund in fund_all.groupby("symbol"):
        snaps = fundamental_snapshots(fund)
        if snaps.empty:
            continue
        px = prices[prices["symbol"] == sym][["date", "close"]].sort_values("date")
        if px.empty:
            continue
        j = pd.merge_asof(px, snaps, on="date", direction="backward")
        mcap = j["close"] * j["shares"]
        j["earnings_yield"] = j["ttm_net_income"] / mcap
        j["sales_yield"] = j["ttm_revenue"] / mcap
        j["fcf_yield"] = j["ttm_fcf"] / mcap
        j["fundamentals_age_days"] = (j["date"] - j["latest_period_end"]).dt.days
        j["symbol"] = sym
        out.append(j.set_index(["date", "symbol"])[FUNDAMENTAL_FEATURES])
    if not out:
        return pd.DataFrame()
    return pd.concat(out).replace([np.inf, -np.inf], np.nan)
