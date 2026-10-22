"""Model B: fundamental valuation intelligence.

Produces (1) a point-in-time quality/value profile and (2) explicit bear/base/bull valuation scenarios via
a simple two-stage FCF-to-equity DCF. All assumptions are shown; nothing here is a price target promise.
Scenario growth rates are anchored to the company's own trailing growth and shrunk toward a long-run rate,
because extrapolating recent growth is the most common valuation error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bigboss.features.fundamental import fundamental_snapshots


def valuation_profile(fund: pd.DataFrame, price: float, as_of: pd.Timestamp, sector_peers: pd.DataFrame | None = None,
                      discount_rate: float = 0.09, terminal_growth: float = 0.025, years: int = 10) -> dict:
    snaps = fundamental_snapshots(fund)
    if snaps.empty:
        return {"available": False, "reason": "no SEC XBRL fundamentals for this symbol"}
    s = snaps[snaps["date"] <= as_of]
    if s.empty:
        return {"available": False, "reason": "no filings available as of this date"}
    cur = s.iloc[-1]
    shares = cur["shares"]
    if not np.isfinite(shares) or shares <= 0 or not np.isfinite(price):
        return {"available": False, "reason": "missing share count or price"}
    mcap = price * shares
    fcf = cur["ttm_fcf"]
    ni = cur["ttm_net_income"]
    rev = cur["ttm_revenue"]
    g_hist = cur["revenue_growth_yoy"] if np.isfinite(cur["revenue_growth_yoy"]) else 0.03
    base_g = float(np.clip(0.5 * g_hist + 0.5 * 0.04, -0.05, 0.20))
    scenarios = {}
    for name, g, r in (("bear", base_g - 0.05, discount_rate + 0.01), ("base", base_g, discount_rate), ("bull", base_g + 0.04, discount_rate - 0.01)):
        if not np.isfinite(fcf) or fcf <= 0:
            scenarios[name] = {"growth": g, "discount_rate": r, "value_per_share": None, "implied_return": None,
                               "note": "trailing FCF <= 0: DCF not meaningful"}
            continue
        v, f = 0.0, fcf
        for t in range(1, years + 1):
            gt = g + (terminal_growth - g) * (t - 1) / (years - 1)  # fade to terminal growth
            f *= 1 + gt
            v += f / (1 + r) ** t
        tv = f * (1 + terminal_growth) / (r - terminal_growth) / (1 + r) ** years
        vps = (v + tv) / shares
        scenarios[name] = {"growth": round(g, 4), "discount_rate": r, "value_per_share": float(vps), "implied_return": float(vps / price - 1)}
    profile = {
        "available": True, "as_of": str(as_of.date()), "price": price, "market_cap": float(mcap),
        "filing_based_on": str(cur["date"].date()),
        "latest_period_end": str(cur["latest_period_end"].date()) if pd.notna(cur["latest_period_end"]) else None,
        "ttm_revenue": _f(rev), "ttm_net_income": _f(ni), "ttm_fcf": _f(fcf),
        "pe": _f(mcap / ni) if ni and ni > 0 else None, "ps": _f(mcap / rev) if rev and rev > 0 else None,
        "fcf_yield": _f(fcf / mcap), "earnings_yield": _f(ni / mcap),
        "revenue_growth_yoy": _f(cur["revenue_growth_yoy"]), "earnings_growth_yoy": _f(cur["earnings_growth_yoy"]),
        "gross_margin": _f(cur["gross_margin"]), "operating_margin": _f(cur["operating_margin"]), "net_margin": _f(cur["net_margin"]),
        "roe": _f(cur["roe"]), "roic": _f(cur["roic"]), "debt_to_equity": _f(cur["debt_to_equity"]),
        "current_ratio": _f(cur["current_ratio"]), "accruals": _f(cur["accruals"]),
        "scenarios": scenarios,
        "assumptions": {"discount_rate_base": discount_rate, "terminal_growth": terminal_growth, "explicit_years": years,
                        "growth_anchor": "50% trailing revenue growth + 50% 4% long-run, clipped to [-5%, 20%]"},
        "history": snaps[["date", "ttm_revenue", "ttm_net_income", "ttm_fcf", "operating_margin", "roic"]]
            .tail(40).assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).replace({np.nan: None}).to_dict("records"),
    }
    profile["quality_score"] = quality_score(profile)
    return profile


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def quality_score(p: dict) -> dict:
    """Transparent 0-100 quality score: profitability, cash conversion, balance sheet, growth. Missing items are skipped."""
    parts = {}
    if p.get("roic") is not None:
        parts["roic"] = float(np.clip(p["roic"] / 0.20, 0, 1))
    if p.get("operating_margin") is not None:
        parts["margin"] = float(np.clip(p["operating_margin"] / 0.25, 0, 1))
    if p.get("accruals") is not None:
        parts["cash_conversion"] = float(np.clip(1 - p["accruals"] / 0.10, 0, 1))
    if p.get("debt_to_equity") is not None:
        parts["balance_sheet"] = float(np.clip(1 - p["debt_to_equity"] / 3, 0, 1))
    if p.get("revenue_growth_yoy") is not None:
        parts["growth"] = float(np.clip((p["revenue_growth_yoy"] + 0.05) / 0.25, 0, 1))
    score = 100 * float(np.mean(list(parts.values()))) if parts else None
    return {"score": score, "components": parts, "n_components": len(parts)}
