"""Ex-ante transaction cost model.

one-way cost (as a fraction of traded notional) =
    commission + half_spread + slippage + market_impact + taxes
market_impact = k * sigma_daily * sqrt(order_notional / ADV_dollar)   (square-root law; Almgren et al., Toth et al.)

Spreads: configured per asset class. The Corwin-Schultz high/low estimator is biased upward on daily
data for liquid names, so it is only used (shrunk 50%) for thinly traded assets (ADV < $20m).
All parameters are configurable; they are estimates and should be checked against your broker's fills.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bigboss.config import Config, market_of


@dataclass
class CostEstimate:
    commission_bps: float
    half_spread_bps: float
    slippage_bps: float
    impact_bps: float
    tax_bps: float
    total_one_way_bps: float
    total_round_trip_bps: float
    order_notional: float
    participation: float
    liquidity_ok: bool
    note: str = ""

    def as_dict(self):
        return asdict(self)


class CostModel:
    def __init__(self, cfg: Config, multiplier: float = 1.0):
        c = cfg.section("costs")
        self.c = c
        self.multiplier = multiplier  # used for cost-sensitivity analysis

    def _class_key(self, symbol: str, asset_class: str) -> str:
        if asset_class.startswith("proxy_"):
            return asset_class
        if market_of(symbol) == "IN":
            return "india" if asset_class == "equity" else "etf"
        if asset_class.endswith("etf"):
            return "etf"
        return "large_cap"

    def estimate(self, symbol: str, asset_class: str, order_notional: float, adv_dollar: float,
                 daily_vol: float, cs_spread: float | None = None, side: str = "buy") -> CostEstimate:
        mkt = market_of(symbol)
        comm = self.c["commission_bps"].get(mkt, self.c["commission_bps"]["default"])
        if order_notional > 0 and self.c["min_commission"].get(mkt, 0):
            comm = max(comm, self.c["min_commission"][mkt] / order_notional * 1e4)
        hs = self.c["half_spread_bps"].get(self._class_key(symbol, asset_class), self.c["half_spread_bps"]["default"])
        note = ""
        adv = adv_dollar if adv_dollar and np.isfinite(adv_dollar) and adv_dollar > 0 else np.nan
        if cs_spread is not None and np.isfinite(cs_spread) and (np.isnan(adv) or adv < 2e7):
            hs = max(hs, 0.5 * 0.5 * cs_spread * 1e4)
            note = "spread from Corwin-Schultz estimate (thin liquidity)"
        slip = self.c["slippage_bps"]
        if np.isnan(adv) and asset_class.startswith("proxy_"):
            # Index/FX/rates proxies stand for deep ETF/futures markets: assume a 1% participation for impact.
            sigma = daily_vol if daily_vol and np.isfinite(daily_vol) else 0.01
            participation, impact, liquid = np.nan, self.c["impact_coefficient"] * sigma * np.sqrt(0.01) * 1e4, True
            note = (note + "; " if note else "") + "index proxy: liquidity assumed (trades via liquid ETF/futures)"
        elif np.isnan(adv):
            participation, impact, liquid = np.nan, 25.0, False
            note = (note + "; " if note else "") + "no volume data: impact set to 25bps and liquidity unverified"
        else:
            participation = order_notional / adv
            sigma = daily_vol if daily_vol and np.isfinite(daily_vol) else 0.02
            impact = self.c["impact_coefficient"] * sigma * np.sqrt(max(participation, 0)) * 1e4
            liquid = participation <= self.c["max_participation"]
        tax = self.c.get("india_stt_bps", 0.0) if mkt == "IN" and asset_class == "equity" else 0.0
        one_way = (comm + hs + slip + impact + tax) * self.multiplier
        return CostEstimate(comm, hs, slip, float(impact), tax, float(one_way), float(2 * one_way),
                            float(order_notional), float(participation) if np.isfinite(participation) else float("nan"),
                            bool(liquid), note)

    def vectorized_one_way(self, notional: np.ndarray, adv: np.ndarray, daily_vol: np.ndarray, base_bps: np.ndarray) -> np.ndarray:
        """Fast per-trade one-way cost in decimal for the backtester. base_bps = commission + half spread + slippage + tax."""
        part = np.where(adv > 0, notional / np.where(adv > 0, adv, 1), 0.05)
        sigma = np.where(np.isfinite(daily_vol), daily_vol, 0.02)
        impact = self.c["impact_coefficient"] * sigma * np.sqrt(np.maximum(part, 0))
        return (base_bps / 1e4 + impact) * self.multiplier

    def base_bps(self, symbol: str, asset_class: str) -> float:
        mkt = market_of(symbol)
        comm = self.c["commission_bps"].get(mkt, self.c["commission_bps"]["default"])
        hs = self.c["half_spread_bps"].get(self._class_key(symbol, asset_class), self.c["half_spread_bps"]["default"])
        tax = self.c.get("india_stt_bps", 0.0) if mkt == "IN" and asset_class == "equity" else 0.0
        return comm + hs + self.c["slippage_bps"] + tax
