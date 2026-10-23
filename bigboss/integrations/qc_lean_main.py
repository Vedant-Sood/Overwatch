# region imports
from AlgorithmImports import *
# endregion
"""BigBoss -> QuantConnect LEAN replay (paste as main.py of a QuantConnect Python project).

Replays a BigBoss walk-forward experiment in QuantConnect's engine and data, as an independent check of BigBoss's
backtester and price data. It is a BACKTEST algorithm only: no brokerage is configured and it must not be deployed live.

Modes (parameter `mode`):
  weights  replay BigBoss's exact target weights (weights.csv: date,symbol,weight)
  signals  rebuild a simple gated top-k portfolio from BigBoss forecasts (signals.csv), remainder in the benchmark

Timing contract: a row dated D was decided at the close of D. It is used only on trading days strictly after D, with
orders placed shortly before that day's close (BigBoss assumes a fill at the close of D+1).

Data: read from the Object Store (keys bigboss/weights.csv, bigboss/signals.csv) or from `weights_url` / `signals_url`.
"""

import csv
import io
from datetime import datetime

DEFAULTS = {
    "mode": "weights", "start": "2025-08-13", "end": "2026-07-07", "cash": "1000000", "benchmark": "SPY",
    "costs": "bigboss", "resolution": "hour", "minutes_before_close": "30",
    "weights_url": "", "signals_url": "", "weights_key": "bigboss/weights.csv", "signals_key": "bigboss/signals.csv",
    "top_k": "10", "max_weight": "0.10", "rebalance_days": "20", "round_trip_cost": "0.0010",
    "min_skill_t": "1.0", "max_disagreement": "0.03", "min_sign_agreement": "0.6",
}
# BigBoss US cost model approximations (configs/bigboss.toml): commission 0.5 bp; half-spread 1-2 bp + slippage 2 bp.
BIGBOSS_COMMISSION_BPS = 0.5
BIGBOSS_SLIPPAGE_FRACTION = 0.0004


# ---------------- pure helpers (unit-tested outside LEAN with a stub AlgorithmImports) ----------------

def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x  # NaN -> None


def parse_table(text):
    """CSV text -> {date 'YYYY-MM-DD': {symbol: row-dict with float values}} (weights or signals format)."""
    out = {}
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        d, s = (row.get("date") or "").strip()[:10], (row.get("symbol") or "").strip().upper()
        if not d or not s:
            continue
        out.setdefault(d, {})[s] = {k: _num(v) for k, v in row.items() if k not in ("date", "symbol")}
    return out


def latest_before(dates_sorted, day):
    """Most recent decision date strictly before trading day `day` ('YYYY-MM-DD'), or None."""
    best = None
    for d in dates_sorted:
        if d < day:
            best = d
        else:
            break
    return best


def passes_gate(row, min_skill_t, max_disagreement, min_sign_agreement):
    """Same abstention rule as BigBoss's ensemble gate. Missing diagnostics (non-ensemble models) do not block."""
    skill, dis, agree = row.get("skill_t"), row.get("disagreement"), row.get("sign_agreement")
    if skill is not None and skill < min_skill_t:
        return False
    if dis is not None and dis > max_disagreement:
        return False
    if agree is not None and agree < min_sign_agreement:
        return False
    return row.get("mean") is not None


def select_signals(rows, top_k, max_weight, round_trip_cost, min_skill_t, max_disagreement, min_sign_agreement):
    """rows: {symbol: signal row}. Returns {symbol: weight}: top-k by expected excess return net of round-trip cost,
    only names that pass the gate and have positive net edge, each weighted min(max_weight, 1/top_k)."""
    scored = []
    for sym, r in rows.items():
        if passes_gate(r, min_skill_t, max_disagreement, min_sign_agreement):
            edge = r["mean"] - round_trip_cost
            if edge > 0:
                scored.append((edge, sym))
    scored.sort(reverse=True)
    w = min(max_weight, 1.0 / max(top_k, 1))
    return {sym: w for _, sym in scored[:top_k]}


def normalize_targets(weights, benchmark=None):
    """Long-only, gross <= 1. If `benchmark` is given the uninvested remainder goes to it (signals mode)."""
    w = {s: max(0.0, x or 0.0) for s, x in weights.items()}
    gross = sum(w.values())
    if gross > 1.0:
        w = {s: x / gross for s, x in w.items()}
        gross = 1.0
    if benchmark is not None and gross < 1.0:
        w[benchmark] = w.get(benchmark, 0.0) + (1.0 - gross)
    return {s: x for s, x in w.items() if x > 1e-6}


# ---------------- LEAN algorithm ----------------

class BigBossFeeModel(FeeModel):
    """Commission in basis points of traded notional (BigBoss default for US: 0.5 bp, no minimum)."""

    def get_order_fee(self, parameters):
        order = parameters.order
        notional = abs(order.quantity) * parameters.security.price
        return OrderFee(CashAmount(notional * BIGBOSS_COMMISSION_BPS / 1e4, "USD"))


class BigBossReplay(QCAlgorithm):

    def param(self, name):
        v = self.get_parameter(name)
        return v if v not in (None, "") else DEFAULTS[name]

    def initialize(self):
        self.mode = self.param("mode").lower()
        start = datetime.strptime(self.param("start"), "%Y-%m-%d")
        end = datetime.strptime(self.param("end"), "%Y-%m-%d")
        self.set_start_date(start.year, start.month, start.day)
        self.set_end_date(end.year, end.month, end.day)
        self.set_cash(float(self.param("cash")))
        self.resolution = Resolution.MINUTE if self.param("resolution").lower() == "minute" else Resolution.HOUR
        self.costs = self.param("costs").lower()

        key, url = ("weights_key", "weights_url") if self.mode == "weights" else ("signals_key", "signals_url")
        self.table = parse_table(self._load(self.param(key), self.param(url)))
        if not self.table:
            raise Exception(f"BigBoss {self.mode} file is empty or unreadable")
        self.dates = sorted(self.table)

        self.benchmark_ticker = self.param("benchmark").upper()
        self.symbols = {}
        self.missing = set()
        for t in sorted({s for rows in self.table.values() for s in rows} | {self.benchmark_ticker}):
            try:
                sec = self.add_equity(t, self.resolution)
            except Exception as e:  # unknown ticker in QuantConnect's security master
                self.missing.add(t)
                self.debug(f"cannot add {t}: {e}")
                continue
            if self.costs == "bigboss":
                sec.set_fee_model(BigBossFeeModel())
                sec.set_slippage_model(ConstantSlippageModel(BIGBOSS_SLIPPAGE_FRACTION))
            self.symbols[t] = sec.symbol
        bench = self.symbols[self.benchmark_ticker]
        self.set_benchmark(bench)

        self.applied = None
        self.rebalance_counter = 0
        self.n_rebalances = 0
        self.unfilled_targets = set()
        self.schedule.on(self.date_rules.every_day(bench),
                         self.time_rules.before_market_close(bench, int(self.param("minutes_before_close"))),
                         self.rebalance)
        self.debug(f"BigBoss {self.mode}: {len(self.dates)} decision dates, {len(self.symbols)} securities, "
                   f"{len(self.missing)} unavailable")

    def _load(self, key, url):
        if url:
            return self.download(url)
        if self.object_store.contains_key(key):
            return self.object_store.read(key)
        raise Exception(f"no data: upload {key} to the Object Store or set the url parameter")

    def rebalance(self):
        day = self.time.strftime("%Y-%m-%d")
        d = latest_before(self.dates, day)
        if d is None or d == self.applied:
            return
        if self.mode == "weights":
            targets = normalize_targets(self.table[d])
        else:
            # Signals arrive daily; rebalance only every `rebalance_days` decision dates, like BigBoss.
            self.rebalance_counter += 1
            if self.applied is not None and self.rebalance_counter < int(self.param("rebalance_days")):
                self.applied = d
                return
            self.rebalance_counter = 0
            picks = select_signals(self.table[d], int(self.param("top_k")), float(self.param("max_weight")),
                                   float(self.param("round_trip_cost")), float(self.param("min_skill_t")),
                                   float(self.param("max_disagreement")), float(self.param("min_sign_agreement")))
            targets = normalize_targets(picks, benchmark=self.benchmark_ticker)
        self.applied = d
        orders = []
        for t, w in targets.items():
            sym = self.symbols.get(t)
            if sym is None or not self.securities[sym].has_data or not self.securities[sym].is_tradable:
                self.unfilled_targets.add(t)  # weight stays in cash (BigBoss: unknown liquidity -> no fill)
                continue
            orders.append(PortfolioTarget(sym, w))
        self.set_holdings(orders, liquidate_existing_holdings=True)
        self.n_rebalances += 1

    def on_end_of_algorithm(self):
        self.set_runtime_statistic("BigBoss decision dates", str(len(self.dates)))
        self.set_runtime_statistic("Rebalances", str(self.n_rebalances))
        self.set_runtime_statistic("Tickers unavailable", str(len(self.missing)))
        self.set_runtime_statistic("Targets not tradable", str(len(self.unfilled_targets)))
        if self.missing or self.unfilled_targets:
            self.log("unavailable: " + ",".join(sorted(self.missing)))
            self.log("not tradable when targeted: " + ",".join(sorted(self.unfilled_targets)))
