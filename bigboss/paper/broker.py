"""Paper brokerage: virtual cash, simulated orders and fills, positions, P&L, audit log, risk limits, kill switch.

Fill model: market orders fill at the latest stored close (the price the system can actually observe) adjusted
by the one-way cost estimate (half spread + slippage + impact) in the adverse direction, plus commission.
This is a simulation; it touches no real account and holds no credentials.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bigboss.config import Config
from bigboss.costs import CostModel

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (account_id TEXT PRIMARY KEY, name TEXT, base_currency TEXT, initial_cash REAL,
    cash REAL, created_at TEXT, profile TEXT, kill_switch INTEGER DEFAULT 0, data_mode TEXT);
CREATE TABLE IF NOT EXISTS orders (order_id TEXT PRIMARY KEY, account_id TEXT, created_at TEXT, symbol TEXT, side TEXT,
    quantity REAL, order_type TEXT, limit_price REAL, status TEXT, reason TEXT, strategy TEXT, recommendation_id TEXT);
CREATE TABLE IF NOT EXISTS fills (fill_id TEXT PRIMARY KEY, order_id TEXT, account_id TEXT, ts TEXT, symbol TEXT, side TEXT,
    quantity REAL, ref_price REAL, fill_price REAL, commission REAL, cost_bps REAL, price_date TEXT);
CREATE TABLE IF NOT EXISTS positions (account_id TEXT, symbol TEXT, quantity REAL, avg_cost REAL, realized_pnl REAL,
    PRIMARY KEY (account_id, symbol));
CREATE TABLE IF NOT EXISTS audit (ts TEXT, account_id TEXT, actor TEXT, action TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS equity_history (account_id TEXT, date TEXT, equity REAL, cash REAL, PRIMARY KEY (account_id, date));
"""


class OrderRejected(Exception):
    pass


@dataclass
class OrderRequest:
    symbol: str
    side: str            # BUY | SELL
    quantity: float
    order_type: str = "MARKET"
    limit_price: float | None = None
    strategy: str = "manual"
    recommendation_id: str | None = None


class PaperBroker:
    _lock = threading.RLock()

    def __init__(self, cfg: Config, store, db_path: Path | None = None):
        self.cfg = cfg
        self.store = store  # market data store (prices) - mode must match the account's data_mode
        self.path = db_path or cfg.data_dir / f"paper_{store.mode.lower()}.sqlite"
        self.cost = CostModel(cfg)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        with self._lock:
            con = sqlite3.connect(self.path)
            con.row_factory = sqlite3.Row
            try:
                yield con
                con.commit()
            finally:
                con.close()

    def _audit(self, con, account_id: str, actor: str, action: str, detail: dict) -> None:
        con.execute("INSERT INTO audit VALUES (?,?,?,?,?)", (datetime.now(timezone.utc).isoformat(), account_id, actor, action,
                                                             json.dumps(detail, default=str)))

    # ---- accounts ----
    def create_account(self, name: str, initial_cash: float | None = None, profile: str = "balanced", currency: str | None = None) -> str:
        aid = "PA-" + uuid.uuid4().hex[:8]
        cash = float(initial_cash or self.cfg.get("paper.initial_cash", 100000))
        with self._conn() as c:
            c.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,0,?)", (aid, name, currency or self.cfg.get("paper.base_currency", "USD"),
                                                                        cash, cash, datetime.now(timezone.utc).isoformat(), profile, self.store.mode))
            self._audit(c, aid, "user", "create_account", {"name": name, "cash": cash, "profile": profile})
        return aid

    def accounts(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM accounts ORDER BY created_at")]

    def account(self, aid: str) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT * FROM accounts WHERE account_id=?", (aid,)).fetchone()
        if r is None:
            raise KeyError(aid)
        return dict(r)

    def set_kill_switch(self, aid: str, on: bool, actor: str = "user") -> None:
        with self._conn() as c:
            c.execute("UPDATE accounts SET kill_switch=? WHERE account_id=?", (int(on), aid))
            if on:
                c.execute("UPDATE orders SET status='CANCELLED', reason='kill switch' WHERE account_id=? AND status='PENDING'", (aid,))
            self._audit(c, aid, actor, "kill_switch_on" if on else "kill_switch_off", {})

    # ---- market data ----
    def _quote(self, symbol: str) -> tuple[float, str, dict]:
        p = self.store.prices([symbol])
        if p.empty:
            raise OrderRejected(f"no price data for {symbol}")
        p = p.sort_values("date")
        last = p.iloc[-1]
        tail = p.tail(60)
        adv = float((tail["close"] * tail["volume"]).tail(20).mean()) if tail["volume"].notna().any() else float("nan")
        import numpy as np
        dv = float(np.log(tail["adj_close"]).diff().std())
        return float(last["close"]), str(last["date"].date()), {"adv": adv, "daily_vol": dv}

    def positions(self, aid: str) -> pd.DataFrame:
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM positions WHERE account_id=? AND ABS(quantity) > 1e-9", (aid,))]
        df = pd.DataFrame(rows, columns=["account_id", "symbol", "quantity", "avg_cost", "realized_pnl"])
        if df.empty:
            return df.assign(price=[], price_date=[], market_value=[], unrealized_pnl=[])
        px = [self._quote(s)[:2] for s in df["symbol"]]
        df["price"] = [p[0] for p in px]
        df["price_date"] = [p[1] for p in px]
        df["market_value"] = df["quantity"] * df["price"]
        df["unrealized_pnl"] = (df["price"] - df["avg_cost"]) * df["quantity"]
        return df

    def summary(self, aid: str) -> dict:
        a = self.account(aid)
        pos = self.positions(aid)
        mv = float(pos["market_value"].sum()) if not pos.empty else 0.0
        with self._conn() as c:
            realized = c.execute("SELECT COALESCE(SUM(realized_pnl),0) FROM positions WHERE account_id=?", (aid,)).fetchone()[0]
            fees = c.execute("SELECT COALESCE(SUM(commission),0) FROM fills WHERE account_id=?", (aid,)).fetchone()[0]
        equity = a["cash"] + mv
        return {**a, "market_value": mv, "equity": equity, "total_return": equity / a["initial_cash"] - 1,
                "realized_pnl": realized, "unrealized_pnl": float(pos["unrealized_pnl"].sum()) if not pos.empty else 0.0,
                "commissions_paid": fees, "n_positions": int(len(pos)),
                "weights": ({r.symbol: r.market_value / equity for r in pos.itertuples()} if equity > 0 else {})}

    # ---- orders ----
    def _check(self, aid: str, req: OrderRequest, price: float, equity: float, pos_qty: float) -> None:
        a = self.account(aid)
        if a["kill_switch"]:
            raise OrderRejected("kill switch is ON for this account")
        if req.side not in ("BUY", "SELL"):
            raise OrderRejected("side must be BUY or SELL")
        if req.quantity <= 0:
            raise OrderRejected("quantity must be positive")
        notional = req.quantity * price
        if notional > self.cfg.get("paper.max_order_notional", 50000):
            raise OrderRejected(f"order notional {notional:,.0f} exceeds max_order_notional")
        if req.side == "SELL" and req.quantity > pos_qty + 1e-9:
            raise OrderRejected("short selling is disabled in paper trading; sell quantity exceeds position")
        if req.side == "BUY":
            new_w = (pos_qty * price + notional) / max(equity, 1e-9)
            # Broad benchmark funds are diversified; the single-position cap does not apply to them.
            exempt = set(self.cfg.section("benchmarks").values())
            if req.symbol not in exempt and new_w > self.cfg.get("paper.max_position_weight", 0.2) + 1e-9:
                raise OrderRejected(f"resulting position weight {new_w:.1%} exceeds max_position_weight")
            if notional > a["cash"]:
                raise OrderRejected("insufficient virtual cash")
        hist = self.equity_history(aid)
        if len(hist) >= 2:
            day_ret = hist["equity"].iloc[-1] / hist["equity"].iloc[-2] - 1
            if day_ret <= -self.cfg.get("paper.daily_loss_limit", 0.05):
                raise OrderRejected("daily loss limit breached; new orders blocked until next session")

    def submit(self, aid: str, req: OrderRequest, actor: str = "user") -> dict:
        req.symbol = req.symbol.upper()
        oid = "PO-" + uuid.uuid4().hex[:10]
        now = datetime.now(timezone.utc).isoformat()
        try:
            price, pdate, liq = self._quote(req.symbol)
            summ = self.summary(aid)
            pos = self.positions(aid)
            pos_qty = float(pos.loc[pos["symbol"] == req.symbol, "quantity"].sum()) if not pos.empty else 0.0
            self._check(aid, req, price, summ["equity"], pos_qty)
            if req.order_type == "LIMIT" and req.limit_price is not None:
                if (req.side == "BUY" and price > req.limit_price) or (req.side == "SELL" and price < req.limit_price):
                    raise OrderRejected(f"limit not marketable at last close {price:.2f}")
        except OrderRejected as e:
            with self._conn() as c:
                c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (oid, aid, now, req.symbol, req.side, req.quantity,
                          req.order_type, req.limit_price, "REJECTED", str(e), req.strategy, req.recommendation_id))
                self._audit(c, aid, actor, "order_rejected", {"order_id": oid, "reason": str(e), **req.__dict__})
            return {"order_id": oid, "status": "REJECTED", "symbol": req.symbol, "side": req.side, "quantity": req.quantity, "reason": str(e)}

        meta = self.store.symbols()
        ac = meta.set_index("symbol")["asset_class"].get(req.symbol, "equity") if not meta.empty else "equity"
        ce = self.cost.estimate(req.symbol, str(ac), req.quantity * price, liq["adv"], liq["daily_vol"])
        exec_bps = ce.half_spread_bps + ce.slippage_bps + ce.impact_bps + ce.tax_bps
        sign = 1 if req.side == "BUY" else -1
        fill_price = price * (1 + sign * exec_bps / 1e4)
        commission = req.quantity * price * ce.commission_bps / 1e4
        with self._conn() as c:
            c.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (oid, aid, now, req.symbol, req.side, req.quantity,
                      req.order_type, req.limit_price, "FILLED", "", req.strategy, req.recommendation_id))
            c.execute("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", ("PF-" + uuid.uuid4().hex[:10], oid, aid, now, req.symbol,
                      req.side, req.quantity, price, fill_price, commission, ce.total_one_way_bps, pdate))
            row = c.execute("SELECT quantity, avg_cost, realized_pnl FROM positions WHERE account_id=? AND symbol=?", (aid, req.symbol)).fetchone()
            q0, ac0, rp0 = (row["quantity"], row["avg_cost"], row["realized_pnl"]) if row else (0.0, 0.0, 0.0)
            if req.side == "BUY":
                q1 = q0 + req.quantity
                ac1 = (q0 * ac0 + req.quantity * fill_price + commission) / q1
                rp1 = rp0
                cash_delta = -(req.quantity * fill_price + commission)
            else:
                q1 = q0 - req.quantity
                ac1 = ac0 if q1 > 1e-9 else 0.0
                rp1 = rp0 + (fill_price - ac0) * req.quantity - commission
                cash_delta = req.quantity * fill_price - commission
            c.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?)", (aid, req.symbol, q1, ac1, rp1))
            c.execute("UPDATE accounts SET cash = cash + ? WHERE account_id=?", (cash_delta, aid))
            self._audit(c, aid, actor, "order_filled", {"order_id": oid, "fill_price": fill_price, "ref_price": price,
                                                        "price_date": pdate, "commission": commission, **req.__dict__})
        self.mark_equity(aid)
        return {"order_id": oid, "status": "FILLED", "symbol": req.symbol, "side": req.side, "quantity": req.quantity,
                "ref_price": price, "fill_price": fill_price, "commission": commission, "price_date": pdate,
                "cost_estimate": ce.as_dict(), "simulated": True}

    def mark_equity(self, aid: str) -> None:
        s = self.summary(aid)
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO equity_history VALUES (?,?,?,?)",
                      (aid, datetime.now(timezone.utc).date().isoformat(), s["equity"], s["cash"]))

    def equity_history(self, aid: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.DataFrame([dict(r) for r in c.execute("SELECT * FROM equity_history WHERE account_id=? ORDER BY date", (aid,))],
                                columns=["account_id", "date", "equity", "cash"])

    def orders(self, aid: str, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM orders WHERE account_id=? ORDER BY created_at DESC LIMIT ?", (aid, limit))]

    def fills(self, aid: str, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM fills WHERE account_id=? ORDER BY ts DESC LIMIT ?", (aid, limit))]

    def audit_log(self, aid: str | None = None, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            q = "SELECT * FROM audit" + (" WHERE account_id=?" if aid else "") + " ORDER BY ts DESC LIMIT ?"
            return [dict(r) for r in c.execute(q, ((aid, limit) if aid else (limit,)))]

    def attribution(self, aid: str) -> list[dict]:
        """P&L attribution by strategy tag (realized from fills + unrealized marked at last close)."""
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("""SELECT o.strategy, f.symbol, f.side, f.quantity, f.fill_price, f.commission
                                                  FROM fills f JOIN orders o ON f.order_id = o.order_id WHERE f.account_id=?""", (aid,))]
        if not rows:
            return []
        df = pd.DataFrame(rows)
        px = {s: self._quote(s)[0] for s in df["symbol"].unique()}
        df["signed_qty"] = df["quantity"].where(df["side"] == "BUY", -df["quantity"])
        df["cash_flow"] = -df["signed_qty"] * df["fill_price"] - df["commission"]
        df["mark"] = df["signed_qty"] * df["symbol"].map(px)
        g = df.groupby("strategy").agg(pnl_cash=("cash_flow", "sum"), mark=("mark", "sum"), commissions=("commission", "sum"), n_fills=("symbol", "size"))
        g["total_pnl"] = g["pnl_cash"] + g["mark"]
        return g.reset_index()[["strategy", "total_pnl", "commissions", "n_fills"]].to_dict("records")
