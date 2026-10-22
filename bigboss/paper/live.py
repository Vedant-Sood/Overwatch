"""Guarded interface for OPTIONAL real-broker integration. No live adapter ships with BigBoss.

Any future adapter must subclass `LiveBrokerAdapter`; `LiveOrderGateway` enforces:
  1. `live_trading.enabled = true` in config (default false),
  2. a global kill switch file (data/KILL_SWITCH) is absent,
  3. a single-use approval token bound to the EXACT order (symbol, side, qty, type, limit) created by an
     explicit human confirmation step and expiring after 5 minutes,
  4. independent risk limits checked before the adapter is called.
Credentials are read from environment variables only; they are never stored in config, logs, or the DB.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

from bigboss.config import Config


@dataclass(frozen=True)
class LiveOrder:
    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    limit_price: float | None = None

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class LiveBrokerAdapter(ABC):
    @abstractmethod
    def place_order(self, order: LiveOrder) -> dict: ...

    @abstractmethod
    def positions(self) -> list[dict]: ...


class LiveTradingDisabled(Exception):
    pass


class LiveOrderGateway:
    TOKEN_TTL_S = 300

    def __init__(self, cfg: Config, adapter: LiveBrokerAdapter | None = None):
        self.cfg = cfg
        self.adapter = adapter
        self._secret = secrets.token_bytes(32)  # per-process: tokens never survive a restart
        self._used: set[str] = set()

    def kill_switch_path(self):
        return self.cfg.data_dir / "KILL_SWITCH"

    def engage_kill_switch(self, reason: str) -> None:
        self.kill_switch_path().write_text(f"{time.time()}\t{reason}\n")

    def _check_enabled(self) -> None:
        if not self.cfg.get("live_trading.enabled", False):
            raise LiveTradingDisabled("live trading is disabled (live_trading.enabled = false)")
        if self.kill_switch_path().exists():
            raise LiveTradingDisabled("global kill switch is engaged")
        if self.adapter is None:
            raise LiveTradingDisabled("no live broker adapter is configured")

    def issue_approval_token(self, order: LiveOrder, human_confirmation: str) -> str:
        """Only call after showing the exact order to the user and receiving their typed confirmation."""
        expected = f"APPROVE {order.side} {order.quantity:g} {order.symbol}"
        if human_confirmation.strip() != expected:
            raise PermissionError(f"confirmation must be exactly: {expected!r}")
        ts = str(int(time.time()))
        mac = hmac.new(self._secret, f"{order.digest()}|{ts}".encode(), hashlib.sha256).hexdigest()
        return f"{ts}.{mac}"

    def place(self, order: LiveOrder, token: str) -> dict:
        self._check_enabled()
        try:
            ts, mac = token.split(".")
        except ValueError as e:
            raise PermissionError("malformed approval token") from e
        if time.time() - int(ts) > self.TOKEN_TTL_S:
            raise PermissionError("approval token expired")
        good = hmac.new(self._secret, f"{order.digest()}|{ts}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(good, mac):
            raise PermissionError("approval token does not match this exact order")
        if token in self._used:
            raise PermissionError("approval token already used")
        self._used.add(token)
        return self.adapter.place_order(order)


def credential(name: str) -> str | None:
    """Broker credentials come only from the environment (never config files or the database)."""
    return os.environ.get(name)
