"""Shared HTTP plumbing for providers: rate limiting, retries, a raw-response cache, and key redaction."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from bigboss.logging_utils import get_logger, log

logger = get_logger("bigboss.data.http")

_SECRET_PARAMS = re.compile(r"(?i)\b(apikey|api_key|apiKey|token|key|access_key|secret)=([^&\s'\"]+)")


class ProviderError(RuntimeError):
    """A provider could not deliver usable data (network, auth, plan limits, or an unexpected payload)."""


def redact(text: str) -> str:
    """Remove credentials from URLs and error messages before they reach logs, the store, or the UI."""
    return _SECRET_PARAMS.sub(lambda m: f"{m.group(1)}=***", str(text))


class HttpClient:
    """GET with a per-client minimum interval between requests, retries with backoff, and an on-disk cache.

    `max_age_s` > 0 reuses a cached body younger than that; 0 always fetches (the body is still cached, so a run
    can be audited and replayed).
    """

    def __init__(self, cache_dir: str | Path | None, min_interval_s: float = 0.0, user_agent: str = "BigBoss research",
                 max_retries: int = 3, timeout_s: float = 30.0, headers: dict | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", **(headers or {})}
        self._last = 0.0
        self._lock = threading.Lock()

    def _key(self, url: str, params: dict | None) -> str:
        safe = {k: v for k, v in (params or {}).items() if not _SECRET_PARAMS.search(f"{k}=x")}
        return hashlib.sha256(json.dumps([url, sorted(safe.items())], default=str).encode()).hexdigest()[:32]

    def _cached(self, key: str, max_age_s: float):
        if not self.cache_dir or max_age_s <= 0:
            return None
        f = self.cache_dir / f"{key}.body"
        if f.exists() and time.time() - f.stat().st_mtime < max_age_s:
            return f.read_text(encoding="utf-8"), datetime.fromtimestamp(f.stat().st_mtime, timezone.utc)
        return None

    def get(self, url: str, params: dict | None = None, max_age_s: float = 0) -> tuple[str, datetime]:
        key = self._key(url, params)
        hit = self._cached(key, max_age_s)
        if hit:
            return hit
        err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            with self._lock:
                wait = self.min_interval_s - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                r = httpx.get(url, params=params, headers=self.headers, timeout=self.timeout_s, follow_redirects=True)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                body = r.text
                fetched = datetime.now(timezone.utc)
                if self.cache_dir:
                    (self.cache_dir / f"{key}.body").write_text(body, encoding="utf-8")
                return body, fetched
            except httpx.HTTPStatusError as e:
                err = e
                code = e.response.status_code
                if code < 500 and code != 429:  # client errors (auth, plan, not found) do not improve with retries
                    break
            except httpx.HTTPError as e:
                err = e
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 30))
        log(logger, 30, "request failed", url=redact(url), error=redact(err))
        raise ProviderError(redact(f"{url}: {err}"))


def to_ny_naive(ts) -> pd.Series | pd.Timestamp:
    """UTC timestamps -> naive America/New_York wall-clock time (the convention for event and news times)."""
    x = pd.to_datetime(ts, utc=True)
    if isinstance(x, pd.Timestamp):
        return x.tz_convert("America/New_York").tz_localize(None)
    return x.dt.tz_convert("America/New_York").dt.tz_localize(None)


def frame(df: pd.DataFrame, source: str, adjustment: str) -> pd.DataFrame:
    """Normalize a daily OHLCV frame to the store schema (date, open, high, low, close, adj_close, volume, ...)."""
    out = df.copy()
    d = pd.to_datetime(out["date"])
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    out["date"] = d.dt.normalize().astype("datetime64[ns]")
    for c in ("open", "high", "low", "close", "adj_close", "volume"):
        if c not in out:
            out[c] = out["close"] if c in ("open", "high", "low", "adj_close") else float("nan")
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["source"] = source
    if "adjustment" not in out:
        out["adjustment"] = adjustment
    out = out.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date")
    return out[["date", "open", "high", "low", "close", "adj_close", "volume", "source", "adjustment"]].reset_index(drop=True)


class PriceProvider:
    """Interface for daily price providers."""

    name = "base"
    disabled_reason: str | None = None

    def supports(self, symbol: str) -> bool:
        return self.disabled_reason is None

    def fetch_daily(self, symbol: str, start: str = "2005-01-01") -> pd.DataFrame:
        raise NotImplementedError
