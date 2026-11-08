"""DuckDB store. One database file per data mode, so SYNTHETIC rows can never mix with OBSERVED ones.

Tables (all keyed for idempotent upserts):
  prices               daily bars per symbol; `adj_close` is the total-return series, `close` the split-adjusted close
  symbols              metadata: name, asset_class, sector, benchmark, market, currency
  universe_membership  point-in-time universe: (date, symbol, rank); a managed symbol is a candidate only on its dates
  macro                macro series with `available_at` (observation date + publication lag)
  fundamentals         SEC XBRL facts: discrete quarters for flows, instants for balances; `available` = filing date
  events               SEC filing events with exact acceptance times (New York)
  news                 headlines (research/agents only, never a model input)
  validation_issues    data-quality findings per ingestion run
  ingestion_log        one row per fetch attempt, with provider, status and row count
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = {
    "prices": """symbol VARCHAR, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, adj_close DOUBLE, volume DOUBLE,
                 source VARCHAR, adjustment VARCHAR, fetched_at TIMESTAMP, PRIMARY KEY (symbol, date)""",
    "symbols": """symbol VARCHAR PRIMARY KEY, name VARCHAR, asset_class VARCHAR, sector VARCHAR, industry VARCHAR, benchmark VARCHAR,
                  market VARCHAR, currency VARCHAR, exchange VARCHAR, cik VARCHAR, source VARCHAR, first_date DATE, last_date DATE,
                  note VARCHAR, updated_at TIMESTAMP""",
    "universe_membership": """date DATE, symbol VARCHAR, rank INTEGER, metric DOUBLE, weight DOUBLE, source VARCHAR,
                              PRIMARY KEY (date, symbol)""",
    "macro": """series_id VARCHAR, date DATE, value DOUBLE, available_at TIMESTAMP, source VARCHAR, fetched_at TIMESTAMP,
                PRIMARY KEY (series_id, date)""",
    "fundamentals": """symbol VARCHAR, metric VARCHAR, period_start DATE, period_end DATE, value DOUBLE, available DATE,
                       kind VARCHAR, form VARCHAR, fy INTEGER, fp VARCHAR, accn VARCHAR, concept VARCHAR, source VARCHAR,
                       PRIMARY KEY (symbol, metric, period_end)""",
    "events": """symbol VARCHAR, accession VARCHAR, form VARCHAR, filed DATE, accepted_at TIMESTAMP, items VARCHAR, event_type VARCHAR,
                 importance_prior DOUBLE, description VARCHAR, url VARCHAR, credibility VARCHAR, source VARCHAR,
                 PRIMARY KEY (symbol, accession)""",
    "news": """symbol VARCHAR, url VARCHAR, published_at TIMESTAMP, title VARCHAR, summary VARCHAR, source_domain VARCHAR,
               provider VARCHAR, sentiment DOUBLE, provider_sentiment DOUBLE, credibility VARCHAR, fetched_at TIMESTAMP,
               PRIMARY KEY (symbol, url)""",
    "validation_issues": """run_id VARCHAR, ts TIMESTAMP, symbol VARCHAR, check_name VARCHAR, severity VARCHAR, detail VARCHAR""",
    "ingestion_log": """run_id VARCHAR, ts TIMESTAMP, kind VARCHAR, symbol VARCHAR, provider VARCHAR, status VARCHAR, n_rows INTEGER,
                        detail VARCHAR""",
}

_DATE_COLS = {"date", "period_start", "period_end", "available", "filed", "first_date", "last_date"}
_TS_COLS = {"ts", "fetched_at", "available_at", "accepted_at", "published_at", "updated_at", "logged_at", "resolved_at"}

# DuckDB allows one writer per database; serialize writes from API threads and background jobs in this process.
_WRITE_LOCKS: dict[str, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    return _WRITE_LOCKS.setdefault(str(path.resolve()), threading.Lock())


def _normalize_times(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in _DATE_COLS or c in _TS_COLS:
            s = pd.to_datetime(df[c], errors="coerce")
            if getattr(s.dt, "tz", None) is not None:
                s = s.dt.tz_convert("UTC").dt.tz_localize(None)
            df[c] = s.astype("datetime64[ns]")
    return df


class Store:
    def __init__(self, path: str | Path, mode: str = "OBSERVED"):
        self.path = Path(path)
        self.mode = mode.upper()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect():
            pass

    @classmethod
    def for_mode(cls, data_dir: str | Path, mode: str) -> "Store":
        mode = mode.upper()
        if mode not in ("OBSERVED", "SYNTHETIC"):
            raise ValueError(f"unknown data mode {mode}")
        return cls(Path(data_dir) / f"bigboss_{mode.lower()}.duckdb", mode)

    @contextmanager
    def connect(self):
        con = duckdb.connect(str(self.path))
        try:
            for name, cols in SCHEMA.items():
                con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")
            yield con
        finally:
            con.close()

    # ---------------- generic ----------------
    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        with self.connect() as con:
            df = con.execute(sql, params or []).df()
        return _normalize_times(df)

    def _columns(self, con, table: str) -> list[str]:
        return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]

    def _prepare(self, con, table: str, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._columns(con, table)
        out = df[[c for c in df.columns if c in cols]].copy()
        out = _normalize_times(out)
        for c in out.columns:
            if out[c].dtype == object or str(out[c].dtype).startswith("str"):
                out[c] = out[c].astype(object).where(out[c].notna(), None)
        return out

    def upsert(self, table: str, df: pd.DataFrame, keys: list[str]) -> int:
        """Insert rows, replacing existing rows with the same key. Returns the number of rows written."""
        if df is None or df.empty:
            return 0
        with _lock_for(self.path), self.connect() as con:
            d = self._prepare(con, table, df).drop_duplicates(keys, keep="last")
            con.register("_up", d)
            on = " AND ".join(f"t.{k} IS NOT DISTINCT FROM _up.{k}" for k in keys)
            con.execute(f"DELETE FROM {table} t USING _up WHERE {on}")
            con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _up")
            con.unregister("_up")
        return len(d)

    def append(self, table: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        with _lock_for(self.path), self.connect() as con:
            d = self._prepare(con, table, df)
            con.register("_ap", d)
            con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _ap")
            con.unregister("_ap")
        return len(d)

    def delete(self, table: str, where: str = "TRUE", params: list | None = None) -> None:
        with _lock_for(self.path), self.connect() as con:
            con.execute(f"DELETE FROM {table} WHERE {where}", params or [])

    # ---------------- typed readers ----------------
    def prices(self, symbols: list[str] | None = None, start=None) -> pd.DataFrame:
        sql, params = "SELECT * FROM prices WHERE TRUE", []
        if symbols is not None:
            if not symbols:
                return self.query("SELECT * FROM prices WHERE FALSE")
            sql += " AND symbol IN (SELECT UNNEST(?))"
            params.append(list(symbols))
        if start is not None:
            sql += " AND date >= ?"
            params.append(pd.Timestamp(start).date())
        return self.query(sql + " ORDER BY symbol, date", params)

    def panel(self, field: str = "adj_close", symbols: list[str] | None = None) -> pd.DataFrame:
        p = self.prices(symbols)
        if p.empty:
            return pd.DataFrame()
        return p.pivot(index="date", columns="symbol", values=field).sort_index()

    def symbols(self) -> pd.DataFrame:
        return self.query("SELECT * FROM symbols ORDER BY symbol")

    def membership(self) -> pd.DataFrame:
        return self.query("SELECT * FROM universe_membership ORDER BY date, rank, symbol")

    def macro(self, series: list[str] | None = None) -> pd.DataFrame:
        if series:
            return self.query("SELECT * FROM macro WHERE series_id IN (SELECT UNNEST(?)) ORDER BY series_id, date", [list(series)])
        return self.query("SELECT * FROM macro ORDER BY series_id, date")

    def fundamentals(self, symbol: str | None = None) -> pd.DataFrame:
        if symbol:
            return self.query("SELECT * FROM fundamentals WHERE symbol = ? ORDER BY metric, period_end", [symbol])
        return self.query("SELECT * FROM fundamentals ORDER BY symbol, metric, period_end")

    def events(self, symbol: str | None = None, limit: int = 500) -> pd.DataFrame:
        if symbol:
            return self.query("SELECT * FROM events WHERE symbol = ? ORDER BY accepted_at DESC LIMIT ?", [symbol, int(limit)])
        return self.query("SELECT * FROM events ORDER BY accepted_at DESC LIMIT ?", [int(limit)])

    def news(self, symbol: str | None = None, limit: int = 60) -> pd.DataFrame:
        if symbol:
            return self.query("SELECT * FROM news WHERE symbol = ? ORDER BY published_at DESC LIMIT ?", [symbol, int(limit)])
        return self.query("SELECT * FROM news ORDER BY published_at DESC LIMIT ?", [int(limit)])

    def freshness(self) -> pd.DataFrame:
        return self.query("""
            SELECT symbol, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS n_rows,
                   ARG_MAX(source, date) AS source, ARG_MAX(adjustment, date) AS adjustment
            FROM prices GROUP BY symbol ORDER BY symbol""")

    def error_symbols(self) -> dict[str, str]:
        """Symbols whose most recent validation run found an ERROR -> detail of that error.

        A later price ingestion of the symbol that recorded no issue clears it (validation writes nothing for clean data).
        """
        iss = self.query("SELECT run_id, symbol, severity, check_name, detail FROM validation_issues")
        if iss.empty:
            return {}
        last_run = iss.groupby("symbol")["run_id"].max()
        ok = self.query("SELECT symbol, MAX(run_id) AS run_id FROM ingestion_log WHERE kind = 'prices' AND status = 'OK' GROUP BY symbol")
        last_ok = ok.set_index("symbol")["run_id"] if not ok.empty else pd.Series(dtype=object)
        out = {}
        for sym, run in last_run.items():
            if sym in last_ok.index and str(last_ok[sym]) > str(run):
                continue
            err = iss[(iss["symbol"] == sym) & (iss["run_id"] == run) & (iss["severity"] == "ERROR")]
            if not err.empty:
                out[sym] = "; ".join(f"{r.check_name}: {r.detail}" for r in err.itertuples())
        return out

    def counts(self) -> dict[str, int]:
        return {t: int(self.query(f"SELECT COUNT(*) AS n FROM {t}")["n"].iloc[0]) for t in SCHEMA}

    def clear(self, tables: list[str] | None = None) -> None:
        for t in tables or list(SCHEMA):
            self.delete(t)

