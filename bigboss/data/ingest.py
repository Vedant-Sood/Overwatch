"""Ingestion: providers -> validation -> store, with an ingestion log entry for every fetch attempt.

No silent substitution: if every provider fails for a symbol it is logged FAILED and nothing is written for it.
Point-in-time universe builders (Massive, DoltHub, S&P 500 history, NIFTY 50 history, the ETF file) replace the
`universe_membership` table: a managed symbol is a modelling candidate only on the dates it was a member.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bigboss.config import Config, market_of
from bigboss.data.providers.base import HttpClient, ProviderError, redact
from bigboss.data.providers.fred_assets import SPEC as XA_SPEC
from bigboss.data.providers.fred_assets import FredCrossAssetProvider
from bigboss.data.providers.keyed import AlphaVantageProvider, FmpProvider, MassiveProvider, _env
from bigboss.data.providers.prices import StooqProvider, YahooChartProvider
from bigboss.data.providers.yfinance_provider import YFinanceProvider
from bigboss.data.store import Store
from bigboss.data.validation import looks_like_split, validate_prices
from bigboss.logging_utils import get_logger, log

logger = get_logger("bigboss.data.ingest")
Progress = Callable[[float, str], None] | None
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) BigBoss research"

GROUP_CLASS = {"us_etfs": "etf", "us_equities": "equity", "india_equities": "equity", "india_etfs": "etf", "indices": "index"}
PRICE_COLS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source", "adjustment"]


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Ingestor:
    def __init__(self, cfg: Config, store: Store):
        if store.mode != "OBSERVED":
            raise ValueError("ingestion writes observed data only; use `bigboss demo-data` for the SYNTHETIC store")
        self.cfg = cfg
        self.store = store
        self.run_id = _now().strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(2)
        self.cache = cfg.data_dir / "cache"
        self._http: dict[str, HttpClient] = {}
        self._providers = None

    # ---------------- plumbing ----------------
    def http(self, name: str, min_interval_s: float = 0.25, user_agent: str = BROWSER_UA) -> HttpClient:
        if name not in self._http:
            self._http[name] = HttpClient(self.cache / name, min_interval_s, user_agent=user_agent)
        return self._http[name]

    def providers(self) -> list:
        if self._providers is None:
            make = {
                "fred_xa": lambda: FredCrossAssetProvider(self.http("fred", 0.2)),
                "yfinance": lambda: YFinanceProvider(),
                "fmp": lambda: FmpProvider.from_env(self.http("fmp", 0.3)),
                "massive": lambda: MassiveProvider.from_env(self.http("massive", self.cfg.get("data.massive_min_interval_s", 12.5))),
                "stooq": lambda: StooqProvider(self.http("stooq", 1.0)),
                "yahoo": lambda: YahooChartProvider(self.http("yahoo", 1.0)),
                "alphavantage": lambda: AlphaVantageProvider.from_env(self.http("alphavantage", 12.5)),
            }
            order = self.cfg.get("data.price_providers", list(make))
            self._providers = [make[n]() for n in order if n in make]
        return self._providers

    def _log(self, kind: str, symbol: str, provider: str, status: str, n_rows: int = 0, detail: str = "") -> None:
        self.store.append("ingestion_log", pd.DataFrame([{"run_id": self.run_id, "ts": _now(), "kind": kind, "symbol": symbol,
                                                          "provider": provider, "status": status, "n_rows": int(n_rows),
                                                          "detail": redact(detail)[:2000]}]))

    def _asset_class(self, symbol: str) -> str:
        s = symbol.upper()
        if s in XA_SPEC:
            return XA_SPEC[s][1]
        for g, cls in GROUP_CLASS.items():
            if s in [x.upper() for x in self.cfg.section("universe").get(g, [])]:
                return cls
        return "index" if s.startswith("^") else "equity"

    def _meta_row(self, symbol: str, **kw) -> dict:
        mkt = market_of(symbol)
        row = {"symbol": symbol, "asset_class": self._asset_class(symbol), "benchmark": self.cfg.benchmark_for(symbol),
               "market": mkt, "currency": "INR" if mkt == "IN" else "USD", "updated_at": _now()}
        if symbol.upper() in XA_SPEC:
            row.update(name=XA_SPEC[symbol.upper()][0], sector="Cross-asset proxy", note=XA_SPEC[symbol.upper()][2])
        row.update({k: v for k, v in kw.items() if v is not None})
        return row

    def _upsert_meta(self, rows: list[dict]) -> None:
        """Insert/refresh metadata without erasing fields (e.g. sector from SEC) that this call does not know."""
        if not rows:
            return
        new = pd.DataFrame(rows)
        old = self.store.symbols()
        if not old.empty:
            old = old.set_index("symbol")
            new = new.set_index("symbol")
            for c in old.columns:
                if c in new:
                    new[c] = new[c].where(new[c].notna(), old[c].reindex(new.index))
                else:
                    new[c] = old[c].reindex(new.index)
            new = new.reset_index()
        self.store.upsert("symbols", new, ["symbol"])

    def _validate(self, symbol: str, df: pd.DataFrame) -> list:
        issues = validate_prices(df, symbol, today=pd.Timestamp.today(), max_staleness_days=self.cfg.get("data.max_staleness_days", 5))
        if issues:
            self.store.append("validation_issues", pd.DataFrame([{"run_id": self.run_id, "ts": _now(), **i.as_dict()} for i in issues]))
        return issues

    def _write_prices(self, symbol: str, df: pd.DataFrame, provider: str, validate: bool = True) -> int:
        d = df.assign(symbol=symbol, fetched_at=_now())
        if validate:
            self._validate(symbol, d)
        n = self.store.upsert("prices", d, ["symbol", "date"])
        self._log("prices", symbol, provider, "OK", n)
        return n

    def _write_bulk(self, px: pd.DataFrame, source: str, adjustment: str, validate: bool = True,
                    progress: Progress = None) -> int:
        px = px.assign(source=source, adjustment=adjustment)
        n = self.store.upsert("prices", px[PRICE_COLS].assign(fetched_at=_now()), ["symbol", "date"])
        issues, logs = [], []
        for i, (sym, g) in enumerate(px.groupby("symbol")):
            if validate:
                issues += [{"run_id": self.run_id, "ts": _now(), **x.as_dict()} for x in validate_prices(g, sym, today=None)]
            logs.append({"run_id": self.run_id, "ts": _now(), "kind": "prices", "symbol": sym, "provider": source, "status": "OK",
                         "n_rows": len(g), "detail": ""})
            if progress and i % 500 == 0:
                progress(0.95, f"validated {i} symbols")
        if issues:
            self.store.append("validation_issues", pd.DataFrame(issues))
        self.store.append("ingestion_log", pd.DataFrame(logs))
        return n

    def _replace_membership(self, mem: pd.DataFrame, source: str) -> None:
        self.store.delete("universe_membership")
        self.store.upsert("universe_membership", mem.assign(source=source), ["date", "symbol"])

    # ---------------- prices ----------------
    def ingest_prices(self, symbols: list[str], start: str | None = None) -> dict[str, str]:
        start = start or self.cfg.get("data.history_start", "2005-01-01")
        out, meta = {}, []
        for sym in symbols:
            errors = []
            for p in self.providers():
                if not p.supports(sym):
                    continue
                try:
                    df = p.fetch_daily(sym, start)
                    if df.empty:
                        raise ProviderError("no rows")
                except ProviderError as e:
                    errors.append(f"{p.name}: {e}")
                    self._log("prices", sym, p.name, "ERROR", 0, str(e))
                    continue
                except Exception as e:  # a provider bug must not abort the whole run
                    errors.append(f"{p.name}: {type(e).__name__}: {e}")
                    self._log("prices", sym, p.name, "ERROR", 0, f"{type(e).__name__}: {e}")
                    continue
                self._write_prices(sym, df, p.name)
                meta.append(self._meta_row(sym, source=p.name, first_date=df["date"].min(), last_date=df["date"].max()))
                out[sym] = "OK"
                break
            else:
                out[sym] = "FAILED"
                self._log("prices", sym, "*", "FAILED", 0, " | ".join(errors) or "no provider supports this symbol")
                log(logger, 30, "all providers failed", symbol=sym, errors=redact(" | ".join(errors))[:500])
        self._upsert_meta(meta)
        return out

    # ---------------- macro ----------------
    def ingest_macro(self) -> dict[str, str]:
        from bigboss.data.providers.macro import FredProvider, TreasuryProvider, WorldBankProvider

        out = {}
        fred = FredProvider(self.http("fred", 0.2))
        for sid in self.cfg.get("macro.fred_series", []):
            out[sid] = self._macro_one(sid, "fred", lambda sid=sid: fred.fetch(sid))
        out["TREAS_AVG_RATE_MARKETABLE"] = self._macro_one("TREAS_AVG_RATE_MARKETABLE", "treasury",
                                                           lambda: TreasuryProvider(self.http("treasury", 0.2)).fetch())
        wb = WorldBankProvider(self.http("worldbank", 0.2))
        for ind in self.cfg.get("macro.worldbank_indicators", []):
            for c in self.cfg.get("macro.worldbank_countries", []):
                sid = f"WB:{c}:{ind}"
                out[sid] = self._macro_one(sid, "world_bank", lambda ind=ind, c=c: wb.fetch(ind, c))
        return out

    def _macro_one(self, sid: str, provider: str, fn) -> str:
        try:
            df = fn()
        except (ProviderError, ValueError, KeyError) as e:
            self._log("macro", sid, provider, "FAILED", 0, str(e))
            return "FAILED"
        n = self.store.upsert("macro", df, ["series_id", "date"])
        self._log("macro", sid, provider, "OK", n)
        return "OK"

    # ---------------- SEC ----------------
    def ingest_sec(self, symbols: list[str]) -> dict[str, str]:
        from bigboss.data.providers.sec import SecEdgarProvider, filings_events, fundamentals_rows

        ua = self.cfg.get("data.sec_user_agent", "")
        if not ua or "example.com" in ua:
            log(logger, 30, "set [data].sec_user_agent to a real contact address (SEC fair-access policy)")
        sec = SecEdgarProvider(self.http("sec", 0.12, user_agent=ua or "BigBoss research"))
        out, meta = {}, []
        for sym in symbols:
            if market_of(sym) != "US" or self._asset_class(sym) != "equity":
                continue
            try:
                facts = sec.company_facts(sym)
                rows = fundamentals_rows(sym, facts)
                n = self.store.upsert("fundamentals", rows, ["symbol", "metric", "period_end"]) if not rows.empty else 0
                subs = sec.submissions(sym)
                ev = filings_events(subs, sym)
                n_ev = self.store.upsert("events", ev, ["symbol", "accession"]) if not ev.empty else 0
                meta.append({"symbol": sym, "name": subs.get("name"), "sector": subs.get("sicDescription") or None,
                             "cik": str(subs.get("cik") or ""), "updated_at": _now()})
                self._log("sec", sym, "sec_edgar", "OK", n + n_ev, f"{n} fundamentals rows, {n_ev} events")
                out[sym] = "OK"
            except ProviderError as e:
                self._log("sec", sym, "sec_edgar", "FAILED", 0, str(e))
                out[sym] = "FAILED"
        self._upsert_meta(meta)
        return out

    # ---------------- news ----------------
    def ingest_news(self, symbols: list[str]) -> dict[str, str]:
        from bigboss.data.providers.news import FinnhubNews, GdeltNews, MassiveNews

        finnhub = FinnhubNews(self.http("finnhub", 1.1))
        massive = MassiveNews(self.http("massive_news", self.cfg.get("data.massive_min_interval_s", 12.5)))
        gdelt = GdeltNews(self.http("gdelt", 5.5))
        out = {}
        for sym in symbols:
            sources = []
            if finnhub.api_key and market_of(sym) == "US":
                sources.append(("finnhub", lambda: finnhub.fetch(sym)))
            if market_of(sym) == "US" and not sym.startswith(("XA:", "^")):
                sources.append(("massive", lambda: massive.fetch(sym)))
            sources.append(("gdelt", lambda: gdelt.fetch(sym)))
            for name, fn in sources:
                try:
                    items = fn()
                except ProviderError as e:
                    self._log("news", sym, name, "ERROR", 0, str(e))
                    continue
                df = pd.DataFrame(items)
                if df.empty:
                    continue
                df["symbol"] = sym
                n = self.store.upsert("news", df.assign(fetched_at=_now()), ["symbol", "url"])
                self._log("news", sym, name, "OK", n)
                out[sym] = "OK"
                break
            else:
                out[sym] = "FAILED"
        return out

    # ---------------- point-in-time universes ----------------
    def ingest_massive_market(self, top_n: int = 500, extra_symbols: list[str] | None = None, years: float = 2.0,
                              progress: Progress = None) -> dict:
        from bigboss.data.providers.massive_market import MassiveMarket, point_in_time_universe

        mm = MassiveMarket(self.http("massive", self.cfg.get("data.massive_min_interval_s", 12.5)), _env("MASSIVE_API_KEY", "POLYGON_API_KEY"))
        start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years) + pd.Timedelta(days=7)
        bars = mm.market_history(start, progress=progress)
        try:
            cs = set(mm.common_stocks())
            bars = bars[bars["symbol"].isin(cs)]
        except ProviderError as e:
            log(logger, 30, "common-stock filter unavailable; using all tickers", error=str(e))
        mem = point_in_time_universe(bars, top_n=top_n)
        members = set(mem["symbol"])
        px = bars[bars["symbol"].isin(members)].assign(adj_close=lambda x: x["close"])
        n = self._write_bulk(px, "massive_grouped", "split_adjusted_only (Massive grouped daily)", progress=progress)
        self._replace_membership(mem, "massive_grouped")
        self._upsert_meta([self._meta_row(s, asset_class="equity", source="massive_grouped") for s in sorted(members)])
        extra = self.ingest_prices(extra_symbols or []) if extra_symbols else {}
        return {"sessions": int(bars["date"].nunique()), "symbols_in_market": int(bars["symbol"].nunique()),
                "ever_in_universe": len(members), "price_rows": n, "extra": extra}

    def ingest_dolthub_market(self, export_dir: str, top_n: int = 500, start: str = "2014-04-01", extra_symbols: list[str] | None = None,
                              listing_csv: str | None = None, universe_from: str | None = "2018-09-07", verify_splits: bool = False,
                              progress: Progress = None) -> dict:
        from bigboss.data.providers.dolthub import (adjust_market, classify_with_listing, common_stocks, confirm_unrecorded_splits,
                                                    load_export, segment_symbols)
        from bigboss.data.providers.massive_market import point_in_time_universe

        say = progress or (lambda f, m: None)
        say(0.0, "loading DoltHub export")
        t = load_export(export_dir, start)
        cls = classify_with_listing(t["symbol"], pd.read_csv(listing_csv)) if listing_csv else common_stocks(t["symbol"])
        bars = t["ohlcv"][t["ohlcv"]["act_symbol"].isin(set(cls["act_symbol"]))]
        say(0.2, f"segmenting {bars['act_symbol'].nunique()} tickers")
        bars = segment_symbols(bars)
        say(0.3, "adjusting splits and dividends")
        px, st = adjust_market(bars, t["split"], t["dividend"])
        say(0.6, "building the point-in-time universe")
        mem = point_in_time_universe(px.assign(close=px["raw_close"], volume=px["volume"] * px["close"] / px["raw_close"]),
                                     top_n=top_n, start=universe_from)
        members = set(mem["symbol"])
        report = []
        if verify_splits:
            cands = [c for c in st["unrecorded_split_candidates"] if c[0] in members]
            say(0.7, f"verifying {len(cands)} large raw moves against yfinance")
            yfp = YFinanceProvider()
            fixes, report = confirm_unrecorded_splits(
                cands, lambda s, d: yfp.adjusted_close(s, str((pd.Timestamp(d) - pd.Timedelta(days=7)).date())))
            if not fixes.empty:
                fixed = set(fixes["act_symbol"])
                redo = bars[bars["symbol"].isin(fixed)]
                px2, _ = adjust_market(redo, pd.concat([t["split"], fixes], ignore_index=True), t["dividend"])
                px = pd.concat([px[~px["symbol"].isin(fixed)], px2], ignore_index=True)
        say(0.8, f"writing prices for {len(members)} universe members")
        n = self._write_bulk(px[px["symbol"].isin(members)], "dolthub", "adj_close_split_and_dividend (adjusted by BigBoss from raw)",
                             progress=progress)
        self._replace_membership(mem, "dolthub")
        names = cls.set_index("act_symbol")["security_name"].to_dict()
        self._upsert_meta([self._meta_row(s, asset_class="equity", source="dolthub", name=names.get(s.split("@")[0]),
                                          benchmark=self.cfg.get("benchmarks.US", "SPY")) for s in sorted(members)])
        extra = self.ingest_prices(extra_symbols or []) if extra_symbols else {}
        last = px.groupby("symbol")["date"].max()
        stopped = int((last[last.index.isin(members)] < px["date"].max() - pd.Timedelta(days=10)).sum())
        st_out = {k: v for k, v in st.items() if k != "unrecorded_split_candidates"}
        return {**st_out, "unrecorded_split_candidates": len(st["unrecorded_split_candidates"]),
                "verification": pd.Series([r["action"].split(" ")[0] for r in report]).value_counts().to_dict() if report else {},
                "ever_in_universe": len(members), "stopped_trading": stopped, "price_rows": n,
                "universe_from": str(mem["date"].min().date()) if not mem.empty else None, "extra": extra}

    def ingest_sp500_history(self, history_csv: str, dolthub_export: str, kaggle_dir: str | None = None, start: str = "2014-01-02",
                             verify_splits: bool = False, progress: Progress = None) -> dict:
        """Point-in-time S&P 500 from historical compositions, priced per day from DoltHub, else the Kaggle 2017 files."""
        from bigboss.data.providers.dolthub import adjust_market, confirm_unrecorded_splits, load_export, segment_symbols
        from bigboss.data.providers.index_membership import (assign_symbols, candidate_prices, coverage_by_year, daily_membership,
                                                             sp500_snapshots)

        say = progress or (lambda f, m: None)
        snap = sp500_snapshots(history_csv)
        say(0.05, "loading DoltHub export")
        t = load_export(dolthub_export, pd.Timestamp(start) - pd.Timedelta(days=400))
        days = pd.DatetimeIndex(sorted(t["ohlcv"]["date"].unique()))
        md = daily_membership(snap, days[days >= pd.Timestamp(start)])
        tickers = sorted(md["ticker"].unique())
        bars = t["ohlcv"][t["ohlcv"]["act_symbol"].isin(tickers)]
        px, st = adjust_market(segment_symbols(bars), t["split"], t["dividend"])
        report = []
        if verify_splits and st["unrecorded_split_candidates"]:
            yfp = YFinanceProvider()
            fixes, report = confirm_unrecorded_splits(
                st["unrecorded_split_candidates"], lambda s, d: yfp.adjusted_close(s, str((pd.Timestamp(d) - pd.Timedelta(days=7)).date())))
            if not fixes.empty:
                fixed = set(fixes["act_symbol"])
                px2, _ = adjust_market(segment_symbols(bars[bars["act_symbol"].isin(fixed)]),
                                       pd.concat([t["split"], fixes], ignore_index=True), t["dividend"])
                px = pd.concat([px[~px["act_symbol"].isin(fixed)], px2], ignore_index=True)
        say(0.5, "choosing a price source per member-day")
        cand = candidate_prices(tickers, px, kaggle_dir)
        md["symbol"] = assign_symbols(md, cand)
        cov = coverage_by_year(md, md["symbol"].notna())
        mem = md.dropna(subset=["symbol"]).copy()
        mem["rank"] = mem.groupby("date").cumcount() + 1
        used = set(mem["symbol"])
        prices = cand[cand["symbol"].isin(used)].copy()
        prices["adj_close"] = prices["adj_close"].fillna(prices["close"])
        say(0.8, f"writing {len(used)} price series")
        n = 0
        for src, g in prices.groupby("source"):
            adj = "adj_close_split_and_dividend (DoltHub, adjusted by BigBoss)" if src == "dolthub" else "total-return closes (Kaggle 2017)"
            n += self._write_bulk(g, src, adj, progress=progress)
        self._replace_membership(mem[["date", "symbol", "rank"]], "sp500_history")
        self._upsert_meta([self._meta_row(s, asset_class="equity", source="sp500_history", benchmark=self.cfg.get("benchmarks.US", "SPY"))
                           for s in sorted(used)])
        extra = self.ingest_prices([self.cfg.get("benchmarks.US", "SPY")])
        return {"snapshots": int(snap["date"].nunique()), "tickers": len(tickers), "priced_series": len(used), "price_rows": n,
                "member_day_coverage_by_year": {int(k): round(float(v), 4) for k, v in cov.items()},
                "verification": pd.Series([r["action"].split(" ")[0] for r in report]).value_counts().to_dict() if report else {},
                "extra": extra}

    def ingest_nifty_history(self, weights_csv: str, sectors_csv: str | None, plan: pd.DataFrame, progress: Progress = None) -> dict:
        """Point-in-time NIFTY 50 from historical constituent weights. `plan`: ticker, source (yfinance|nse), symbol.

        NSE archive prices are raw: split-like jumps (bonus issues, splits) are neutralised; other large moves (real crashes) kept.
        """
        from bigboss.data.providers.index_membership import coverage_by_year, daily_membership, nifty_snapshots
        from bigboss.data.providers.nse import NseProvider

        say = progress or (lambda f, m: None)
        snap = nifty_snapshots(weights_csv)
        bench = self.cfg.get("benchmarks.IN", "^NSEI")
        self.ingest_prices(["^NSEI", bench] if bench != "^NSEI" else ["^NSEI"], start=str(snap["date"].min().date()))
        cal = self.store.panel("adj_close", ["^NSEI"]).index
        days = cal if len(cal) else pd.bdate_range(snap["date"].min(), pd.Timestamp.today())
        md = daily_membership(snap, days)
        sectors = {}
        if sectors_csv:
            s = pd.read_csv(sectors_csv)
            s.columns = [c.strip().lower() for c in s.columns]
            tcol = next((c for c in s.columns if c in ("ticker", "symbol", "company")), s.columns[0])
            scol = next((c for c in s.columns if "sector" in c or "industry" in c), s.columns[-1])
            sectors = dict(zip(s[tcol].astype(str), s[scol].astype(str)))
        yfp, nse = YFinanceProvider(), NseProvider()
        sym_of, neutralised = {}, 0
        for i, r in enumerate(plan.itertuples()):
            say(i / max(len(plan), 1), f"prices for {r.ticker}")
            sym = r.symbol if str(r.symbol).endswith((".NS", ".BO")) else f"{r.symbol}.NS"
            try:
                if str(r.source).lower() == "nse":
                    df = nse.fetch_daily(sym, str(snap["date"].min().date()))
                    df, k = _neutralise_split_jumps(df)
                    neutralised += k
                    src = "nse"
                else:
                    df, src = yfp.fetch_daily(sym, str(snap["date"].min().date())), "yfinance"
            except ProviderError as e:
                self._log("prices", sym, str(r.source), "FAILED", 0, str(e))
                continue
            self._write_prices(sym, df, src)
            sym_of[r.ticker] = sym
        md["symbol"] = md["ticker"].map(sym_of)
        cov = coverage_by_year(md, md["symbol"].notna())
        mem = md.dropna(subset=["symbol"]).copy()
        mem["rank"] = mem.groupby("date")["weight"].rank(ascending=False, method="first").astype(int)
        self._replace_membership(mem[["date", "symbol", "rank", "weight"]], "nifty50_history")
        self._upsert_meta([self._meta_row(s, asset_class="equity", sector=sectors.get(t), benchmark=bench, source="nifty50_history")
                           for t, s in sym_of.items()])
        return {"snapshots": int(snap["date"].nunique()), "constituents": int(snap["ticker"].nunique()), "priced": len(sym_of),
                "split_jumps_neutralised": neutralised,
                "weight_coverage_by_year": {int(k): round(float(v), 4) for k, v in cov.items()}}

    def ingest_etf_file(self, csv: str | Path, meta_csv: str | Path | None = None, top_n: int = 100, start: str | None = None,
                        progress: Progress = None, beta_max: float = 1.6, beta_min: float = -0.3) -> dict:
        """Bulk ETF prices -> point-in-time ETF universe (top-N by trailing 60-day median dollar volume, price >= $5,
        >= 40 sessions). Leveraged, inverse and volatility products are excluded: category "Trading--*", or for
        uncategorized funds a beta to the benchmark above `beta_max` or below `beta_min`."""
        from bigboss.data.providers.local_files import load_etf_file, load_etf_meta
        from bigboss.data.providers.massive_market import point_in_time_universe

        say = progress or (lambda f, m: None)
        say(0.0, "reading the ETF file")
        bars = load_etf_file(csv, start)
        meta = load_etf_meta(meta_csv).set_index("symbol")
        bench = self.cfg.get("benchmarks.US", "SPY")
        cat = meta["category"].reindex(sorted(bars["symbol"].unique()))
        adj = bars.pivot(index="date", columns="symbol", values="adj_close").sort_index()
        bench_r = adj[bench].pct_change() if bench in adj else self.store.panel("adj_close", [bench]).get(bench, pd.Series(dtype=float)).pct_change()
        excluded = {}
        for s in adj.columns:
            if s == bench:
                continue
            c = cat.get(s)
            if isinstance(c, str) and c.startswith("Trading--"):
                excluded[s] = f"category {c}"
                continue
            if isinstance(c, str) and c.strip():
                continue
            j = pd.concat([adj[s].pct_change(), bench_r], axis=1, join="inner").dropna()
            if len(j) >= 60:
                b = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var())
                if b > beta_max or b < beta_min:
                    excluded[s] = f"beta {b:.2f} to {bench}"
        cands = bars[(bars["symbol"] != bench) & ~bars["symbol"].isin(excluded)]
        say(0.4, "building the point-in-time universe")
        mem = point_in_time_universe(cands, top_n=top_n, lookback=60, min_history=40)
        members = set(mem["symbol"])
        keep = bars[bars["symbol"].isin(members | {bench})]
        n = self._write_bulk(keep, "local_etf_file", "adj_close_split_and_dividend (bulk ETF file)", progress=progress)
        self._replace_membership(mem, "etf_file")
        rows = [self._meta_row(s, asset_class="etf", source="local_etf_file", benchmark=bench,
                               sector=cat.get(s) if isinstance(cat.get(s), str) else "Uncategorized ETF")
                for s in sorted(members | ({bench} if bench in adj else set()))]
        self._upsert_meta(rows)
        per_year = mem.groupby("date").size().groupby(lambda d: d.year).median().to_dict() if not mem.empty else {}
        return {"etfs_in_file": int(bars["symbol"].nunique()), "etfs_ever_in_universe": len(members),
                "excluded_trading_products": len(excluded), "excluded": excluded, "price_rows": n,
                "first_date": str(mem["date"].min().date()) if not mem.empty else None,
                "median_members_by_year": {int(k): float(v) for k, v in per_year.items()}}


def _neutralise_split_jumps(df: pd.DataFrame, threshold: float = 0.3) -> tuple[pd.DataFrame, int]:
    """Treat split-like raw jumps as corporate actions: rescale history before them so the series is continuous."""
    d = df.sort_values("date").reset_index(drop=True)
    c = d["close"].to_numpy(dtype=float)
    factor = np.ones(len(d))
    k_count = 0
    for k in range(1, len(c)):
        q = c[k] / c[k - 1] if c[k - 1] > 0 else np.nan
        if np.isfinite(q) and abs(q - 1) >= threshold and looks_like_split(q):
            factor[:k] *= q
            k_count += 1
    for f in ("open", "high", "low", "close", "adj_close"):
        d[f] = d[f] * factor
    d["volume"] = d["volume"] / factor
    if k_count:
        d["adjustment"] = "NSE raw closes; split-like jumps neutralised by BigBoss"
    return d, k_count
