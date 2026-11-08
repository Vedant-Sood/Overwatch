"""DoltHub `post-no-preference/stocks`: whole-US-market end-of-day bars, splits, dividends and the symbol directory.

Keyless; requires the `dolt` CLI. Prices are RAW (unadjusted), so BigBoss adjusts them itself:
* every recorded split is checked against the raw price jump near its date (bogus records dropped, misdated ones moved);
* dividends are applied multiplicatively (Yahoo/CRSP style); implausible ones (> 25% of the prior close) are dropped;
* remaining raw moves beyond ±30% are reported as unrecorded-split candidates, which `confirm_unrecorded_splits`
  verifies against an independent source over the same span.
Ticker reuse is handled by splitting a ticker's history at gaps longer than 30 days (older pieces become TICKER@YYYYMM).
Daily full-market collection began on 2018-09-07: before that only names that survived to 2018 are present.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bigboss.data.providers.base import ProviderError
from bigboss.data.validation import looks_like_split

REPO = "post-no-preference/stocks"
TABLES = ("ohlcv", "split", "dividend", "symbol")
FULL_MARKET_FROM = "2018-09-07"

_EXCLUDE = re.compile(r"\b(?:warrants?|preferred|pfd|units?|rights?|depositary|notes?|debentures?|bonds?|etf|etn|fund|"
                      r"subordinated|trust preferred|test)\b", re.I)
_INCLUDE = re.compile(r"common stock|common shares|ordinary shares|class [a-c] (?:common|ordinary)", re.I)


def _truthy(x) -> bool:
    return str(x).strip().lower() in ("1", "true", "y", "yes", "t")


# ---------------- sync / load ----------------
def sync(repo_dir: str | Path, export_dir: str | Path, dolt: str | None = None) -> dict:
    """Clone (or pull) the DoltHub repository and export its tables to Parquet."""
    exe = dolt or shutil.which("dolt")
    if not exe:
        raise ProviderError("the `dolt` CLI is not installed (https://docs.dolthub.com/introduction/installation)")
    repo, out = Path(repo_dir), Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (repo / ".dolt").exists():
        subprocess.run([exe, "pull"], cwd=repo, check=True)
        action = "pulled"
    else:
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([exe, "clone", REPO, str(repo)], check=True)
        action = "cloned"
    counts = {}
    for t in TABLES:
        f = out / f"{t}.parquet"
        subprocess.run([exe, "table", "export", "-f", t, str(f)], cwd=repo, check=True)
        counts[t] = _parquet_rows(f)
    return {"repo": str(repo), "export_dir": str(out), "action": action, "rows": counts}


def _parquet_rows(f: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(f).metadata.num_rows)


def load_export(export_dir: str | Path, start=None) -> dict[str, pd.DataFrame]:
    d = Path(export_dir)
    missing = [t for t in TABLES if not (d / f"{t}.parquet").exists()]
    if missing:
        raise ProviderError(f"DoltHub export {d} lacks {missing} - run `bigboss dolthub-sync`")
    filters = [("date", ">=", pd.Timestamp(start).date())] if start is not None else None
    bars = pd.read_parquet(d / "ohlcv.parquet", filters=filters)
    out = {"ohlcv": bars}
    for t in ("split", "dividend", "symbol"):
        out[t] = pd.read_parquet(d / f"{t}.parquet")
    out["ohlcv"]["date"] = pd.to_datetime(out["ohlcv"]["date"])
    for t in ("split", "dividend"):
        out[t]["ex_date"] = pd.to_datetime(out[t]["ex_date"])
    return out


# ---------------- classification ----------------
def common_stocks(sym: pd.DataFrame) -> pd.DataFrame:
    """Operating companies' common stock (delisted ones included); no ETFs, ADRs, warrants, units, preferreds or test issues."""
    name = sym["security_name"].fillna("")
    keep = (~sym["is_etf"].map(_truthy) & ~sym["is_test_issue"].map(_truthy) & name.str.contains(_INCLUDE)
            & ~name.str.contains(_EXCLUDE) & ~name.str.contains("%", regex=False))
    return sym[keep].reset_index(drop=True)


def classify_with_listing(sym: pd.DataFrame, listing: pd.DataFrame) -> pd.DataFrame:
    """Common stocks from DoltHub's directory plus names it never saw (it starts Oct 2017), from Alpha Vantage LISTING_STATUS."""
    base = common_stocks(sym)[["act_symbol", "security_name"]].assign(class_source="dolthub_symbol")
    known = set(sym["act_symbol"])
    name = listing["name"].fillna("")
    extra = listing[(listing["assetType"].str.lower() == "stock") & ~listing["symbol"].isin(known)
                    & ~name.str.contains(_EXCLUDE) & ~name.str.contains("%", regex=False)]
    extra = extra.drop_duplicates("symbol").rename(columns={"symbol": "act_symbol", "name": "security_name"})
    return pd.concat([base, extra[["act_symbol", "security_name"]].assign(class_source="alphavantage_listing")],
                     ignore_index=True)


def segment_symbols(bars: pd.DataFrame, max_gap_days: int = 30) -> pd.DataFrame:
    """Split each ticker's history at gaps > max_gap_days. The latest piece keeps the ticker; older ones get TICKER@YYYYMM."""
    b = bars.sort_values(["act_symbol", "date"]).reset_index(drop=True)
    gap = b.groupby("act_symbol")["date"].diff().dt.days > max_gap_days
    seg = gap.groupby(b["act_symbol"]).cumsum()
    last = seg.groupby(b["act_symbol"]).transform("max")
    start = b.groupby([b["act_symbol"], seg])["date"].transform("min")
    b["symbol"] = np.where(seg == last, b["act_symbol"], b["act_symbol"] + "@" + start.dt.strftime("%Y%m"))
    return b


# ---------------- adjustment ----------------
def _match_split(c: np.ndarray, i: int, ratio: float, window: int, used: set, tol: float = 0.2):
    """Index k near i where the raw close jumps by ~ratio (c[k-1]/c[k]); prefers k == i."""
    best = None
    for k in sorted(range(max(1, i - window), min(len(c) - 1, i + window) + 1), key=lambda k: abs(k - i)):
        if k in used or not (c[k] > 0 and c[k - 1] > 0):
            continue
        if abs((c[k - 1] / c[k]) / ratio - 1) <= tol:
            best = k
            break
    return best


def adjust_market(bars: pd.DataFrame, splits: pd.DataFrame, divs: pd.DataFrame, window: int = 5, implausible_yield: float = 0.25,
                  move_threshold: float = 0.3) -> tuple[pd.DataFrame, dict]:
    """Split- and dividend-adjust segmented raw bars. Returns (prices, stats).

    prices: symbol, act_symbol, date, open, high, low, close (split-adjusted), adj_close (total return), volume
    (split-adjusted), raw_close.
    """
    st = {"splits_used": 0, "splits_moved": 0, "splits_dropped_unconfirmed": 0, "splits_dropped_duplicate": 0,
          "dividends_used": 0, "dividends_dropped_implausible": 0, "unrecorded_split_candidates": []}
    sp = splits.copy()
    if not sp.empty:
        sp["ratio"] = pd.to_numeric(sp["to_factor"], errors="coerce") / pd.to_numeric(sp["for_factor"], errors="coerce")
        sp = sp[np.isfinite(sp["ratio"]) & (sp["ratio"] > 0) & (sp["ratio"] != 1)]
    dv = divs.copy()
    if not dv.empty:
        dv["amount"] = pd.to_numeric(dv["amount"], errors="coerce")
        dv = dv[dv["amount"] > 0]
    sp_by = {k: g for k, g in sp.groupby("act_symbol")} if not sp.empty else {}
    dv_by = {k: g for k, g in dv.groupby("act_symbol")} if not dv.empty else {}

    out = []
    for sym, g in bars.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        act = g["act_symbol"].iloc[0] if "act_symbol" in g else sym
        dates = g["date"].to_numpy()
        c = g["close"].to_numpy(dtype=float)
        n = len(g)
        lo, hi = dates[0], dates[-1]
        factor_at = np.ones(n)  # split ratio taking effect at index k
        used: set[int] = set()
        recs = pd.concat([x for x in (sp_by.get(act), sp_by.get(sym) if sym != act else None) if x is not None]) \
            if (act in sp_by or sym in sp_by) else pd.DataFrame()
        for r in recs.sort_values("ex_date").itertuples() if not recs.empty else []:
            ex = np.datetime64(r.ex_date)
            if ex <= lo or ex > hi + np.timedelta64(window, "D"):
                continue
            i = int(np.searchsorted(dates, ex))
            if i >= n:
                continue
            k = _match_split(c, i, r.ratio, window, used)
            if k is None:
                dup = any(abs((c[j - 1] / c[j]) / r.ratio - 1) <= 0.2 for j in used)
                st["splits_dropped_duplicate" if dup else "splits_dropped_unconfirmed"] += 1
                continue
            used.add(k)
            factor_at[k] *= r.ratio
            st["splits_used"] += 1
            if k != i:
                st["splits_moved"] += 1
        # F[t] = product of split ratios taking effect after t
        F = np.r_[np.cumprod(factor_at[::-1])[::-1][1:], 1.0]
        adj = {f: g[f].to_numpy(dtype=float) / F for f in ("open", "high", "low", "close") if f in g}
        vol = g["volume"].to_numpy(dtype=float) * F if "volume" in g else np.full(n, np.nan)
        # dividends: multiplicative total-return factor, yield measured on the ex-date share basis
        div_at = np.ones(n)
        drec = dv_by.get(act)
        if drec is not None:
            amt = drec.groupby("ex_date")["amount"].sum()
            for ex, a in amt.items():
                ex = np.datetime64(ex)
                if ex <= lo or ex > hi:
                    continue
                i = int(np.searchsorted(dates, ex))
                if i >= n or i == 0:
                    continue
                prev = adj["close"][i - 1] * F[i]
                y = a / prev if prev > 0 else np.inf
                if not np.isfinite(y) or y > implausible_yield:
                    st["dividends_dropped_implausible"] += 1
                    continue
                div_at[i] *= 1 - y
                st["dividends_used"] += 1
        G = np.r_[np.cumprod(div_at[::-1])[::-1][1:], 1.0]
        close = adj["close"]
        q = close[1:] / close[:-1]
        for k in np.where(np.abs(q - 1) >= move_threshold)[0] + 1:
            ratio = float(close[k] / close[k - 1])
            st["unrecorded_split_candidates"].append((sym, str(pd.Timestamp(dates[k]).date()), round(ratio, 4),
                                                      str(pd.Timestamp(dates[k - 1]).date()), bool(looks_like_split(ratio))))
        out.append(pd.DataFrame({"symbol": sym, "act_symbol": act, "date": dates, **{f: adj[f] for f in adj},
                                 "adj_close": close * G, "volume": vol, "raw_close": c}))
    px = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["symbol", "act_symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "raw_close"])
    return px, st


def confirm_unrecorded_splits(cands: list[tuple], fetch: Callable[[str, str], pd.Series], tol: float = 0.05):
    """Check each raw move against an independent adjusted series over the SAME span (prev_date -> date).

    A move the other source also shows is real and kept; a move it does not show is an unrecorded split/conversion and
    becomes a corrective split record (to_factor = true price ratio / raw ratio). Names without independent data are kept.
    Returns (fixes: act_symbol, ex_date, to_factor, for_factor; report: list of dicts).
    """
    fixes, rep = [], []
    for c in cands:
        sym, date, ratio, prev = c[0], c[1], float(c[2]), c[3]
        row = {"symbol": sym, "date": date, "raw_ratio": ratio, "prev_date": prev}
        try:
            s = fetch(sym, prev)
            y0, y1 = s.get(pd.Timestamp(prev)), s.get(pd.Timestamp(date))
        except Exception as e:  # no independent data (delisted, renamed, provider error)
            rep.append({**row, "action": f"kept (no independent data: {type(e).__name__})"})
            continue
        if y0 is None or y1 is None or not (np.isfinite(y0) and np.isfinite(y1)) or y0 <= 0:
            rep.append({**row, "action": "kept (independent source lacks one of the two dates)"})
            continue
        true = float(y1 / y0)
        row["independent_ratio"] = round(true, 4)
        if abs(true / ratio - 1) <= tol:
            rep.append({**row, "action": "kept (real move, confirmed by the independent source)"})
        else:
            fixes.append({"act_symbol": sym, "ex_date": pd.Timestamp(date), "to_factor": true / ratio, "for_factor": 1.0})
            rep.append({**row, "action": f"corrected (unrecorded split/conversion x{true / ratio:.4f})"})
    return pd.DataFrame(fixes, columns=["act_symbol", "ex_date", "to_factor", "for_factor"]), rep
