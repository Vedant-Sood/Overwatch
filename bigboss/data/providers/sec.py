"""SEC EDGAR: XBRL company facts (point-in-time fundamentals) and filing events (submissions API).

Point-in-time rules:
* A fact is usable the day AFTER its `filed` date (`available` = filed; readers require available < as_of).
* For each period the FIRST-filed value is kept, so later restatements cannot leak backwards.
* Flow metrics (revenue, income, cash flow) are stored as discrete quarters derived from quarterly and
  year-to-date facts (Q4 = FY - 9M YTD); balance-sheet metrics are stored as instants.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bigboss.data.providers.base import HttpClient, ProviderError, to_ny_naive

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# metric -> (kind, [(taxonomy, concept), ...] in order of preference)
CONCEPTS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "revenue": ("flow", [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                         ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"), ("us-gaap", "SalesRevenueNet"),
                         ("us-gaap", "RevenuesNetOfInterestExpense")]),
    "net_income": ("flow", [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss"), ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic")]),
    "operating_income": ("flow", [("us-gaap", "OperatingIncomeLoss")]),
    "gross_profit": ("flow", [("us-gaap", "GrossProfit")]),
    "pretax_income": ("flow", [("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
                               ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments")]),
    "income_tax": ("flow", [("us-gaap", "IncomeTaxExpenseBenefit")]),
    "cfo": ("flow", [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
                     ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")]),
    "capex": ("flow", [("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"), ("us-gaap", "PaymentsToAcquireProductiveAssets")]),
    "equity": ("instant", [("us-gaap", "StockholdersEquity"),
                           ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")]),
    "total_assets": ("instant", [("us-gaap", "Assets")]),
    "cash": ("instant", [("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
                         ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")]),
    "lt_debt": ("instant", [("us-gaap", "LongTermDebtNoncurrent"), ("us-gaap", "LongTermDebt")]),
    "st_debt": ("instant", [("us-gaap", "LongTermDebtCurrent"), ("us-gaap", "DebtCurrent"), ("us-gaap", "ShortTermBorrowings")]),
    "current_assets": ("instant", [("us-gaap", "AssetsCurrent")]),
    "current_liabilities": ("instant", [("us-gaap", "LiabilitiesCurrent")]),
}

# 8-K items -> (event_type, importance prior). The most important item of a filing names the event.
EIGHT_K_ITEMS = {
    "1.01": ("material_agreement", 0.5), "1.02": ("agreement_terminated", 0.5), "1.03": ("bankruptcy", 1.0),
    "1.05": ("cybersecurity_incident", 0.6), "2.01": ("acquisition_or_disposition", 0.6), "2.02": ("earnings_release", 0.8),
    "2.03": ("new_debt_obligation", 0.4), "2.04": ("debt_acceleration", 0.7), "2.05": ("restructuring", 0.6),
    "2.06": ("impairment", 0.6), "3.01": ("delisting_notice", 0.8), "3.02": ("unregistered_equity_sale", 0.4),
    "3.03": ("security_holder_rights_change", 0.4), "4.01": ("auditor_change", 0.6), "4.02": ("non_reliance_restatement", 0.9),
    "5.01": ("change_of_control", 0.8), "5.02": ("executive_change", 0.5), "5.03": ("bylaw_change", 0.2),
    "5.07": ("shareholder_vote", 0.2), "7.01": ("reg_fd_disclosure", 0.3), "8.01": ("other_event", 0.3),
}
FORMS = {
    "10-K": ("annual_report", 0.5), "10-K/A": ("annual_report_amended", 0.5), "10-Q": ("quarterly_report", 0.4),
    "10-Q/A": ("quarterly_report_amended", 0.4), "4": ("insider_transaction", 0.2), "SC 13D": ("activist_stake", 0.6),
    "SC 13D/A": ("activist_stake", 0.5), "SC 13G": ("passive_stake", 0.2), "SC 13G/A": ("passive_stake", 0.1),
    "DEF 14A": ("proxy_statement", 0.2), "S-1": ("registration", 0.3), "S-3": ("shelf_registration", 0.3),
    "424B5": ("securities_offering", 0.4), "6-K": ("foreign_issuer_report", 0.3), "20-F": ("annual_report", 0.5),
    "8-K": ("other_event", 0.3),
}


class SecEdgarProvider:
    """SEC fair-access policy: a descriptive User-Agent with contact details and <= 10 requests per second."""

    def __init__(self, http: HttpClient):
        self.http = http
        self._cik: dict[str, int] | None = None

    def cik_map(self) -> dict[str, int]:
        if self._cik is None:
            body, _ = self.http.get(TICKERS_URL, max_age_s=7 * 86400)
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                raise ProviderError(f"SEC ticker map: unexpected response ({e})") from e
            self._cik = {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in data.values()}
        return self._cik

    def cik(self, symbol: str) -> int:
        s = symbol.upper().replace("-", ".")
        m = self.cik_map()
        if s not in m:
            raise ProviderError(f"{symbol}: not in the SEC ticker map (non-US, ETF, or delisted)")
        return m[s]

    def _json(self, url: str, max_age_s: float) -> dict:
        body, _ = self.http.get(url, max_age_s=max_age_s)
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise ProviderError(f"{url}: not JSON ({e})") from e

    def company_facts(self, symbol: str, max_age_s: float = 86400) -> dict:
        return self._json(FACTS_URL.format(cik=self.cik(symbol)), max_age_s)

    def submissions(self, symbol: str, max_age_s: float = 3600) -> dict:
        return self._json(SUBMISSIONS_URL.format(cik=self.cik(symbol)), max_age_s)


# ---------------- XBRL facts ----------------
def _facts_frame(facts: dict, taxonomy: str, concept: str) -> pd.DataFrame:
    """All reported values of one concept (first unit: USD, shares, or USD/shares) as a frame."""
    node = facts.get("facts", {}).get(taxonomy, {}).get(concept)
    units = (node or {}).get("units") or {}
    if not units:
        return pd.DataFrame(columns=["start", "end", "value", "filed", "accn", "form", "fy", "fp"])
    unit = "USD" if "USD" in units else next(iter(units))
    df = pd.DataFrame(units[unit])
    if df.empty:
        return pd.DataFrame(columns=["start", "end", "value", "filed", "accn", "form", "fy", "fp"])
    df = df.rename(columns={"val": "value"})
    for c in ("start", "accn", "form", "fy", "fp"):
        if c not in df:
            df[c] = None
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["end", "filed", "value"])
    df["concept"] = concept
    return df[["start", "end", "value", "filed", "accn", "form", "fy", "fp", "concept"]].reset_index(drop=True)


def _first_filed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values("filed", kind="stable").drop_duplicates(keys, keep="first")


def discrete_quarters(df: pd.DataFrame) -> pd.DataFrame:
    """Discrete fiscal quarters (start, end, value, available) from quarterly, year-to-date and annual facts.

    A YTD fact minus the previous YTD fact with the same fiscal-year start gives a quarter; its availability is the
    later of the two filing dates. Directly reported quarters win over derived ones. First-filed values only.
    """
    cols = ["start", "end", "value", "available"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = _first_filed(df.dropna(subset=["start"]), ["start", "end"]).copy()
    d["days"] = (d["end"] - d["start"]).dt.days
    direct = d[d["days"].between(80, 100)].assign(available=lambda x: x["filed"], derived=False)
    derived = []
    for start, g in d.groupby("start"):
        g = g.sort_values("end")
        prev = None
        for r in g.itertuples():
            if prev is not None and 80 <= (r.end - prev.end).days <= 100 and r.days > 100:
                derived.append({"start": prev.end + pd.Timedelta(days=1), "end": r.end, "value": r.value - prev.value,
                                "available": max(r.filed, prev.filed), "derived": True})
            prev = r
    q = pd.concat([direct[["start", "end", "value", "available", "derived"]], pd.DataFrame(derived)], ignore_index=True) \
        if derived else direct[["start", "end", "value", "available", "derived"]]
    if q.empty:
        return pd.DataFrame(columns=cols)
    q = q.sort_values(["end", "derived", "available"]).drop_duplicates("end", keep="first")
    return q[cols].sort_values("end").reset_index(drop=True)


def instants(df: pd.DataFrame) -> pd.DataFrame:
    """Balance-sheet values per period end, first-filed."""
    if df.empty:
        return pd.DataFrame(columns=["start", "end", "value", "available"])
    d = _first_filed(df, ["end"])
    return d.assign(start=pd.NaT, available=d["filed"])[["start", "end", "value", "available"]].sort_values("end").reset_index(drop=True)


def ttm_as_of(q: pd.DataFrame, as_of: pd.Timestamp, max_age_days: int = 200) -> float:
    """Sum of the latest four consecutive discrete quarters usable at `as_of` (available < as_of); NaN otherwise."""
    if q is None or q.empty:
        return np.nan
    u = q[q["available"] < as_of].sort_values("end").drop_duplicates("end", keep="last")
    if len(u) < 4:
        return np.nan
    last = u.tail(4)
    if (as_of - last["end"].iloc[-1]).days > max_age_days:
        return np.nan
    gaps = last["end"].diff().dt.days.iloc[1:]
    if not gaps.between(80, 100).all():
        return np.nan
    return float(last["value"].sum())


def latest_instant(s: pd.DataFrame, as_of: pd.Timestamp, max_age_days: int = 400) -> float:
    if s is None or s.empty:
        return np.nan
    u = s[s["available"] < as_of]
    if u.empty:
        return np.nan
    r = u.sort_values("end").iloc[-1]
    if (as_of - r["end"]).days > max_age_days:
        return np.nan
    return float(r["value"])


def share_count_series(facts: dict) -> pd.DataFrame:
    """Shares outstanding per period end (end, value, available).

    Cover-page counts (dei) are reported per share class in one filing: the classes of the first filing for a date are
    summed, later re-filings ignored. Without them, quarterly weighted-average diluted (then basic) shares are used.
    """
    cover = _facts_frame(facts, "dei", "EntityCommonStockSharesOutstanding")
    if not cover.empty:
        first = cover.sort_values("filed").groupby("end")["accn"].first()
        c = cover[cover.apply(lambda r: r["accn"] == first.get(r["end"]), axis=1)]
        s = c.groupby("end").agg(value=("value", "sum"), available=("filed", "min")).reset_index()
        return s[["end", "value", "available"]].sort_values("end").reset_index(drop=True)
    for concept in ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"):
        f = _facts_frame(facts, "us-gaap", concept)
        if f.empty:
            continue
        f = f.dropna(subset=["start"])
        f = f[(f["end"] - f["start"]).dt.days.between(80, 100)]
        if f.empty:
            continue
        f = _first_filed(f, ["end"])
        return f.rename(columns={"filed": "available"})[["end", "value", "available"]].sort_values("end").reset_index(drop=True)
    return pd.DataFrame(columns=["end", "value", "available"])


def normalize_company_facts(facts: dict) -> dict[str, pd.DataFrame]:
    """metric -> frame(start, end, value, available, concept). Concepts without values are skipped."""
    out: dict[str, pd.DataFrame] = {}
    for metric, (kind, candidates) in CONCEPTS.items():
        for tax, concept in candidates:
            raw = _facts_frame(facts, tax, concept)
            if raw.empty:
                continue
            s = discrete_quarters(raw) if kind == "flow" else instants(raw)
            if s.empty:
                continue
            out[metric] = s.assign(concept=concept, kind=kind)
            break
    sh = share_count_series(facts)
    if not sh.empty:
        out["shares"] = sh.assign(start=pd.NaT, concept="shares_outstanding", kind="instant")
    return out


def fundamentals_rows(symbol: str, facts: dict) -> pd.DataFrame:
    """Store rows for the `fundamentals` table."""
    parts = []
    for metric, s in normalize_company_facts(facts).items():
        parts.append(pd.DataFrame({"symbol": symbol, "metric": metric, "period_start": s["start"], "period_end": s["end"],
                                   "value": s["value"].astype(float), "available": s["available"], "kind": s["kind"],
                                   "concept": s["concept"], "source": "sec_edgar"}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------- filing events ----------------
def classify_filing(form: str, items: str) -> tuple[str, float]:
    form = (form or "").strip()
    if form.startswith("8-K"):
        best = ("other_event", 0.3)
        for it in [x.strip() for x in (items or "").split(",") if x.strip()]:
            t = EIGHT_K_ITEMS.get(it)
            if t and t[1] > best[1]:
                best = t
        return best
    return FORMS.get(form, ("other_filing", 0.1))


def filings_events(subs: dict, symbol: str | None = None) -> pd.DataFrame:
    """Recent filings -> events with acceptance time in New York wall-clock time."""
    rec = subs.get("filings", {}).get("recent", {})
    if not rec or not rec.get("accessionNumber"):
        return pd.DataFrame()
    df = pd.DataFrame({k: rec.get(k) for k in ("accessionNumber", "filingDate", "acceptanceDateTime", "form", "items",
                                                 "primaryDocument", "primaryDocDescription")})
    df["items"] = df["items"].fillna("")
    cls = [classify_filing(f, i) for f, i in zip(df["form"], df["items"])]
    cik = int(subs.get("cik", 0) or 0)
    out = pd.DataFrame({
        "symbol": symbol, "accession": df["accessionNumber"], "form": df["form"], "filed": pd.to_datetime(df["filingDate"]),
        "accepted_at": to_ny_naive(df["acceptanceDateTime"]), "items": df["items"],
        "event_type": [c[0] for c in cls], "importance_prior": [c[1] for c in cls],
        "description": df["primaryDocDescription"].fillna(""),
        "url": [f"https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace('-', '')}/{d}"
                for a, d in zip(df["accessionNumber"], df["primaryDocument"].fillna(""))],
        "credibility": "primary_regulatory_filing", "source": "sec_edgar",
    })
    return out.reset_index(drop=True)
