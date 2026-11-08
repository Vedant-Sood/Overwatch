"""Macro data: FRED (keyless CSV, or the API with FRED_API_KEY), US Treasury Fiscal Data, and the World Bank.

Every value carries `available_at` = observation date + a conservative publication lag, so features only use values
that were public at the time. Latest vintages are used (not ALFRED real-time vintages): revisions can leak slightly.
"""

from __future__ import annotations

import io
import json
import os

import pandas as pd

from bigboss.data.providers.base import HttpClient, ProviderError

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
TREASURY = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
WORLD_BANK = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

# Monthly releases are published weeks after the observation month; daily market series by the next morning.
MONTHLY_LAG_DAYS = {"CPIAUCSL": 45, "UNRATE": 37}
DAILY_LAG_DAYS = 1


def publication_lag(series_id: str, freq_days: float) -> pd.Timedelta:
    if series_id in MONTHLY_LAG_DAYS:
        return pd.Timedelta(days=MONTHLY_LAG_DAYS[series_id])
    if freq_days >= 25:  # other monthly series: month + ~6 weeks
        return pd.Timedelta(days=45)
    return pd.Timedelta(days=DAILY_LAG_DAYS)


class FredProvider:
    def __init__(self, http: HttpClient, api_key: str | None = None):
        self.http = http
        self.api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY")

    def fetch(self, series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
        if self.api_key:
            body, fetched = self.http.get(FRED_API, {"series_id": series_id, "api_key": self.api_key, "file_type": "json",
                                                     "observation_start": start}, max_age_s=6 * 3600)
            try:
                obs = json.loads(body)["observations"]
            except (json.JSONDecodeError, KeyError) as e:
                raise ProviderError(f"FRED API {series_id}: {body[:120]!r}") from e
            df = pd.DataFrame(obs)[["date", "value"]]
        else:
            body, fetched = self.http.get(FRED_CSV, {"id": series_id}, max_age_s=6 * 3600)
            df = pd.read_csv(io.StringIO(body))
            if df.shape[1] < 2:
                raise ProviderError(f"FRED {series_id}: unexpected CSV {body[:120]!r}")
            df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        df = df[df["date"] >= pd.Timestamp(start)].sort_values("date")
        if df.empty:
            raise ProviderError(f"FRED {series_id}: no observations")
        freq = df["date"].diff().dt.days.median() if len(df) > 2 else 1
        return pd.DataFrame({"series_id": series_id, "date": df["date"], "value": df["value"],
                             "available_at": df["date"] + publication_lag(series_id, freq), "source": "fred",
                             "fetched_at": fetched})


class TreasuryProvider:
    """Average interest rates on marketable Treasury debt (monthly; published early the next month)."""

    def __init__(self, http: HttpClient):
        self.http = http

    def fetch(self, start: str = "2001-01-01") -> pd.DataFrame:
        body, _ = self.http.get(TREASURY, {"filter": f"record_date:gte:{start},security_type_desc:eq:Marketable",
                                           "fields": "record_date,security_desc,avg_interest_rate_amt",
                                           "page[size]": 10000, "sort": "record_date"}, max_age_s=86400)
        try:
            data = pd.DataFrame(json.loads(body)["data"])
        except (json.JSONDecodeError, KeyError) as e:
            raise ProviderError(f"Treasury: {body[:120]!r}") from e
        if data.empty:
            raise ProviderError("Treasury: no rows")
        data = data[data["security_desc"].str.contains("Total Marketable", case=False, na=False)]
        d = pd.to_datetime(data["record_date"])
        return pd.DataFrame({"series_id": "TREAS_AVG_RATE_MARKETABLE", "date": d,
                             "value": pd.to_numeric(data["avg_interest_rate_amt"], errors="coerce"),
                             "available_at": d + pd.Timedelta(days=10), "source": "treasury_fiscal_data"}).dropna(subset=["value"])


class WorldBankProvider:
    """Annual indicators per country. A year's value is assumed public by the end of June of the next year."""

    def __init__(self, http: HttpClient):
        self.http = http

    def fetch(self, indicator: str, country: str) -> pd.DataFrame:
        body, _ = self.http.get(WORLD_BANK.format(country=country, indicator=indicator), {"format": "json", "per_page": 200},
                                max_age_s=7 * 86400)
        try:
            data = json.loads(body)
            rows = data[1] or []
        except (json.JSONDecodeError, IndexError, TypeError) as e:
            raise ProviderError(f"World Bank {indicator}/{country}: {body[:120]!r}") from e
        df = pd.DataFrame([{"year": int(r["date"]), "value": r["value"]} for r in rows if r.get("value") is not None])
        if df.empty:
            raise ProviderError(f"World Bank {indicator}/{country}: no values")
        d = pd.to_datetime(df["year"].astype(str) + "-12-31")
        return pd.DataFrame({"series_id": f"WB:{country}:{indicator}", "date": d, "value": df["value"].astype(float),
                             "available_at": d + pd.Timedelta(days=182), "source": "world_bank"}).sort_values("date")
