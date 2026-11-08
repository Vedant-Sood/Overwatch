"""News headlines for the research agents (never a model input): Finnhub, Massive, and GDELT DOC 2.0.

Sentiment is a transparent headline lexicon score in [-1, 1]; it is a weak signal and labelled as such.
Times are stored as New York wall-clock time.
"""

from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

import pandas as pd

from bigboss.data.providers.base import HttpClient, ProviderError, to_ny_naive

POSITIVE = {"beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies", "gain", "gains", "record",
            "upgrade", "upgrades", "upgraded", "strong", "growth", "profit", "profits", "raise", "raises", "raised", "outperform",
            "bullish", "rebound", "rebounds", "boost", "boosts", "approval", "approved", "win", "wins", "expand", "expands", "tops",
            "exceeds", "rise", "rises", "climb", "climbs", "higher", "optimistic", "buyback", "dividend"}
NEGATIVE = {"miss", "misses", "missed", "slump", "slumps", "plunge", "plunges", "fall", "falls", "drop", "drops", "decline",
            "declines", "downgrade", "downgrades", "downgraded", "weak", "loss", "losses", "cut", "cuts", "lawsuit", "probe",
            "investigation", "fraud", "recall", "bankruptcy", "default", "layoffs", "warning", "warns", "bearish", "sink", "sinks",
            "tumble", "tumbles", "lower", "slowdown", "fined", "halt", "halts", "crash", "crashes", "concern", "concerns",
            "delay", "delays", "selloff", "sell-off", "short", "subpoena", "resigns"}
_WORD = re.compile(r"[a-z][a-z\-]*")

MAJOR_NEWSWIRES = {"reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com", "marketwatch.com",
                   "nytimes.com", "barrons.com", "economist.com", "dowjones.com", "nikkei.com", "economictimes.indiatimes.com",
                   "livemint.com", "business-standard.com", "afp.com"}
PRESS_RELEASE = {"businesswire.com", "prnewswire.com", "globenewswire.com", "accesswire.com", "newswire.ca"}
SOCIAL = {"twitter.com", "x.com", "reddit.com", "stocktwits.com", "facebook.com", "youtube.com", "tiktok.com"}


def lexicon_sentiment(text: str) -> float:
    words = _WORD.findall((text or "").lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def domain_of(url_or_domain: str) -> str:
    s = url_or_domain or ""
    host = urlparse(s).netloc if "://" in s else s
    host = host.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def credibility_for(domain: str) -> str:
    d = domain_of(domain)
    if d == "sec.gov" or d.endswith(".sec.gov"):
        return "primary_regulatory_filing"
    if any(d == m or d.endswith("." + m) for m in MAJOR_NEWSWIRES):
        return "major_newswire"
    if any(d == m or d.endswith("." + m) for m in PRESS_RELEASE):
        return "company_press_release"
    if any(d == m or d.endswith("." + m) for m in SOCIAL):
        return "social_media"
    return "aggregator"


def _item(title: str, url: str, published_at, provider: str, summary: str = "", provider_sentiment=None, symbol=None) -> dict:
    dom = domain_of(url)
    return {"symbol": symbol, "title": title, "url": url, "published_at": published_at, "summary": summary or "",
            "source_domain": dom, "provider": provider, "sentiment": lexicon_sentiment(title),
            "provider_sentiment": provider_sentiment, "credibility": credibility_for(dom)}


def _json(body: str, who: str):
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ProviderError(f"{who}: {body[:160]!r}") from e


class GdeltNews:
    """GDELT DOC 2.0 article search (keyless; about one request per 5 seconds)."""

    URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, http: HttpClient):
        self.http = http

    @staticmethod
    def parse(body: str, symbol: str | None = None) -> list[dict]:
        data = _json(body, "gdelt")
        out = []
        for a in data.get("articles", []) or []:
            ts = to_ny_naive(pd.to_datetime(a["seendate"], format="%Y%m%dT%H%M%SZ", utc=True))
            out.append(_item(a.get("title", ""), a.get("url", ""), ts, "gdelt", symbol=symbol))
        return out

    def fetch(self, symbol: str, query: str | None = None, days: int = 30, limit: int = 50) -> list[dict]:
        q = query or f'"{symbol}" (stock OR shares OR earnings)'
        body, _ = self.http.get(self.URL, {"query": f"{q} sourcelang:english", "mode": "artlist", "format": "json",
                                           "maxrecords": limit, "timespan": f"{days}d", "sort": "datedesc"}, max_age_s=3600)
        return self.parse(body, symbol)


class FinnhubNews:
    URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self, http: HttpClient, api_key: str | None = None):
        self.http = http
        self.api_key = api_key if api_key is not None else os.environ.get("FINNHUB_API_KEY")

    @staticmethod
    def parse(body: str, symbol: str | None = None) -> list[dict]:
        data = _json(body, "finnhub")
        if isinstance(data, dict):
            raise ProviderError(f"finnhub: {data.get('error') or data}")
        out = []
        for a in data:
            ts = to_ny_naive(pd.to_datetime(a["datetime"], unit="s", utc=True))
            out.append(_item(a.get("headline", ""), a.get("url", ""), ts, "finnhub", a.get("summary", ""),
                             symbol=symbol or a.get("related")))
        return out

    def fetch(self, symbol: str, days: int = 30) -> list[dict]:
        if not self.api_key:
            raise ProviderError("finnhub: FINNHUB_API_KEY not set")
        today = pd.Timestamp.today().normalize()
        body, _ = self.http.get(self.URL, {"symbol": symbol, "from": (today - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                                           "to": today.strftime("%Y-%m-%d"), "token": self.api_key}, max_age_s=3600)
        return self.parse(body, symbol)


class MassiveNews:
    URL = "https://api.massive.com/v2/reference/news"
    SENT = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    def __init__(self, http: HttpClient, api_key: str | None = None):
        self.http = http
        self.api_key = api_key if api_key is not None else (os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY"))

    @classmethod
    def parse(cls, body: str, symbol: str) -> list[dict]:
        data = _json(body, "massive news")
        if data.get("status") not in ("OK", "DELAYED"):
            raise ProviderError(f"massive news: {data.get('error') or data.get('message') or data.get('status')}")
        out = []
        for a in data.get("results", []) or []:
            ps = next((cls.SENT.get(i.get("sentiment")) for i in a.get("insights", []) or [] if i.get("ticker") == symbol), None)
            ts = to_ny_naive(pd.to_datetime(a["published_utc"], utc=True))
            out.append(_item(a.get("title", ""), a.get("article_url", ""), ts, "massive", a.get("description", ""), ps, symbol))
        return out

    def fetch(self, symbol: str, limit: int = 50) -> list[dict]:
        params = {"ticker": symbol, "limit": limit, "order": "desc", "sort": "published_utc"}
        if self.api_key:
            params["apiKey"] = self.api_key
        body, _ = self.http.get(self.URL, params, max_age_s=3600)
        return self.parse(body, symbol)
