# Data sources: what BigBoss uses, what was verified, and recommendations

Everything below was tested from the development environment on 2026-10-22/23. "Verified" means data was downloaded,
validated, and (where possible) cross-checked against a second source.

## Current setup (recommended defaults)

| Need | Source | Status | Why |
|---|---|---|---|
| Long daily history, dividends, ETFs, indices, **India (NSE/BSE)** | **yfinance** (keyless) | verified | Back to 1999-2002 for most names; dividend-adjusted; covers everything FMP's free plan refuses |
| Independent cross-check / fallback for US prices | **FMP** (`FMP_API` or `FMP_API_KEY`) | verified | Official licensed API; dividend-adjusted series matched yfinance (daily-return corr ≥ 0.99996, identical 10y totals) |
| **Survivorship-free** US universe (incl. delisted names) | **Massive** grouped-daily (network secret) | verified | One call = the whole US market for a day; only source here that includes names after they delist |
| Point-in-time fundamentals and filing events | **SEC EDGAR** (keyless) | verified | Filing dates make them point-in-time; first-filed values ignore restatements |
| Macro, rates, FX, cross-asset proxies | **FRED** (keyless CSV), Treasury Fiscal Data, World Bank | verified | Long history, official |
| News | Massive news (with sentiment), GDELT (keyless) | verified | Research/agents only - not a model input |

Provider order is set in `configs/bigboss.toml` (`[data] price_providers`): `yfinance` first, then `fmp`, then `massive`.

## yfinance - can BigBoss use it? Yes, with caveats

**What worked:** SPY/AAPL/^GSPC from 2000, RELIANCE.NS/TCS.NS/^NSEI, all 25 configured US ETFs, all 10 configured NSE
large caps - in about a minute, with no key.

**Caveats you should know before relying on it:**
1. **Terms.** yfinance is an unofficial scraper, not affiliated with Yahoo. Yahoo's terms allow personal, non-commercial
   use only. Do not redistribute the data or build a commercial product on it.
2. **Fragility.** Yahoo changes endpoints and throttles shared IPs (the raw endpoint answered HTTP 429 here; yfinance's
   browser impersonation got through). Expect occasional breakage after yfinance or Yahoo updates. Pin a version and keep
   FMP as a fallback.
3. **No delisted tickers.** Yahoo drops most delisted symbols, so any universe built from it is survivorship-biased.
   Measured here: the bias was worth ~2.5%/yr for a US top-500 universe in 2025-26. Use Massive for point-in-time universes.
4. **Corporate-action errors, especially in India.** Validation caught, in Yahoo's data: an unadjusted Dec-2019 split
   and 247 zero prices in NIFTYBEES/BANKBEES/GOLDBEES, and mis-adjusted Reliance (2005 demerger), L&T (2004/2006) and TCS (2004)
   histories. BigBoss now **excludes any symbol whose latest validation has an ERROR** and refuses a broken benchmark.
   For India, start histories in 2007 and benchmark to ^NSEI (a price index: it excludes dividends, so it flatters
   excess returns by roughly 1.2-1.5%/yr).
5. **Fundamentals from yfinance are not point-in-time** (no filing dates), so BigBoss does not use them for modelling.

## Recommendations

**Keep (free):** yfinance + FMP + SEC EDGAR + FRED as above. That covers long-history research for US and India.

**For better evidence (the current bottleneck is history depth of the survivorship-free data):**
1. **Upgrade Massive to a plan with 10+ years of history.** This is the single most valuable upgrade: the only
   survivorship-free real test here covers 11 months out of sample, far too short to judge any strategy. The ingestion
   code already handles longer windows (`bigboss ingest --no-prices --massive-market 500`).
2. **If you trade Indian stocks seriously**, get an official NSE data feed or a broker API (e.g. a Zerodha Kite Connect
   subscription) for adjusted history and corporate actions. Yahoo's Indian data had the most errors of any source tested.
3. **Total-return benchmarks.** For India, NIFTY 50 TRI (from NSE/niftyindices) instead of the ^NSEI price index.
4. **Fix the Finnhub connector key** (it returns "Invalid API key") if you want analyst estimates and transcripts for the
   agents. It is optional: nothing in the models depends on it.

**Not recommended:** scraping websites without an API (RBI DBIE, SEBI), or using any provider's data for redistribution
without checking its license.

## Keys

Keys are read from environment variables (never config files or git): `FMP_API` / `FMP_API_KEY`, `MASSIVE_API_KEY`
(or a network secret), `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY`, `FRED_API_KEY`. Provider errors are redacted so keys
never reach logs. Don't paste keys into chats or commit them.
