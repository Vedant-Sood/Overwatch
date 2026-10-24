# Data sources: what BigBoss uses, what was verified, and recommendations

Everything below was tested from the development environment on 2026-10-22/23. "Verified" means data was downloaded,
validated, and (where possible) cross-checked against a second source.

## Current setup (recommended defaults)

| Need | Source | Status | Why |
|---|---|---|---|
| Long daily history, dividends, ETFs, indices, **India (NSE/BSE)** | **yfinance** (keyless) | verified | Back to 1999-2002 for most names; dividend-adjusted; covers everything FMP's free plan refuses |
| Independent cross-check / fallback for US prices | **FMP** (`FMP_API` or `FMP_API_KEY`) | verified | Official licensed API; dividend-adjusted series matched yfinance (daily-return corr ≥ 0.99996, identical 10y totals) |
| **Survivorship-free** US universe, **2018-09 → today** | **DoltHub** `post-no-preference/stocks` (keyless, `dolt` CLI) | verified | Whole US market daily incl. names that later delisted; split/dividend tables; ~8 years |
| Survivorship-free US universe, last 2 years | **Massive** grouped-daily (network secret) | verified | One call = the whole US market for a day; free plan limited to 2 years |
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

## DoltHub `post-no-preference/stocks` (verified 2026-10-23)

Free, versioned Dolt database of US end-of-day bars, splits, dividends and the Nasdaq Trader symbol directory.
`bigboss dolthub-sync` clones it (4.5 GB) and exports Parquet tables (0.9 GB); `bigboss ingest --dolthub <export>`
builds the point-in-time universe. 29.4 M bars, 21,842 tickers, 2011-01-03 → 2026-10-22.

**What was verified**

| check | result |
|---|---|
| delisted names kept | yes from 2018: SIVB ends 2023-03-09, TWTR 2022-10-27, ATVI 2023-10-12, PXD 2024-05-02, FRC 2023-04-28 (all correct) |
| before 2018 | **not survivorship-free**: no ticker in `ohlcv` stops trading before 2018. EMC, LinkedIn, Yahoo, Time Warner are absent; pre-2018 bars are a backfill of names alive in 2018, and some (Celgene, Red Hat, Express Scripts) start only in 2018. BigBoss's universe therefore starts on **2018-09-07**, when daily full-market collection began |
| prices | RAW (unadjusted) closes; occasional missing sessions (e.g. GME 2021-02-02) |
| split table | starts 2014-03; **25 bogus/duplicate records** (e.g. GOOGL 20:1 on both 2022-06-30 and 07-18) and **24 missing splits/conversions** among universe members (TJX 2:1 2018, ODFL 3:2 2020, EW 3:1 2020, CP 5:1 2021, AZN ADR→share 2026, ...). BigBoss checks every recorded split against the raw price jump and verifies every ±30% raw move of a universe member against yfinance over the same span (`--verify-splits`); 681 were real moves, 24 were corrected, 174 had no independent data (mostly delisted names; left as is) |
| dividends | 23,199 applied; amounts are not consistently split-adjusted (AAPL Aug-2020 stored post-split); 7 implausible (> 25% yield) dropped. **Spin-offs are not in the tables** (e.g. MRK/Organon), which understates those holders' returns |
| total return vs yfinance (33 random current members + SPY/AAPL/JNJ, 2018-09 → 2026-10) | median daily-return correlation 0.9997, worst 0.9945; median annual total-return gap 0 bp, mean −7 bp/yr, worst −71 bp/yr (CBOE) |
| ticker reuse | handled by splitting series at gaps > 30 days (e.g. BBBY 2023 vs the 2025 re-listing, FB → META in 2022 becomes a sale and a later re-entry) |
| licence | none stated in the repository; personal research use |

**Alpha Vantage connector:** `LISTING_STATUS` (state=delisted) returns 9,599 delisted US stocks/ETFs with dates (thin
before 2015; bankruptcies like SIVB/BBBY/FRC missing). BigBoss uses it to classify names that DoltHub's symbol
directory (which starts in Oct 2017) never saw. Full daily price history (raw or adjusted) is premium on this plan.

**Kaggle "Huge Stock Market Dataset"** (not used): it ends in Nov 2017 and contains only names listed when it was
scraped, so it cannot fix the pre-2018 survivorship gap either.

## Historical index membership (verified 2026-10-24)

| dataset | content | use |
|---|---|---|
| `Vedant-Sood/sp500dataset` (= fja05680/sp500, MIT) | every S&P 500 composition since 1996-01-02 (2,720 snapshots, 1,209 tickers, 759 removed) | point-in-time S&P 500 universe (`bigboss ingest --sp500-history ... --sp500-prices <dolthub export> --kaggle-dir <Training-Data/Stocks>`). No prices: priced from DoltHub + the Kaggle 2017 set; member-day coverage 91% (2014) → 97.8% (2017) → ≥ 99.5% (2019+). Validated: equal weight of members tracks RSP |
| Historical Nifty 50 Constituent Weights (Bera 2025, CC BY-NC-SA 4.0, non-commercial) | NIFTY 50 constituents and weights, monthly + rebalance dates, 2008-01-31 → 2025-08-31 (100 stocks) | point-in-time NIFTY 50 universe (`--nifty-weights --nifty-sectors --nifty-plan`). Priced from yfinance and **NSE's archive (jugaad-data), which still serves delisted names** (Satyam, Ranbaxy, RCOM, ...): 100% weight coverage. Validated: rebuilt index vs ^NSEI daily correlation 0.9988 |

## Yahoo vs NSE exchange data (India cross-check)

`bigboss.data.crosscheck.compare_returns` compares Yahoo's adjusted closes with NSE's raw closes (via `jugaad-data`,
optional extra `pip install -e .[india]`). Days are classified as: corporate action (raw jump ≥ 15%), payout-like
(raw falls more than adjusted: dividends, small demergers - what a correct adjustment looks like), or **flagged**.
Results for the 10 configured NSE large caps, 2002 → 2026-10-23:

| symbol | daily-return corr (normal days) | corporate-action days | payout-like days | flagged days |
|---|---|---|---|---|
| RELIANCE | 0.9976 | 7 | 5 | 0 |
| TCS | 0.9986 | 5 | 2 | 0 |
| HDFCBANK | 0.9994 | 6 | 0 | 0 |
| INFY | 0.9985 | 9 | 3 | 0 |
| ICICIBANK | 0.9969 | 7 | 3 | 2: 2025-03-18/19 (Yahoo 0.0% then +3.5%; NSE +3.2% then +0.2%) |
| HINDUNILVR | 0.9978 | 2 | 3 | 0 |
| ITC | 0.9951 | 3 | 15 | 2: 2008-07-16 (+3.5% vs +1.2%), 2023-05-30 (+2.4% vs +0.2%) |
| SBIN | 0.9982 | 3 | 4 | 0 |
| BHARTIARTL | 0.9986 | 4 | 1 | 0 |
| LT | 0.9496 | 6 | 0 | 3: **2006-09-27 (Yahoo −50.6% vs NSE −1.1%)**, 2025-03-18/19 (same one-day lag as ICICI) |

Conclusions: Yahoo's Indian data agrees with the exchange on ordinary days and handles dividends correctly. The errors
are rare but large: a mis-applied L&T bonus adjustment in 2006 (already caught by validation, and outside the 2007+
window BigBoss uses for India) and a stale Yahoo bar on 2025-03-18 that shifts one day's return by a session. Run the
cross-check before trusting any new Indian symbol.

## Recommendations

**Keep (free):** yfinance + FMP + SEC EDGAR + FRED as above. That covers long-history research for US and India.

**For better evidence (the current bottleneck is history depth of the survivorship-free data):**
1. **For history before 2018 that includes delisted names**, a paid source is still needed (Massive 10+ year plans,
   Norgate, CRSP via a university). DoltHub now gives ~8 free survivorship-free years (2018-09 →); see
   RESULTS Part 1f.
2. **If you trade Indian stocks seriously**, get an official NSE data feed or a broker API (e.g. a Zerodha Kite Connect
   subscription) for adjusted history and corporate actions. Yahoo's Indian data had the most errors of any source tested.
3. **Total-return benchmarks.** For India, NIFTY 50 TRI (from NSE/niftyindices) instead of the ^NSEI price index.
4. **Fix the Finnhub connector key** (it returns "Invalid API key") if you want analyst estimates and transcripts for the
   agents. It is optional: nothing in the models depends on it.

5. **Independent engine check: QuantConnect (free plan).** QuantConnect's cloud backtester has survivorship-free US
   equity data that cannot be downloaded on the free plan, but BigBoss's decisions can be replayed there:
   `bigboss export-qc` + `bigboss qc-compare` (see `docs/QUANTCONNECT.md`).

**Not recommended:** scraping websites without an API (RBI DBIE, SEBI), or using any provider's data for redistribution
without checking its license.

## Keys

Keys are read from environment variables (never config files or git): `FMP_API` / `FMP_API_KEY`, `MASSIVE_API_KEY`
(or a network secret), `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY`, `FRED_API_KEY`. Provider errors are redacted so keys
never reach logs. Don't paste keys into chats or commit them.
