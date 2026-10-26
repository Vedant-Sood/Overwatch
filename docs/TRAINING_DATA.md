# The `Vedant-Sood/Training-Data` repository: what it contains and what BigBoss uses

Checked 2026-10-23 at commit `0e14fc3` ("Track CSV files via Git LFS"). About 2.5 GB of CSVs in Git LFS, plus 534 MB of
zip files and the older Kaggle text files. Loader code: `bigboss/data/providers/local_files.py`.

| part | content | verdict |
|---|---|---|
| `Data/evenmoredata/ETF prices.csv` | 2,310 US ETFs, daily OHLCV, raw `close` and dividend-adjusted `adj_close`, 1993-01-29 → 2021-11-30 (3.87 M rows) | **Used.** Prices match yfinance (19 ETFs: median daily-return correlation 0.99996, median annual gap 0.7 bp, no day off by > 2%). **Survivors only**: ETFs closed before Nov 2021 (XIV, TVIX) are absent; some large ETFs (TLT, XLK) are missing |
| `Data/evenmoredata/ETFs.csv` | ETF metadata | **Category only.** `fund_category` checks out (EEM "Diversified Emerging Mkts", TQQQ "Trading--Leveraged Equity"), but names and fund families are misaligned (GLD is listed as an "American Century ... Value Fund", PPLT as an "American Beacon Large Cap Value Fund"); XLE, LQD, VNQ have no row |
| `zipped_datasets/*.zip` (25 zips) and `datasetsmore/` (400 files) | ~17,000 per-ticker CSVs, daily OHLCV, epoch timestamps, 2010-01-04 → 2026-10-22; split-adjusted, not dividend-adjusted; includes OTC names, warrants, units, rights; no ETFs | **Cross-checks only.** All 16,893 zipped series end in Oct 2026: only currently listed names. SIVB, TWTR, FRC, BBBY, CELG, ATVI, XLNX, PXD are absent, so any backtest on it is survivorship-biased |
| `Data/price_data/` | 569 current large caps, 2010 → 2026-10, dividend-adjusted | Cross-checks only (survivors) |
| `Data/stock_details_5_years.csv` (also in `Data/moredata/`) | Current S&P 500 members, 2018-11 → 2023 | Not used (survivors; DoltHub covers the period without the bias) |
| `Data/01-07_company_*.csv` | Snapshot for ~512 companies: info, news with sentiment, balance sheet, financials, filings, officers, cash flow | **Not used for modelling.** Statement values have period dates but no filing dates, so they cannot be used point-in-time (SEC EDGAR provides that). `07_company_cash_flow.csv` is an exact copy of `03_company_balance_sheet.csv` |
| `MutualFund prices - *.csv` | ~24,000 mutual funds, daily NAV, 75 M rows | Not used: NAV without distributions is not a total return, and funds trade once a day at NAV |
| `Stocks/`, `ETFs/`, `Data/Stocks`, `Data/ETFs` | The Kaggle "Huge Stock Market Dataset" (7,195 stocks, 1,344 ETFs, ending Nov 2017); `Data/` duplicates the top level | Not used: survivors as of Nov 2017 |

## How the ETF file is used

```bash
bigboss ingest --no-prices --no-sec --symbols SPY \
  --etf-file "Training-Data/Data/evenmoredata/ETF prices.csv" --etf-meta "Training-Data/Data/evenmoredata/ETFs.csv" \
  --etf-top 100 --etf-start 2003-01-01
```

* On each date the universe is the 100 ETFs with the highest trailing 60-day median dollar volume (raw close × volume,
  price ≥ $5, ≥ 40 sessions), from information before that date. SPY is the benchmark and is not ranked.
* Leveraged, inverse and volatility products are excluded (241 ETFs: category "Trading--*", or for uncategorized
  funds a beta to SPY above 1.6 or below −0.3). They decay by design and would add leverage to a long-only,
  no-leverage portfolio.
* Fund categories are used as sectors, so the 30% sector cap applies per category.
* Result: 352 ETFs were ever members; the median day had 38 members in 2003, 79 in 2006 and the full 100 from 2007.

## What would make this repository more useful

Data for names that **stopped trading**: per-ticker files for delisted stocks and closed ETFs, in the same formats.
That is the one thing no free source here provides before 2018.
